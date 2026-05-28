import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F




class ChannelPositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, S, D]
        x = x + self.pos_emb[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_channels=62,
        band_dim=30,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        out_dim=32,
    ):
        super().__init__()

        self.input_proj = nn.Linear(band_dim, d_model)

        self.pos_encoder = ChannelPositionalEncoding(
            seq_len=num_channels,
            d_model=d_model,
            dropout=dropout
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )


        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: [B, 62, 30]
        x = self.input_proj(x)  # [B, 62, d_model]

        # prepend CLS token
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)           # [B, 63, d_model]

        self.cls_pos_emb = nn.Parameter(torch.zeros(1, 1, x.size(-1))).to(x.device)
        nn.init.trunc_normal_(self.cls_pos_emb, std=0.02)

        # Add positional embedding
        x = x + torch.cat([self.cls_pos_emb, self.pos_encoder.pos_emb[:, :x.size(1)-1, :]], dim=1)

        x = self.pos_encoder.dropout(x)
        x = self.encoder(x)      # [B, 63, d_model]

        cls_out = x[:, 0, :]     # use CLS token representation
        # out = self.reg_head(cls_out)
        out = self.fc(cls_out)
        return out


class NeuralBehaviorEncoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
    ):
        super().__init__()
        # Encoder
        # Takes pose | condition (n * poses), neural (n * latent_size) as input
        input_size = frame_size * (num_future_predictions + num_condition_frames)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(frame_size + hidden_size, hidden_size)
        self.mu = nn.Linear(frame_size + hidden_size + latent_size , latent_size)
        self.logvar = nn.Linear(frame_size + hidden_size + latent_size , latent_size)

        self.neural_encoder = TransformerEncoder(out_dim=latent_size)  

    def encode(self, x, c, n):
        h1 = F.elu(self.fc1(torch.cat((x, c), dim=1)))
        h2 = F.elu(self.fc2(torch.cat((x, h1), dim=1)))
        n_embedding = self.neural_encoder(n)

        s = torch.cat((x, h2, n_embedding), dim=1)  
        return self.mu(s), self.logvar(s), n_embedding

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c, n):
        mu, logvar, n_embedding = self.encode(x, c, n)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar, n_embedding


class NeuralBehaviorMixedDecoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
        num_experts,
    ):
        super().__init__()

        input_size = latent_size + (frame_size+latent_size) * num_condition_frames
        inter_size = latent_size + hidden_size
        output_size = num_future_predictions * frame_size
        self.decoder_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.decoder_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 64
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, num_experts),
        )

    def forward(self, z, c, n):
        coefficients = F.softmax(self.gate(torch.cat((z, c, n), dim=1)), dim=1)
        layer_out = torch.cat((c, n), dim=1)

        for (weight, bias, activation) in self.decoder_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )

            input = torch.cat((z, layer_out), dim=1).unsqueeze(1)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out


class NeuralBehaviorMixtureVAE(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_condition_frames,
        num_future_predictions,
        normalization,
        num_experts,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self.mode = normalization.get("mode")
        self.data_max = normalization.get("max")
        self.data_min = normalization.get("min")
        self.data_avg = normalization.get("avg")
        self.data_std = normalization.get("std")

        hidden_size = 256
        args = (
            frame_size,
            latent_size,
            hidden_size,
            num_condition_frames,
            num_future_predictions,
        )

        self.encoder = NeuralBehaviorEncoder(*args)
        self.decoder = NeuralBehaviorMixedDecoder(*args, num_experts)


    def normalize(self, t):
        if self.mode == "minmax":
            return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t):
        if self.mode == "minmax":
            return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
        elif self.mode == "zscore":
            return t * self.data_std + self.data_avg
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def encode(self, x, c, n):
        _, mu, logvar, n_embedding = self.encoder(x, c, n)
        return mu, logvar

    def forward(self, x, c, n):
        z, mu, logvar, n_embedding = self.encoder(x, c, n)
        return self.decoder(z, c, n_embedding), mu, logvar

    def sample(self, z, c, n, deterministic=False):
        n_embedding = self.encoder.neural_encoder(n)
        return self.decoder(z, c, n_embedding)


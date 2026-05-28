import os
import time
from types import SimpleNamespace

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from common.logging_utils import CSVLogger
from common.misc_utils import update_linear_schedule
from vae_motion.models import NeuralBehaviorMixtureVAE
from tqdm import tqdm
from datetime import datetime
from scipy.io import savemat, loadmat

class StatsLogger:
    def __init__(self, args, csv_path):
        self.start = time.time()
        self.logger = CSVLogger(log_path=csv_path)
        self.num_epochs = args.num_epochs
        self.progress_format = None

    def time_since(self, ep):
        now = time.time()
        elapsed = now - self.start
        estimated = elapsed * self.num_epochs / ep
        remaining = estimated - elapsed

        em, es = divmod(elapsed, 60)
        rm, rs = divmod(remaining, 60)

        if self.progress_format is None:
            time_format = "%{:d}dm %02ds".format(int(np.log10(rm) + 1))
            perc_format = "%{:d}d %5.1f%%".format(int(np.log10(self.num_epochs) + 1))
            self.progress_format = f"{time_format} (- {time_format}) ({perc_format})"

        return self.progress_format % (em, es, rm, rs, ep, ep / self.num_epochs * 100)

    def log_stats(self, data):
        self.logger.log_epoch(data)

        ep = data["epoch"]
        ep_recon_loss = data["ep_recon_loss"]
        ep_kl_loss = data["ep_kl_loss"]
        ep_perplexity = data["ep_perplexity"]

        print(
            "{} | Recon: {:.3e} | KL: {:.3e} | PP: {:.3e}".format(
                self.time_since(ep), ep_recon_loss, ep_kl_loss, ep_perplexity
            ),
            flush=True,
        )


def feed_vae(pose_vae, ground_truth, condition, neural, future_weights):
    condition = condition.flatten(start_dim=1, end_dim=2)
    flattened_truth = ground_truth.flatten(start_dim=1, end_dim=2)
    neural = neural.flatten(start_dim=1, end_dim=2)

    output_shape = (-1, pose_vae.num_future_predictions, pose_vae.frame_size)

    # PoseVAE and PoseMixtureVAE
    vae_output, mu, logvar = pose_vae(flattened_truth, condition, neural)
    vae_output = vae_output.view(output_shape)

    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum().clamp(max=0)
    kl_loss /= logvar.numel()


    recon_err = torch.nn.functional.smooth_l1_loss(vae_output, ground_truth, reduction='none', beta=1.0)
    recon_loss = (recon_err.mean(dim=(0, -1)) * future_weights).sum()


    if torch.isnan(kl_loss):
        print(f"NaN detected in kl_loss! mu_max: {mu.max()}, logvar_max: {logvar.max()}")
        import pdb; pdb.set_trace()

    if torch.isnan(recon_loss):
        print(f"NaN detected in recon_loss! vae_output_max: {vae_output.max()}")
        import pdb; pdb.set_trace()

    return (vae_output, mu, logvar), (recon_loss, kl_loss)


def main():
    env_path = os.path.join(parent_dir, "environments")

    # setup parameters
    args = SimpleNamespace(
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        mocap_file=os.path.join(env_path, "mocap.npz"),
        norm_mode="zscore",
        latent_size=16,
        num_embeddings=12,
        num_experts=6,
        num_condition_frames=1,
        num_future_predictions=1,
        num_steps_per_rollout=8,
        kl_beta=0.2,
        load_saved_model=False,
    )

    # learning parameters
    teacher_epochs = 30
    ramping_epochs = 30
    student_epochs = 100
    args.num_epochs = teacher_epochs + ramping_epochs + student_epochs
    args.mini_batch_size = 64
    args.initial_lr = 1e-4
    args.final_lr = 1e-7

    raw_data = np.load(args.mocap_file)
    mocap_data_raw = torch.from_numpy(raw_data["data"]).float().to(args.device)
    end_indices = raw_data["end_indices"]

    max = mocap_data_raw.max(dim=0)[0]
    min = mocap_data_raw.min(dim=0)[0]
    avg = mocap_data_raw.mean(dim=0)
    std = mocap_data_raw.std(dim=0)

    # Make sure we don't divide by 0
    std[std == 0] = 1.0

    normalization = {
        "mode": args.norm_mode,
        "max": max,
        "min": min,
        "avg": avg,
        "std": std,
    }

    if args.norm_mode == "minmax":
        mocap_data_raw = 2 * (mocap_data_raw - min) / (max - min) - 1

    elif args.norm_mode == "zscore":
        mocap_data_raw = (mocap_data_raw - avg) / std


    mocap_data = mocap_data_raw[3:17999-6000,:]  # align with ecog
    print(end_indices)
    end_indices = np.array([17995-6000])

    neural_feature_raw = loadmat(rf"environments\ecog_feature.mat")['features']
    neural_feature_raw = np.transpose(neural_feature_raw, [2,0,1])  # (17996, 62, 30)

    neural_feature = neural_feature_raw[:11996]
    print(f"Neural feature shape: {neural_feature.shape}")
    print(f"Mocap data shape: {mocap_data.shape}")
    mean_n = neural_feature.mean(axis=0, keepdims=True)   # (1,62,30)
    std_n  = neural_feature.std(axis=0, keepdims=True) + 1e-8
    neural_feature = (neural_feature - mean_n) / std_n
    neural_feature  = (neural_feature  - std_n) / std_n

    test_mocap_data = mocap_data_raw[17999-6000:17999,:]
    test_neural_feature = neural_feature_raw[11996:,:,:]
    print(f"test_mocap_data feature shape: {test_mocap_data.shape}")
    print(f"test_neural_feature data shape: {test_neural_feature.shape}")

    batch_size = mocap_data.size()[0]
    frame_size = mocap_data.size()[1]
    chn_size = neural_feature.shape[1]
    band_size = neural_feature.shape[2]

    bad_indices = np.sort(
        np.concatenate(
            [
                end_indices - i
                for i in range(
                    args.num_steps_per_rollout
                    + (args.num_condition_frames - 1)
                    + (args.num_future_predictions - 1)
                )
            ]
        )
    )
    all_indices = np.arange(batch_size)
    good_masks = ~np.isin(all_indices, bad_indices)
    selectable_indices = all_indices[good_masks]

    pose_vae = NeuralBehaviorMixtureVAE(
        frame_size,
        args.latent_size,
        args.num_condition_frames,
        args.num_future_predictions,
        normalization,
        args.num_experts,
    ).to(args.device)

    pose_vae_path = "pose_neural_vae_c{}_n{}_l{}_e{}.pt".format(
        args.num_condition_frames, args.num_embeddings, args.latent_size, args.num_epochs
    )

    log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
    prefix = f'{log_time}_train_beta_m1s1_chn_{str(args.kl_beta)}_'
    pose_vae_path = prefix + pose_vae_path
    print(pose_vae_path)

    if args.load_saved_model:
        pose_vae = torch.load(pose_vae_path, map_location=args.device)
    pose_vae.train()

    vae_optimizer = optim.Adam(pose_vae.parameters(), lr=args.initial_lr)

    sample_schedule = torch.cat(
        (
            # First part is pure teacher forcing
            torch.zeros(teacher_epochs),
            # Second part with schedule sampling
            torch.linspace(0.0, 1.0, ramping_epochs),
            # last part is pure student
            torch.ones(student_epochs),
        )
    )


    future_weights = (
        torch.ones(args.num_future_predictions)
        .to(args.device)
        .div_(args.num_future_predictions)
    )

    shape = (args.mini_batch_size, args.num_condition_frames, frame_size)
    history = torch.empty(shape).to(args.device)
    shape_n = (args.mini_batch_size, args.num_condition_frames, chn_size, band_size)

    log_path = os.path.join(current_dir, f"./log/log_posevae_progress_{log_time}")
    logger = StatsLogger(args, csv_path=log_path)


    test_ep_recon_loss_list = []
    test_ep_kl_loss_list = []
    best_test_recon_loss = float('inf')
    for ep in range(1, args.num_epochs + 1):
        sampler = BatchSampler(
            SubsetRandomSampler(selectable_indices),
            args.mini_batch_size,
            drop_last=True,
        )
        ep_recon_loss = 0
        ep_kl_loss = 0
        ep_perplexity = 0

        update_linear_schedule(
            vae_optimizer, ep - 1, args.num_epochs, args.initial_lr, args.final_lr
        )

        num_mini_batch = 1
        for num_mini_batch, indices in enumerate(sampler):
            t_indices = torch.LongTensor(indices)


            condition_range = (
                t_indices.repeat((args.num_condition_frames, 1)).t()
                + torch.arange(args.num_condition_frames - 1, -1, -1).long()
            )

            t_indices += args.num_condition_frames
            history[:, : args.num_condition_frames].copy_(mocap_data[condition_range])

            for offset in range(args.num_steps_per_rollout): 
                # dims: (num_parallel, num_window, feature_size)
                use_student = torch.rand(1) < sample_schedule[ep - 1]

                prediction_range = (
                    t_indices.repeat((args.num_future_predictions, 1)).t()
                    + torch.arange(offset, offset + args.num_future_predictions).long()
                )
                ground_truth = mocap_data[prediction_range]
                condition = history[:, : args.num_condition_frames]
                neural = neural_feature[prediction_range]

                (vae_output, _, _), (recon_loss, kl_loss) = feed_vae(
                    pose_vae, ground_truth, condition, neural, future_weights    # ground_truth: x_t, condition: x_{t-1}
                )

                history = history.roll(1, dims=1)
                next_frame = vae_output[:, 0] if use_student else ground_truth[:, 0]
                history[:, 0].copy_(next_frame.detach())

                vae_optimizer.zero_grad()
                (recon_loss + args.kl_beta * kl_loss).backward()
                torch.nn.utils.clip_grad_norm_(pose_vae.parameters(), max_norm=1.0)

                vae_optimizer.step()

                ep_recon_loss += float(recon_loss.detach()) / args.num_steps_per_rollout
                ep_kl_loss += float(kl_loss.detach()) / args.num_steps_per_rollout


        avg_ep_recon_loss = ep_recon_loss / num_mini_batch
        avg_ep_kl_loss = ep_kl_loss / num_mini_batch
        avg_ep_perplexity = ep_perplexity / num_mini_batch

        logger.log_stats(
            {
                "epoch": ep,
                "ep_recon_loss": avg_ep_recon_loss,
                "ep_kl_loss": avg_ep_kl_loss,
                "ep_perplexity": avg_ep_perplexity,
            }
        )

        if ep % 10 == 0:
            pose_vae.eval()
            offset = 0
            prediction_range = (
                t_indices.repeat((args.num_future_predictions, 1)).t()
                + torch.arange(offset, offset + args.num_future_predictions).long()
            )
            test_ground_truth = test_mocap_data.unsqueeze(dim=1)
            test_condition = test_ground_truth[:-1,:,:]  #p_{t-1}
            test_ground_truth = test_ground_truth[1:,:,:]  # p_t
            test_neural = test_neural_feature.unsqueeze(dim=1)
            test_neural = test_neural[1:,:,:,:]  # n_t
            
            (_, _, _), (test_ep_recon_loss, test_ep_kl_loss) = feed_vae(
                pose_vae, test_ground_truth, test_condition, test_neural, future_weights    # ground_truth: x_t, condition: x_{t-1}
            )
            test_ep_recon_loss = float(test_ep_recon_loss.detach()) 
            test_ep_kl_loss = float(test_ep_kl_loss.detach()) 
            print(f"Evaluation of Epoch {ep} | Recon Loss: {test_ep_recon_loss:.6f} | KL Loss: {test_ep_kl_loss:.6f}")
            test_ep_recon_loss_list.append(test_ep_recon_loss)
            test_ep_kl_loss_list.append(test_ep_kl_loss)

            if test_ep_recon_loss < best_test_recon_loss:
                best_test_recon_loss = test_ep_recon_loss
                pose_vae_cpu = pose_vae.to('cpu')
                torch.save(pose_vae_cpu, 'BEST_'+pose_vae_path)
                pose_vae.to(args.device)
           

    pose_vae_cpu = pose_vae.to('cpu')
    torch.save(pose_vae_cpu, pose_vae_path)
    pose_vae.to(args.device)
    print('test_ep_recon_loss_list:', test_ep_recon_loss_list )
    print('test_ep_kl_loss_list:', test_ep_kl_loss_list )


if __name__ == "__main__":
    main()

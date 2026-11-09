import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler
from tqdm import tqdm
from neural_behavior.data import load_motion, load_neural, load_labels, prepare_neural, valid_starts, motion_normalization
from neural_behavior.checkpoints import build_model, save_checkpoint


PRESETS = ('behavior', 'neural', 'label')


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=PRESETS, required=True)
    parser.add_argument('--data-dir', type=Path, default=root / 'environments')
    parser.add_argument('--neural-file', type=Path, help='Optional neural feature MAT file; defaults to environments/neural_feature.mat.')
    parser.add_argument('--labels', type=Path, help='Integer label array aligned with all mocap rows.')
    parser.add_argument('--neural-normalization', choices=['zscore', 'legacy-double'], default='legacy-double')
    parser.add_argument('--motion-offset', type=int, default=3)
    parser.add_argument('--motion-stop', type=int, default=17999)
    parser.add_argument('--test-splits', type=int, default=6000)
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--teacher-epochs', type=int)
    parser.add_argument('--ramping-epochs', type=int)
    parser.add_argument('--student-epochs', type=int)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--rollout', type=int, default=8)
    parser.add_argument('--latent-size', type=int, default=16)
    parser.add_argument('--num-experts', type=int, default=6)
    parser.add_argument('--num-labels', type=int, default=16)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--initial-lr', type=float, default=1e-4)
    parser.add_argument('--final-lr', type=float, default=1e-7)
    parser.add_argument('--output-dir', type=Path, default=root / 'outputs')
    args = parser.parse_args()
    if args.experiment == 'label' and args.labels is None:
        parser.error('label requires --labels; annotations are not included in the release.')
    if min(args.batch_size, args.rollout, args.latent_size, args.num_experts, args.num_labels) < 1:
        parser.error('Batch size, rollout and model dimensions must be positive.')
    if args.test_splits < 1 or args.motion_offset < 0:
        parser.error('Invalid split parameters.')
    defaults = (15, 15, 50) if args.experiment == 'label' else (30, 30, 100)
    for key, default in zip(('teacher_epochs', 'ramping_epochs', 'student_epochs'), defaults):
        if getattr(args, key) is None:
            setattr(args, key, default)
    if min(args.teacher_epochs, args.ramping_epochs, args.student_epochs) < 0 or sum((args.teacher_epochs, args.ramping_epochs, args.student_epochs)) < 1:
        parser.error('Epoch counts must be nonnegative and sum to a positive number.')
    return args


def feed_vae(model, truth, condition, auxiliary=None):
    import torch
    inputs = [truth.flatten(1), condition.flatten(1)]
    if auxiliary is not None:
        inputs.append(auxiliary.flatten(1, 2))
    output, mu, logvar = model(*inputs)
    output = output.view_as(truth)
    recon = torch.nn.functional.smooth_l1_loss(output, truth, beta=1.0)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum().clamp(max=0) / logvar.numel()
    if not torch.isfinite(recon + kl):
        raise FloatingPointError('Non-finite reconstruction or KL loss.')
    return output, recon, kl


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    raw, ends = load_motion(args.data_dir / 'kinematic.npz')
    start, stop = args.motion_offset, args.motion_stop
    split = stop - args.test_splits
    if not (0 <= start < split <= stop <= len(raw)) or split - start <= args.rollout:
        raise ValueError('Data are too short for the requested split/rollout; check alignment options.')
    normalization = motion_normalization(raw)
    normalized = (raw - normalization['avg']) / normalization['std']
    motion = normalized[start:split].to(device)
    auxiliary = None
    kind = 'behavior'
    if args.experiment == 'neural':
        kind = 'neural'
        neural = load_neural(args.neural_file or args.data_dir / 'neural_feature.mat')
        neural = prepare_neural(neural, args.neural_normalization)
        if len(neural) < stop - start:
            raise ValueError('ECoG does not cover the aligned motion window.')
        neural = neural[:stop-start]
        auxiliary = neural[:split-start].to(device)
    elif args.experiment == 'label':
        kind = 'label'
        labels = load_labels(args.labels, args.num_labels)
        if len(labels) < stop:
            raise ValueError('Labels must cover all requested mocap rows, including held-out rows.')
        auxiliary = labels[start:split].to(device)

    config = dict(model=kind, frame_size=motion.shape[1], latent_size=args.latent_size,
                  num_experts=args.num_experts, num_labels=args.num_labels)
    device_normalization = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in normalization.items()}
    model = build_model(config, device_normalization).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)
    schedule = torch.cat((torch.zeros(args.teacher_epochs),
                          torch.linspace(0, 1, args.ramping_epochs),
                          torch.ones(args.student_epochs)))
    train_ends = ends[(ends >= start) & (ends < split)] - start
    starts = valid_starts(len(motion), train_ends, args.rollout)
    if len(starts) < args.batch_size:
        raise ValueError('Not enough valid training windows for one full batch.')
    run = args.output_dir / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.experiment)
    run.mkdir(parents=True, exist_ok=False)
    (run / 'config.json').write_text(json.dumps(vars(args), default=str, indent=2), encoding='utf-8')
    print(f'Output: {run}\nTrain rows: [{start}:{split}], held-out rows: [{split}:{stop}]')
    with (run / 'loss.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'reconstruction', 'kl'])
        progress = tqdm(enumerate(schedule, 1), total=len(schedule),
                        desc='Training', unit='epoch', dynamic_ncols=True)
        for epoch, probability in progress:
            model.train()
            lr = args.initial_lr - (args.initial_lr - args.final_lr) * (epoch-1) / len(schedule)
            for group in optimizer.param_groups:
                group['lr'] = lr
            total_recon = total_kl = count = 0
            sampler = BatchSampler(SubsetRandomSampler(starts), args.batch_size, drop_last=True)
            for indices in sampler:
                indices = torch.as_tensor(indices, device=device)
                history = motion[indices].unsqueeze(1).clone()
                for offset in range(1, args.rollout + 1):
                    target = indices + offset
                    truth = motion[target].unsqueeze(1)
                    aux = auxiliary[target].unsqueeze(1) if auxiliary is not None else None
                    use_student = torch.rand(1).item() < probability.item()
                    output, recon, kl = feed_vae(model, truth, history, aux)
                    history = (output if use_student else truth).detach().clone()
                    optimizer.zero_grad()
                    (recon + args.beta * kl).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_recon += recon.item()
                    total_kl += kl.item()
                    count += 1
            writer.writerow([epoch, total_recon/count, total_kl/count])
            stream.flush()
            save_checkpoint(run / 'last.pt', model)
            progress.set_postfix(Recon=f'{total_recon/count:.6g}',
                                 KL=f'{total_kl/count:.6g}', refresh=False)


if __name__ == '__main__':
    main(parse_args())

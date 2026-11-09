import argparse
import json
import pickle
from pathlib import Path


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['gt', 'lstm', 'behavior', 'neural', 'neural-v1', 'label'], required=True)
    parser.add_argument('--vae', type=Path, help='Trusted full-model checkpoint for model-based modes.')
    parser.add_argument('--legacy-checkpoint', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--neural-normalization', choices=['zscore', 'legacy-double'], default='legacy-double')
    parser.add_argument('--data-dir', type=Path, default=root / 'environments')
    parser.add_argument('--testframe', type=int, required=True, help='Original experiment frame index (initial state).')
    parser.add_argument('--savejoint', type=int, choices=[0, 1], default=0)
    parser.add_argument('--steps', '--len', dest='steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--initial-pose', type=Path, help='Explicit initial condition; supply together with --initial-state.')
    parser.add_argument('--initial-state', type=Path, help='Explicit root state; supply together with --initial-pose.')
    parser.add_argument('--neural-file', type=Path)
    parser.add_argument('--motion-offset', type=int, default=3, help='Motion/ECoG alignment offset (default: 3).')
    parser.add_argument('--label-id', type=int, help='Fixed label ID for the whole rollout, as in the original label player.')
    parser.add_argument('--labels', type=Path, help='Integer labels aligned with mocap, or trusted legacy frame->label dictionary.')
    parser.add_argument('--legacy-labels', action='store_true', help='Allow a trusted NumPy pickled label dictionary.')
    parser.add_argument('--lstm-file', type=Path, help='Precomputed joints; no LSTM inference is performed.')
    parser.add_argument('--lstm-start-frame', type=int, help='Frame index represented by the first LSTM prediction.')
    parser.add_argument('--lstm-scale', type=float, default=1.0, help='Coordinate divisor; legacy rendered NPZ uses 1.')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--gui', type=int, choices=[0, 1], default=1)
    parser.add_argument('--track', type=int, choices=[0, 1], default=1,
                        help='Use the original paper camera target [3, 1, 3].')
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--save', action='store_true', help='Save rendered PNG frames.')
    parser.add_argument('--csv', type=Path, help='Write generated 151-D feature rows, for modes other than LSTM.')
    parser.add_argument('--output-dir', type=Path, default=root / 'outputs' / 'play')
    args = parser.parse_args()
    if args.steps < 1 or args.testframe < 0 or args.fps <= 0 or args.lstm_scale <= 0:
        parser.error('Invalid frame, step count, FPS or coordinate scale.')
    if args.mode not in {'gt', 'lstm'} and args.vae is None:
        parser.error('This mode requires --vae.')
    if args.mode == 'lstm' and (args.lstm_file is None or args.lstm_start_frame is None):
        parser.error('LSTM playback requires --lstm-file and --lstm-start-frame.')
    if args.mode == 'label' and args.label_id is None and args.labels is None:
        parser.error('Label playback requires --label-id or --labels.')
    if args.mode == 'lstm' and args.csv:
        parser.error('LSTM joint predictions do not contain 151-D pose features.')
    if args.headless and args.save:
        parser.error('--save needs the renderer; remove --headless.')
    if (args.initial_pose is None) != (args.initial_state is None):
        parser.error('Supply --initial-pose and --initial-state together.')
    if args.initial_pose is None and args.testframe < 1:
        parser.error('Automatic initialization requires --testframe >= 1 (uses testframe - 1).')
    return args


def prepare_initial_files(args, run_dir):
    """Use the author's exact previous-frame slices, without overwriting data."""
    import numpy as np
    if args.initial_pose is not None:
        return args.initial_pose, args.initial_state
    source = args.data_dir / 'pose_presentation.pkl'
    index = args.testframe - 1
    if index < 0:
        raise ValueError('Automatic initialization requires testframe >= 1.')
    with source.open('rb') as stream:
        representation = pickle.load(stream)
    if not isinstance(representation, (list, tuple)) or len(representation) != 6:
        raise ValueError('Expected [eulers, root_eulers, Proot_lower_list, bone_length, body_part, Rroot_lower_list].')
    roots = np.asarray(representation[1])
    positions = np.asarray(representation[2])
    with np.load(args.data_dir / 'kinematic.npz', allow_pickle=False) as archive:
        motion = archive['data']
    if (roots.ndim != 2 or roots.shape[1] < 2
            or positions.ndim != 2 or positions.shape[1] != 3
            or motion.ndim != 2 or motion.shape[1] != 151):
        raise ValueError('Invalid root angles, root positions or mocap dimensions.')
    if index >= min(len(roots), len(positions)) or args.testframe >= len(motion):
        raise ValueError('testframe is outside the initialization/target data range.')

    pose_index = args.testframe - 1
    root_index = args.testframe

    pose = motion[pose_index:pose_index + 1]
    facing = roots[root_index:root_index + 1, 0] % (2 * np.pi)
    pitching = roots[root_index:root_index + 1, 1] % (2 * np.pi)
    xyz = positions[root_index:root_index + 1, :]

    if not all(np.isfinite(value).all() for value in (pose, facing, pitching, xyz)):
        raise ValueError('Initialization contains non-finite values.')
    run_dir.mkdir(parents=True, exist_ok=True)
    pose_path = run_dir / 'pose0_testdataset.npy'
    state_path = run_dir / 'init_state_testdataset.npz'
    np.save(pose_path, pose)
    np.savez(state_path, root_facing=facing, root_pitching=pitching, root_xyz=xyz)
    print(f'Initialized from mocap/root row {index}; first target mocap row {args.testframe}.')
    return pose_path, state_path


def load_lstm(args):
    import numpy as np
    with np.load(args.lstm_file, allow_pickle=False) as archive:
        key = 'predict_joint_xyzs' if 'predict_joint_xyzs' in archive else 'arr_0'
        joints = archive[key]
    if joints.ndim == 3 and joints.shape[1:] == (3, 15):
        joints = joints.transpose(0, 2, 1)
    if joints.ndim != 3 or joints.shape[1:] != (15, 3) or not np.isfinite(joints).all():
        raise ValueError('LSTM joints must be finite with shape (T, 15, 3) or (T, 3, 15).')
    start = args.testframe - args.lstm_start_frame
    if start < 0 or start + args.steps > len(joints):
        raise ValueError('Requested frames are outside the LSTM prediction file.')
    return joints[start:start+args.steps] / args.lstm_scale


def save_joints(args, records):
    import numpy as np
    names = {'gt': 'GT_GT', 'behavior': 'BOnly_ContinuePrediction',
             'neural': 'NB_ContinuesPrediction', 'neural-v1': 'NB_ContinuesPrediction',
             'label': 'LB_ContinuesPrediction',
             'lstm': 'LSTM_ContinuesPrediction'}
    joint_only = args.mode == 'lstm'
    suffix = '.npz' if joint_only else '.pkl'
    prefix = f'Init_{args.testframe}_{names[args.mode]}_Step_{len(records)}_Sample_'
    sample = 0
    while True:
        path = args.output_dir / f'{prefix}{sample}{suffix}'
        try:
            with path.open('xb') as stream:
                if joint_only:
                    np.savez(stream, predict_joint_xyzs=np.concatenate([r[0] for r in records]))
                else:
                    pickle.dump(records, stream)
            return path
        except FileExistsError:
            sample += 1


def main(args):
    import numpy as np
    import torch
    from neural_behavior.data import load_motion, load_neural, prepare_neural, load_labels, motion_normalization
    from neural_behavior.checkpoints import load_checkpoint
    from neural_behavior.models import PoseMixtureVAE, NeuralBehaviorMixtureVAE, BehaviorLabelMixtureVAE
    from environments.playback import PoseState, Viewer

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = None
    if args.mode not in {'gt', 'lstm'}:
        raw_motion, _ = load_motion(args.data_dir / 'kinematic.npz')
        model = load_checkpoint(args.vae, motion_normalization(raw_motion))
        expected = BehaviorLabelMixtureVAE if args.mode == 'label' else (PoseMixtureVAE if args.mode == 'behavior' else NeuralBehaviorMixtureVAE)
        if not isinstance(model, expected):
            raise ValueError('Checkpoint architecture does not match --mode.')
        if model.num_condition_frames != 1 or model.num_future_predictions != 1 or model.frame_size != 151:
            raise ValueError('These experiment players require one condition frame and one 151-D predicted frame.')
    preprocessing = {}
    offset = args.motion_offset
    if offset < 0:
        raise ValueError('motion-offset cannot be negative.')
    neural = label = motion = lstm = None
    if args.mode.startswith('neural'):
        neural = load_neural(args.neural_file or args.data_dir / 'neural_feature.mat')
        encoder = model.encoder.neural_encoder
        expected = (encoder.pos_encoder.pos_emb.shape[1], encoder.input_proj.in_features)
        if tuple(neural.shape[1:]) != expected:
            raise ValueError(f'Neural dimensions {tuple(neural.shape[1:])} do not match model {expected}.')
        preprocessing = dict(neural_normalization=args.neural_normalization,
                             neural_stats_frames=len(neural), neural_normalization_scope='all')
        neural = prepare_neural(neural, args.neural_normalization)
        first = args.testframe - offset
        if first < 0 or first + args.steps > len(neural):
            raise ValueError('Requested rollout exceeds the ECoG time range.')
        neural = neural[first:first+args.steps]
    elif args.mode == 'label':
        num_labels = model.encoder.mu.in_features - model.frame_size - model.encoder.fc1.out_features
        label_id = args.label_id
        if label_id is None:
            if args.legacy_labels:
                labels = np.load(args.labels, allow_pickle=True).item()
                label_id = int(labels[args.testframe])
            else:
                labels = load_labels(args.labels, num_labels)
                if args.testframe >= len(labels):
                    raise ValueError('testframe is outside the label array.')
                label_id = int(labels[args.testframe].argmax())
        if not 0 <= label_id < num_labels:
            raise ValueError(f'Label ID must be in [0, {num_labels-1}].')
        label = torch.nn.functional.one_hot(torch.tensor([label_id]), num_labels).float()
    elif args.mode == 'gt':
        motion, _ = load_motion(args.data_dir / 'kinematic.npz')
        if args.testframe + args.steps > len(motion):
            raise ValueError('Requested rollout exceeds the mocap time range.')
    else:
        if args.mode == 'lstm':
            lstm = load_lstm(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.mode
    frames_dir = args.output_dir / run_id
    initial_pose, initial_state = prepare_initial_files(args, frames_dir)
    state = PoseState(initial_pose, initial_state, args.data_dir / 'body_info.npz')
    if args.save:
        frames_dir.mkdir(exist_ok=True)
    from types import SimpleNamespace
    session = SimpleNamespace(mocap_data=state.pose.copy(), num_condition_frames=1,
                              data_fps=args.fps, debug_frame_index=-1, frame_skip=1,
                              pose_vae_model=SimpleNamespace(num_future_predictions=1),
                              reset_requested=False)

    def request_reset():
        session.reset_requested = True

    session.reset = request_reset
    viewer = None if args.headless else Viewer(
        state.body_part, args.fps, bool(args.track), bool(args.gui),
        env=session, output_dir=frames_dir)
    records, features = [], []
    # Original BehaviorOnly averaged two independent N(0,I) latent vectors:
    # base_action from the observation and a fresh action, giving variance 1/2.
    base_action = torch.randn(1, model.latent_size) if args.mode == 'behavior' else None
    step = rendered_frames = reset_count = 0
    try:
        with torch.no_grad():
            while step < args.steps:
                if session.reset_requested:
                    state = PoseState(initial_pose, initial_state,
                                      args.data_dir / 'body_info.npz')
                    step = 0
                    records.clear()
                    features.clear()
                    reset_count += 1
                    session.reset_requested = False
                    if args.mode == 'behavior':
                        base_action = torch.randn(1, model.latent_size)
                if args.mode == 'lstm':
                    record = [lstm[step:step+1], None, None]
                else:
                    if args.mode == 'gt':
                        pose = motion[args.testframe+step].numpy()
                    else:
                        condition = model.normalize(torch.from_numpy(state.pose))
                        z = torch.randn(1, model.latent_size)
                        if args.mode == 'behavior':
                            output = model.sample((base_action + z)/2, condition)
                            base_action = torch.randn_like(base_action)
                        else:
                            auxiliary = label if args.mode == 'label' else neural[step:step+1]
                            output = model.sample(z, condition, auxiliary)
                        pose = model.denormalize(output).numpy()
                        if viewer:
                            alpha = viewer.controller_autonomy
                            pose = alpha * pose + (1 - alpha) * session.mocap_data
                    if not np.isfinite(pose).all():
                        raise FloatingPointError('Non-finite generated pose.')
                    record = list(state.advance(pose))
                    features.append(np.asarray(pose).reshape(151))
                records.append(record)
                if viewer:
                    if not viewer.render(record[0], root_xyzs=state.root / 20 if args.mode != 'lstm' else None,
                                         action=z if model is not None else None):
                        break
                    if args.save:
                        from imageio.v2 import imwrite
                        imwrite(frames_dir / f'frame_{rendered_frames:04d}.png', viewer.camera.dump_rgb_array())
                rendered_frames += 1
                step += 1
                if session.reset_requested:
                    step = 0

                import time
                time.sleep(1 / args.fps)
    finally:
        if viewer:
            viewer.close()
    if args.savejoint and records:
        print(f'Saved joints: {save_joints(args, records)}')
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.csv, features, delimiter=',')
    metadata = dict(vars(args), completed_steps=len(records), reset_count=reset_count,
                    effective_initial_pose=initial_pose, effective_initial_state=initial_state,
                    effective_motion_offset=offset,
                    effective_neural_normalization=(preprocessing.get('neural_normalization')
                                                    if args.mode.startswith('neural') else None),
                    neural_stats_frames=preprocessing.get('neural_stats_frames'),
                    neural_normalization_scope=preprocessing.get('neural_normalization_scope'))
    (args.output_dir / f'{run_id}.json').write_text(json.dumps(metadata, default=str, indent=2), encoding='utf-8')
    print(f'Completed {len(records)} frames.')


if __name__ == '__main__':
    main(parse_args())

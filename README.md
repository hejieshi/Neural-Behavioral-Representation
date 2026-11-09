# Neural–Behavioral Representation of Natural Whole-body Movement in Monkeys

This repository is the official implementation of our paper "Neural-Behavioral Representation of Natural Whole-body Movement in Monkeys", including the model training code and example data for training models and playing whole-body movements.

## Overview
In this work, we introduce a setup captures large-scale neural activity and 3D kinematics in freely moving monkeys. And we provide the demonstration that whole-body movement of freely moving monkeys can be reconstructed from cortical activity.


## Installation

Use Python 3.10 or later and install the dependencies: in `requirements.txt`.
 

## Released data

The data and the original videos are available at: [google drive](https://drive.google.com/drive/folders/1wAkJphJIWIgWKQkpPBO4CZ9QaFQNRs32?usp=sharing). Download the files to the `environments/` folder.

| File | Contents |
| --- | --- |
| `kinematic.npz` | Kinematic representations, starting with $p_1$; `data` has 151 features per row and `end_indices` contains inclusive clip ends. |
| `neural_feature.mat` | Neural features in `features`, with dimensions channel × band/feature × timepoint; starts with $p_5$. The supplied array has shape 62 × 30 × 17996. |
| `body_info.npz` | Joint definitions, skeleton topology and bone lengths. |
| `pose_presentation.pkl` | Root angles and positions, starting with $p_0$, used to initialize playback at the requested frame. |

## Training

```bash
python train.py --experiment neural
python train.py --experiment behavior
python train.py --experiment label --labels labels.npy
```
Trained model can be downloaded from google drive. Use models to reproduce the results reported in the paper.


| Experiment | Details |
| --- | --- |
| `neural` | Neural–Behavioral Model|
| `behavior` | Behavioral Model, without neural input. |
| `label` | Behavioral-Label Model, requires a one-dimensional integer `.npy` label array aligned with kinematic rows. |

The LSTM implementation follows the standard pipeline described in [A general model based on Riemannian manifold for stable decoding movement trajectory from ECoG signals](https://www.sciencedirect.com/science/article/pii/S2589004225027828). 

## Play

Replace `model_path` with the checkpoint created by your training run.

```bash
python play.py --mode gt --testframe 14310 --savejoint 0  # groug_truth
python play.py --mode neural --vae <path/to/Neural–Behavioral Model> --testframe 14310 --savejoint 0
python play.py --mode behavior --vae <path/to/Behavioral Model> --testframe 14310
```

Use `--csv path.csv` to export
151-dimensional generated motion features, or `--save` to export rendered PNG
frames. `--savejoint 1` saves joint trajectories under `outputs/play/`.

LSTM mode displays precomputed joint predictions; it does not train or run an
LSTM model:

```bash
python play.py --mode lstm --lstm-file predictions.npz --testframe 14310
```

The prediction array must have shape `(T, 15, 3)` or `(T, 3, 15)`, stored under
`predict_joint_xyzs` or `arr_0`.



## Citation

Cite our paper:

```
{he2026neuralbehavioralrepresentationnaturalwholebody,
      title={Neural-Behavioral Representation of Natural Whole-body Movement in Monkeys}, 
      author={Jieshi He and Puzhe Li and Yanan Sui and Mu-ming Poo},
      year={2026},
      eprint={2605.29355},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.29355}, 
}
```
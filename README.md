# robomimic

> **DR-LfD customized fork** — branch `pc` (point cloud), based on upstream
> robomimic `0.3.0`. This fork adds point-cloud / instance-segmentation
> observations and the extraction tooling DR-LfD's visuomotor policies consume.
> The original upstream README follows after the divider.

## DR-LfD fork — what's customized

The `pc` branch extends robomimic's robosuite env layer to emit 3D observations,
then provides scripts that bake those observations into HDF5 datasets for policy
training/eval.

1. **Point-cloud & instance-segmentation observations** —
   `robomimic/envs/env_robosuite.py` (`EnvRobosuite`). From depth + instance
   segmentation it builds per-object instance point clouds (`get_instance_pcd`,
   with a `KDTree` per object), an obstacle point cloud (`get_obstacle_pcd`), and a
   voxelized workspace (`get_all_pcd`, a `[4, 64, 64, 64]` RGB+occupancy grid).
   `get_observation` injects these into the obs dict as `{obj}_point_cloud`,
   `{obj}_kdtree`, `voxels`, and `point_cloud`. Outlier filtering uses Open3D radius
   + statistical removal. New deps: `open3d`, `scipy`.

2. **Adaptive PC workspace.** The bounding box auto-adapts per environment — a tight
   box for LIBERO (centered via `workspace_offset`, narrow z to drop the gripper)
   versus a wide box for the DexMimicGen bimanual tasks (centered on `table_offset`).

3. **Switchable LIBERO envs** — `robomimic/envs/env_libero_switchable.py` (and
   `_V4`). `Libero_env_switchable` (requires robosuite ≥ 1.5) adds switchable
   controllers, TAMP replay hooks (`replay_tamp_step`, `inference_once`,
   `get_cur_jpose_robosuite`), and `rerun`-based visualization
   (`robomimic/utils/rerun_logger.py`).

4. **Dataset-extraction scripts.**
   `robomimic/scripts/dataset_extract_instance_pc.py` regenerates instance
   point-cloud observations and writes `<name>_pc_instance<N>.hdf5` (plus an
   `instance_name2id` map); `dataset_extract_frontview_vid.py` renders preview
   videos. `dataset_states_to_obs.py` is lightly patched for the new camera sets
   (`agentview` / `birdview`, plus `sideview` / `robot0_eye_in_hand` where needed).

### Note
Point-cloud observations are produced at the **env-wrapper + extraction-script**
layer and stored in the HDF5 as ordinary obs keys (`*_point_cloud`, `voxels`); no
new obs modality is registered in `obs_utils.py`, so the **consuming** training
config (Diffusion-Policy / DR-LfD) must register those keys itself.

### Install
```bash
git clone https://github.com/Dr-LfD/robomimic-customized.git -b pc robomimic
pip install -e robomimic
```
`rerun_logger.py` is version-sensitive to the `rerun` SDK — pin a known-good version.

---

<p align="center">
  <img width="24.0%" src="docs/images/task_lift.gif">
  <img width="24.0%" src="docs/images/task_can.gif">
  <img width="24.0%" src="docs/images/task_tool_hang.gif">
  <img width="24.0%" src="docs/images/task_square.gif">
  <img width="24.0%" src="docs/images/task_lift_real.gif">
  <img width="24.0%" src="docs/images/task_can_real.gif">
  <img width="24.0%" src="docs/images/task_tool_hang_real.gif">
  <img width="24.0%" src="docs/images/task_transport.gif">
 </p>

[**[Homepage]**](https://robomimic.github.io/) &ensp; [**[Documentation]**](https://robomimic.github.io/docs/introduction/overview.html) &ensp; [**[Study Paper]**](https://arxiv.org/abs/2108.03298) &ensp; [**[Study Website]**](https://robomimic.github.io/study/) &ensp; [**[ARISE Initiative]**](https://github.com/ARISE-Initiative)

-------
## Latest Updates
- [07/03/2023] **v0.3.0**: BC-Transformer and IQL :brain:, support for DeepMind MuJoCo bindings :robot:, pre-trained image reps :eye:, wandb logging :chart_with_upwards_trend:, and more
- [05/23/2022] **v0.2.1**: Updated website and documentation to feature more tutorials :notebook_with_decorative_cover:
- [12/16/2021] **v0.2.0**: Modular observation modalities and encoders :wrench:, support for [MOMART](https://sites.google.com/view/il-for-mm/home) datasets :open_file_folder: [[release notes]](https://github.com/ARISE-Initiative/robomimic/releases/tag/v0.2.0) [[documentation]](https://robomimic.github.io/docs/v0.2/introduction/overview.html)
- [08/09/2021] **v0.1.0**: Initial code and paper release

-------

## Colab quickstart
Get started with a quick colab notebook demo of robomimic without installing anything locally.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b62r_km9pP40fKF0cBdpdTO2P_2eIbC6?usp=sharing)


-------

**robomimic** is a framework for robot learning from demonstration.
It offers a broad set of demonstration datasets collected on robot manipulation domains and offline learning algorithms to learn from these datasets.
**robomimic** aims to make robot learning broadly *accessible* and *reproducible*, allowing researchers and practitioners to benchmark tasks and algorithms fairly and to develop the next generation of robot learning algorithms.

## Core Features

<p align="center">
  <img width="50.0%" src="docs/images/core_features.png">
 </p>

<!-- **Standardized Datasets**
- Simulated and real-world tasks
- Multiple environments and robots
- Diverse human-collected and machine-generated datasets

**Suite of Learning Algorithms**
- Imitation Learning algorithms (BC, BC-RNN, HBC)
- Offline RL algorithms (BCQ, CQL, IRIS, TD3-BC)

**Modular Design**
- Low-dim + Visuomotor policies
- Diverse network architectures
- Support for external datasets

**Flexible Workflow**
- Hyperparameter sweep tools
- Dataset visualization tools
- Generating new datasets -->


## Reproducing benchmarks

The robomimic framework also makes reproducing the results from different benchmarks and datasets easy. See the [datasets page](https://robomimic.github.io/docs/datasets/overview.html) for more information on downloading datasets and reproducing experiments.

## Troubleshooting

Please see the [troubleshooting](https://robomimic.github.io/docs/miscellaneous/troubleshooting.html) section for common fixes, or [submit an issue](https://github.com/ARISE-Initiative/robomimic/issues) on our github page.

## Contributing to robomimic
This project is part of the broader [Advancing Robot Intelligence through Simulated Environments (ARISE) Initiative](https://github.com/ARISE-Initiative), with the aim of lowering the barriers of entry for cutting-edge research at the intersection of AI and Robotics.
The project originally began development in late 2018 by researchers in the [Stanford Vision and Learning Lab](http://svl.stanford.edu/) (SVL).
Now it is actively maintained and used for robotics research projects across multiple labs.
We welcome community contributions to this project.
For details please check our [contributing guidelines](https://robomimic.github.io/docs/miscellaneous/contributing.html).

## Citation

Please cite [this paper](https://arxiv.org/abs/2108.03298) if you use this framework in your work:

```bibtex
@inproceedings{robomimic2021,
  title={What Matters in Learning from Offline Human Demonstrations for Robot Manipulation},
  author={Ajay Mandlekar and Danfei Xu and Josiah Wong and Soroush Nasiriany and Chen Wang and Rohun Kulkarni and Li Fei-Fei and Silvio Savarese and Yuke Zhu and Roberto Mart\'{i}n-Mart\'{i}n},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2021}
}
```

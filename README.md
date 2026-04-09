# LightPerception

> A lightweight research repository for multi-camera 3D perception built on top of MMDetection3D and PETR-style models.

## Overview

LightPerception is an OpenMMLab-style project for camera-based 3D object detection, with a focus on PETR and its depth-aware variants for autonomous driving scenes.

The repository contains:

- PETR-based multi-view 3D detectors implemented under `projects/mmdet3d_plugin/`
- nuScenes dataset extensions and custom image pipelines
- multiple research configs for PETR, PETR-Depth, and PETR-DepthV2
- distributed training and evaluation scripts under `tools/`
- ONNX export utilities under `export/`

The current example configuration used in this README is:

```bash
projects/configs/petr/petr_r50dcn_gridmask_c5_idav2.py
```

This configuration is a PETR-style multi-camera 3D detector for nuScenes 10-class detection, using:

- `ResNet-50` image backbone with `DCNv2`
- `GridMask` augmentation
- custom `ResizeCropFlipImageV2` image augmentation
- `CustomNuScenesDataset`
- `NMSFreeCoder` for query-based box decoding

## Repository Layout

```text
LightPerception/
|-- projects/
|   |-- configs/
|   |   |-- petr/
|   |   |-- petr_depth/
|   |   `-- petr_depthv2/
|   `-- mmdet3d_plugin/
|       |-- core/
|       |-- datasets/
|       `-- models/
|-- tools/
|   |-- train.py
|   |-- test.py
|   |-- dist_train.sh
|   |-- dist_test.sh
|   `-- create_data.py
|-- export/
|   |-- pth2onnx.py
|   `-- sensor_info_nuscenes.yaml
|-- ckpts/
|-- data/
`-- work_dirs/
```

## What This Project Does

This repository is mainly used to study and deploy camera-only 3D perception models for autonomous driving. In practice, it supports the following workflow:

1. Prepare nuScenes metadata in MMDetection3D-compatible `.pkl` files.
2. Train PETR-style 3D detectors with custom plugin modules.
3. Evaluate checkpoints with the standard nuScenes detection metrics.
4. Export trained models to ONNX for downstream deployment experiments.

Compared with a plain upstream PETR setup, this repository also includes custom dataset wrappers, image transforms, model variants, and export utilities that are directly useful for engineering and deployment.

## Environment Setup

The repository does not provide its own `setup.py`, so the environment is prepared by installing the required OpenMMLab packages directly.

Recommended environment:

- Python `3.8`
- CUDA `11.3`
- PyTorch `1.12.1`
- MMCV `1.6.2`
- MMDetection `2.28.2`
- MMSegmentation `0.30.0`
- MMDetection3D `1.0.0rc6`

Create the environment:

```bash
conda create -n light_perception python==3.8 -y
conda activate light_perception

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu113

pip install numba==0.53.0 numpy==1.23.5 yapf==0.30.0 einops distro

pip install mmcv-full==1.6.2 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

pip install mmdet==2.28.2 mmsegmentation==0.30.0 mmdet3d==1.0.0rc6
```

Optional packages for model export:

```bash
pip install onnx onnxruntime onnxsim pyyaml
```

## Build / Compilation

There is no extra project-local CUDA extension that must be compiled inside this repository. In most cases, installing the matching binary package of `mmcv-full` is enough.

Still, please note:

- `DCNv2` support is expected through the installed OpenMMLab stack.
- If your environment differs from CUDA 11.3 or PyTorch 1.12.1, you should reinstall `mmcv-full` with a matching build.
- The repository relies on dynamic plugin loading through `plugin=True` and `plugin_dir='projects/mmdet3d_plugin/'`, so keeping `PYTHONPATH` correct is important. The provided shell scripts already handle this.

## Data Preparation

This project is configured for the nuScenes dataset.

Expected local structure:

```text
data/nuscenes/
|-- samples/
|-- sweeps/
|-- maps/
|-- v1.0-trainval/
`-- v1.0-test/
```

Generate metadata files:

```bash
python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag mmdet3d_nuscenes \
  --version v1.0 \
  --max-sweeps 10
```

This command will generate files such as:

- `mmdet3d_nuscenes_infos_train.pkl`
- `mmdet3d_nuscenes_infos_val.pkl`
- `mmdet3d_nuscenes_infos_test.pkl`
- `mmdet3d_nuscenes_dbinfos_train.pkl`

## Configuration Notes

Before training or evaluation, check the dataset path in the selected config file:

```python
data_root = '/media/zichen/MyPassport/nuScenes/'
```

The example config currently contains a machine-specific absolute path. Replace it with the path on your machine, for example:

```python
data_root = 'data/nuscenes/'
```

You should verify that the annotation files referenced by the config also exist, especially:

> [How to generate mmdet3d_nuscenes_30f_infos_train.pkl](https://github.com/megvii-research/PETR/issues/4)
> python -m tools.create_data nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes

- `mmdet3d_nuscenes_infos_train.pkl`
- `mmdet3d_nuscenes_infos_val.pkl`

## Training

Single-node distributed training with the current PETR config:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash tools/dist_train.sh \
  projects/configs/petr_depth/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis.py \
  4 \
  --work-dir work_dirs/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis
```

Useful options:

- `--work-dir`: directory for logs and checkpoints
- `--resume-from`: resume from a specific checkpoint
- `--no-validate`: disable validation during training
- `--cfg-options`: override config values from the command line

## Evaluation

Evaluate a trained checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash tools/dist_test.sh \
  projects/configs/petr_depth/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis.py \
  work_dirs/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis/latest.pth \
  2 \
  --eval bbox
```

Save raw prediction results:

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    projects/configs/petr_depth/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis.py \
  work_dirs/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis/latest.pth \
  --out work_dirs/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis/results.pkl
```

## Export

Export the current PETR configuration to ONNX:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python export/pth2onnx.py \
  projects/configs/petr_depth/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis.py \
  --checkpoint work_dirs/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis/latest.pth \
  --section 3dppe_v_pe_di \
  --sensor-info export/sensor_info_nuscenes.yaml
```

By default, the export script will:

- build the model from the config
- load the checkpoint
- infer the input image shape from the config
- export an ONNX file into the checkpoint directory
- create a simplified ONNX model
- optionally validate ONNX outputs with `onnxruntime`

### ONNX Export Tips

- Keep `all_cls_scores` and `all_bbox_preds` as exported ONNX outputs, even if they are not needed by the final deployment interface. These intermediate outputs help preserve numerical correctness. Without them, later graph optimizations such as constant folding may incorrectly treat part of the computation as constants and lead to inaccurate results.
- Camera parameters must not appear as NumPy constants inside the exported graph, otherwise they may be folded into constants during export. Camera parameters should remain true model inputs so the exported model can generalize to different machines and sensor setups.
- TensorRT `8.5.3` does not support the `IsInf` operator, so `torch.nan_to_num` should not be used in export-friendly code paths.
- `MMCVDeformConv` is a customized operator, which is not supported by TensorRT natively.

The exported files will usually be written to:

```text
work_dirs/petr_r50dcn_gridmask_c5_idav2/
|-- petr_r50dcn_gridmask_c5_idav2.onnx
`-- simplify_petr_r50dcn_gridmask_c5_idav2.onnx
```

Skip ONNX runtime validation if needed:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python export/pth2onnx.py \
  projects/configs/petr_depth/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis.py \
  --checkpoint work_dirs/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis/latest.pth \
  --section 3dppe_v_pe_di \
  --sensor-info export/sensor_info_nuscenes.yaml \
  --no-validate
```

## Notes

- The current example config uses `CustomNuScenesDataset` and custom image augmentations from `projects/mmdet3d_plugin/`.
- The repository includes a customized `NMSFreeCoder`. The current implementation selects the best class per query before top-k filtering, which avoids assigning multiple class results to a single query.
- Old README examples referred to `tools.pth2onnx`, but the actual export script in this repository is `export/pth2onnx.py`.

## Citation

If you use this repository in academic research, please cite the original PETR and OpenMMLab projects that this work builds on.

```bibtex
@inproceedings{liu2022petr,
  title={PETR: Position Embedding Transformation for Multi-View 3D Object Detection},
  author={Liu, Yingfei and Wang, Tiancai and Zhang, Xiangyu and Sun, Jian},
  booktitle={European Conference on Computer Vision},
  pages={531--548},
  year={2022},
  organization={Springer}
}

@inproceedings{wang2022detr3d,
  title={{DETR3D}: 3D Object Detection from Multi-view Images via 3D-to-2D Queries},
  author={Wang, Yue and Guizilini, Vitor Campagnolo and Zhang, Tianyuan and Wang, Yilun and Zhao, Hang and Solomon, Justin},
  booktitle={Conference on Robot Learning},
  pages={180--191},
  year={2022},
  organization={PMLR}
}

@misc{mmdet3d2020,
    title={{MMDetection3D: OpenMMLab} next-generation platform for general {3D} object detection},
    author={MMDetection3D Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmdetection3d}},
    year={2020}
}
```

## Acknowledgements

This repository is built on top of the following excellent open-source projects:

- [PETR](https://github.com/megvii-research/PETR)
- [DETR3D](https://github.com/WangYueFt/detr3d)
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
- [MMCV](https://github.com/open-mmlab/mmcv)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [nuScenes](https://www.nuscenes.org/)

The ONNX export utility also includes code adapted from NVIDIA/OpenMMLab-based deployment work and local project modifications for sensor-parameter-driven export and numerical validation.

# LightPerception

> A minimum repository for auto-driving perception


### Installation

```shell
conda create -n light_perception python==3.8 -y
conda activate light_perception
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install numba==0.53.0 numpy==1.23.5 yapf==0.30.0 einops distro
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
pip install mmdet==2.28.2 mmsegmentation==0.30.0 mmdet3d==1.0.0rc6
```

### Training

```shell
#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2,3,4,5 tools/dist_train.sh projects/configs/petr_depth/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg.py 4 --work-dir work_dirs/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg/

# CUDA_VISIBLE_DEVICES=2,3,4,5 tools/dist_train.sh projects/configs/petr_depth/petr_depth_3dpe_dfl_mobilenetv4_hybrid_large_wogridmask_p4_800x320_pdg.py 4 --work-dir work_dirs/petr_depth_3dpe_dfl_mobilenetv4_hybrid_large_wogridmask_p4_800x320_pdg/

# CUDA_VISIBLE_DEVICES=2,3,4,5 tools/dist_train.sh projects/configs/petr_depth/petr_depth_3dpe_dfl_hgnetv2_hybrid_large_wogridmask_p4_800x320_pdg.py 4 --work-dir work_dirs/petr_depth_3dpe_dfl_hgnetv2_hybrid_large_wogridmask_p4_800x320_pdg/

# CUDA_VISIBLE_DEVICES=2,3,4,5 tools/dist_train.sh projects/configs/petr_depth/petr_depth_3dpe_dfl_r50_hybrid_large_wogridmask_p4_800x320_pdg.py 4 --work-dir work_dirs/petr_depth_3dpe_dfl_r50_hybrid_large_wogridmask_p4_800x320_pdg/

CUDA_VISIBLE_DEVICES=2,3,4,5 tools/dist_train.sh projects/configs/petr_depth/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis.py 4 --work-dir work_dirs/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis/
```

### Evaluation
```shell
### [How to generate mmdet3d_nuscenes_30f_infos_train.pkl](https://github.com/megvii-research/PETR/issues/4)

# python -m tools.create_data nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
# tools/dist_test.sh projects/configs/petr_depth/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg.py work_dirs/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg/latest.pth 2 --eval bbox
tools/dist_test.sh projects/configs/petr_depth/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis.py work_dirs/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis/latest.pth 2 --eval bbox
```

### Export 
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m tools.pth2onnx --section 3dppe_v_pe_di projects/configs/petr_depth/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis.py --checkpoint work_dirs/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis/latest.pth
```
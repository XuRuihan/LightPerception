# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This fils is modified from test.py in the original repo.
# Modification: adding 2 proxy class TrtEncoderContainer and TrtPtsHeadContainer
#               and then export these two torch.nn.Module to onnx

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

import argparse
import os
import sys

import numpy as np
import onnx

sys.path.append("./")
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import wrap_fp16_model, load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from mmdet.models.utils.transformer import inverse_sigmoid
from onnxsim import simplify


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet benchmark a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--checkpoint", default="", help="checkpoint file")
    parser.add_argument("--section", type=str, help="section can be either extract_img_feat or pts_head_memory")
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument("--samples", default=300, help="samples to benchmark")
    parser.add_argument("--log-interval", default=50, help="interval of logging")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    print(model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    if args.checkpoint:
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        # palette for visualization in segmentation tasks
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        elif hasattr(dataset, 'PALETTE'):
            # segmentation dataset has `PALETTE` attribute
            model.PALETTE = dataset.PALETTE
    model = MMDataParallel(model, device_ids=[0])

    # Wrapper Class for onnx conversion
    class Trt3DPPEContainer(torch.nn.Module):
        def __init__(self, mod, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.mod = mod
            img_meta = {
                "filename": None,
                "ori_shape": None,
                "img_shape": [
                    (320, 800, 3),
                    (320, 800, 3),
                    (320, 800, 3),
                    (320, 800, 3),
                    (320, 800, 3),
                    (320, 800, 3),
                ],
                "lidar2img": [
                    np.array(
                        [
                            [6.84086187e02, 4.08196937e02, 1.33285465e01, -1.53439636e02],
                            [5.56854283e00, 1.12487443e02, -6.81814085e02, -2.66874187e02],
                            [5.94895437e-04, 9.99816458e-01, 1.91493307e-02, -4.00761352e-01],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                        ]
                    ),
                    np.array(
                        [
                            [7.11646043e02, -3.46926154e02, -1.51690026e01, -2.43744651e02],
                            [7.85284767e01, 7.25540271e01, -6.80025470e02, -2.91996210e02],
                            [8.35849521e-01, 5.48927364e-01, 5.85901484e-03, -5.93941342e-01],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                        ]
                    ),
                    np.array(
                        [
                            [5.81606899e01, 8.00107360e02, 1.99466804e01, -9.33475458e01],
                            [-7.32207159e01, 7.06222872e01, -6.85974003e02, -2.78176017e02],
                            [-8.16678826e-01, 5.76989416e-01, 1.09045200e-02, -4.90841216e-01],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                        ]
                    ),
                    np.array(
                        [
                            [-4.39456320e02, -4.13087131e02, -7.74372017e00, -4.29285441e02],
                            [3.99178193e00, -9.08159359e01, -4.37668435e02, -2.19321657e02],
                            [-6.03377536e-03, -9.99952101e-01, -7.70642470e-03, -1.02939076e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                        ]
                    ),
                    np.array(
                        [
                            [-5.90256421e02, 5.18395229e02, 5.23057445e00, -3.22997952e02],
                            [-8.13276059e01, -9.31426483e00, -6.81081263e02, -2.09408660e02],
                            [-9.48287070e-01, -3.16048189e-01, -2.94138764e-02, -4.41955096e-01],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                        ]
                    ),
                    np.array(
                        [
                            [1.32634130e02, -7.79173058e02, -3.21416916e01, -1.77752615e02],
                            [9.46631407e01, -1.00198857e01, -6.81524886e02, -2.38296306e02],
                            [9.33242957e-01, -3.58761667e-01, -1.86453674e-02, -5.02135675e-01],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                        ]
                    ),
                ],
                "pad_shape": [
                    (320, 800, 3),
                    (320, 800, 3),
                    (320, 800, 3),
                    (320, 800, 3),
                    (320, 800, 3),
                    (320, 800, 3),
                ],
                "scale_factor": None,
                "flip": None,
                "pcd_horizontal_flip": None,
                "pcd_vertical_flip": None,
                "box_mode_3d": None,
                "box_type_3d": None,
                "img_norm_cfg": None,
                "sample_idx": None,
                "pcd_scale_factor": None,
                "pts_filename": None,
                "intrinsics": [
                    np.array(
                        [
                            [683.86531682, 0.0, 408.78420818, 0.0],
                            [0.0, 683.86531682, 99.41382607, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [680.85764714, 0.0, 404.30286938, 0.0],
                            [0.0, 680.85764714, 101.48060114, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [687.20291872, 0.0, 414.37238381, 0.0],
                            [0.0, 687.20291872, 93.06590338, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [436.97935227, 0.0, 415.77860197, 0.0],
                            [0.0, 436.97935227, 94.16035921, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [678.64042682, 0.0, 395.740807, 0.0],
                            [0.0, 678.64042682, 100.09891369, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [680.13744694, 0.0, 403.91658623, 0.0],
                            [0.0, 680.13744694, 104.64574213, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                ],
                "extrinsics": [
                    np.array(
                        [
                            [9.99967371e-01, -7.49039239e-04, 8.04340031e-03, 1.51861422e-02],
                            [8.05626761e-03, 1.91439208e-02, -9.99784280e-01, -3.31984913e-01],
                            [5.94895437e-04, 9.99816458e-01, 1.91493307e-02, -4.00761352e-01],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                        ]
                    ),
                    np.array(
                        [
                            [0.54888079, -0.83550367, -0.02575842, -0.00530575],
                            [-0.00924427, 0.02474607, -0.99965103, -0.3403394],
                            [0.83584952, 0.54892736, 0.00585901, -0.59394134],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [0.57707823, 0.81638024, 0.02245064, 0.16013247],
                            [0.00405155, 0.02462773, -0.99968848, -0.33832137],
                            [-0.81667883, 0.57698942, 0.01090452, -0.49084122],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [-0.99992735, 0.00611369, -0.01038849, -0.00294474],
                            [0.0104351, 0.00764318, -0.99991634, -0.2800907],
                            [-0.00603378, -0.9999521, -0.00770642, -1.02939076],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [-0.3167812, 0.9481728, 0.02485977, -0.21822792],
                            [0.02003255, 0.03289196, -0.99925813, -0.24338284],
                            [-0.94828707, -0.31604819, -0.02941388, -0.4419551],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [-0.35921883, -0.93255161, -0.03618462, 0.03685772],
                            [-0.00440611, 0.04046681, -0.99917117, -0.27310648],
                            [0.93324296, -0.35876167, -0.01864537, -0.50213568],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                ],
                "input_shape": [320, 800],
            }
            self.img_metas = [img_meta]

        def forward(self, img):
            mod = self.mod
            img_feats, _ = mod.extract_img_feat(img, self.img_metas)
            # outs = mod.pts_bbox_head(img_feats.unsqueeze(0), self.img_metas)  # For PETR
            outs = mod.pts_bbox_head(img_feats, self.img_metas)  # For 3DPPE

            # all_cls_scores: [nb_dec, bs, num_query, cls_out_channels]
            # all_bbox_preds: [nb_dec, bs, num_query, 10]  // assume x,y,z,w,l,h,yaw,vx,vy,vz
            all_cls_scores, all_bbox_preds = outs["all_cls_scores"], outs["all_bbox_preds"]
            return all_cls_scores[-1], all_bbox_preds[-1]

    model.eval()
    model = model.float()

    tm = Trt3DPPEContainer(model.module)
    arrs = [
        torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, 6, 3, 256, 704))).float(),
    ]
    input_names = ["img"]
    output_names = [
        # "img_feats",
        "all_cls_scores",
        "all_bbox_preds",
    ]

    tm = tm.float()
    tm.cpu()
    tm.eval()
    tm.training = False
    tm.mod.pts_bbox_head.with_dn = False

    tm_args = tuple(arrs)

    filename = args.section + ".onnx"
    onnx_path = os.path.join("work_dirs/", args.section, filename)
    with torch.no_grad():
        torch.onnx.export(
            tm,
            tm_args,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,  # `False` when debugging, `True` for best performance and simplify (but with bugs)
            verbose=False,
            opset_version=11,
        )

    onnx_model = onnx.load(onnx_path)
    onnx_model_simp, check = simplify(onnx_model)
    simpified_onnx_path = os.path.join("work_dirs/", args.section, "simplify_" + filename)
    onnx.save(onnx_model_simp, simpified_onnx_path)

    print(args.section + " onnx export success!")


if __name__ == "__main__":
    main()


# pip install onnx onnxruntime onnxsim
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m tools.pth2onnx --section petr projects/configs/petr/petr_vovnet_gridmask_p4_800x320.py --checkpoint work_dirs/petr/epoch_24.pth
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m tools.pth2onnx --section 3dppe projects/configs/petr_depth/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg.py --checkpoint work_dirs/3dppe/epoch_24.pth
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m tools.pth2onnx --section 3dppe projects/configs/petr_depth/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg.py --checkpoint work_dirs/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg/epoch_23.pth

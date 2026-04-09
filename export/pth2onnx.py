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

from copy import deepcopy
import numpy as np
import onnx
import yaml
try:
    import onnxruntime as ort
except ImportError:
    ort = None

sys.path.append("./")
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import wrap_fp16_model, load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from onnxsim import simplify


def denormalize_bbox_without_atan2(normalized_bboxes):
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    w = normalized_bboxes[..., 2:3].exp()
    l = normalized_bboxes[..., 3:4].exp()
    h = normalized_bboxes[..., 5:6].exp()

    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]

    vx = normalized_bboxes[..., 8:9]
    vy = normalized_bboxes[..., 9:10]
    return torch.cat([cx, cy, cz, w, l, h, rot_sine, rot_cosine, vx, vy], dim=-1)


def decode_nms_free_preds(preds_dicts, max_num):
    cls_scores = preds_dicts["all_cls_scores"][-1, 0]
    bbox_preds = preds_dicts["all_bbox_preds"][-1, 0]

    cls_scores = cls_scores.sigmoid()
    query_scores, labels = cls_scores.max(dim=-1)
    scores, bbox_indices = query_scores.topk(max_num)
    labels = labels[bbox_indices]
    selected_bbox_preds = bbox_preds[bbox_indices]
    decoded_bboxes = denormalize_bbox_without_atan2(selected_bbox_preds)

    return scores, labels, decoded_bboxes


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet benchmark a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--checkpoint", default="", help="checkpoint file")
    parser.add_argument("--section", type=str, help="section can be either extract_img_feat or pts_head_memory")
    parser.add_argument(
        "--opset-version",
        type=int,
        default=11,
        help="ONNX opset version to export with",
    )
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument("--samples", default=300, help="samples to benchmark")
    parser.add_argument("--log-interval", default=50, help="interval of logging")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip numerical validation between PyTorch and ONNX Runtime",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for numerical validation",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for numerical validation",
    )
    parser.add_argument(
        "--sensor-info",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "sensor_info_nuscenes.yaml"),
        help="Path to sensor info yaml used to build intrinsics/extrinsics/lidar2img/img2lidar",
    )
    args = parser.parse_args()
    return args


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def resolve_input_hw(cfg):
    ida_aug_conf = cfg.get("ida_aug_conf", None)
    if ida_aug_conf is not None and ida_aug_conf.get("final_dim", None) is not None:
        final_dim = ida_aug_conf["final_dim"]
        return int(final_dim[0]), int(final_dim[1])

    pipeline_candidates = []
    if cfg.get("test_pipeline", None) is not None:
        pipeline_candidates.append(cfg.test_pipeline)

    data_cfg = cfg.get("data", None)
    if data_cfg is not None and data_cfg.get("test", None) is not None:
        test_cfg = data_cfg.test
        if test_cfg.get("pipeline", None) is not None:
            pipeline_candidates.append(test_cfg.pipeline)
        if test_cfg.get("dataset", None) is not None and test_cfg.dataset.get("pipeline", None) is not None:
            pipeline_candidates.append(test_cfg.dataset.pipeline)

    for pipeline in pipeline_candidates:
        for step in pipeline:
            data_aug_conf = step.get("data_aug_conf", None)
            if data_aug_conf is not None and data_aug_conf.get("final_dim", None) is not None:
                final_dim = data_aug_conf["final_dim"]
                return int(final_dim[0]), int(final_dim[1])
            if step.get("input_size", None) is not None:
                input_size = step["input_size"]
                return int(input_size[0]), int(input_size[1])
            for sub_step in step.get("transforms", []):
                data_aug_conf = sub_step.get("data_aug_conf", None)
                if data_aug_conf is not None and data_aug_conf.get("final_dim", None) is not None:
                    final_dim = data_aug_conf["final_dim"]
                    return int(final_dim[0]), int(final_dim[1])
                if sub_step.get("input_size", None) is not None:
                    input_size = sub_step["input_size"]
                    return int(input_size[0]), int(input_size[1])

    raise ValueError("Failed to resolve input image size from config.")


def validate_onnx_output(torch_outputs, ort_outputs, output_names, rtol, atol):
    bbox_field_names = ["cx", "cy", "cz", "w", "l", "h", "sin", "cos", "vx", "vy"]
    all_passed = True
    for name, torch_out, ort_out in zip(output_names, torch_outputs, ort_outputs):
        torch_np = to_numpy(torch_out)
        ort_np = np.asarray(ort_out)
        if np.issubdtype(torch_np.dtype, np.integer):
            matched = np.array_equal(torch_np, ort_np)
            diff_msg = "exact match" if matched else f"max_abs_diff={np.abs(torch_np - ort_np).max()}"
        else:
            abs_diff = np.abs(torch_np - ort_np)
            max_abs_diff = float(abs_diff.max()) if abs_diff.size > 0 else 0.0
            matched = np.allclose(torch_np, ort_np, rtol=rtol, atol=atol)
            diff_msg = f"max_abs_diff={max_abs_diff:.6e}"
        all_passed &= matched
        print(f"[ONNX][{name}] {'PASS' if matched else 'FAIL'}: {diff_msg}")
        if name == "decoded_bboxes" and torch_np.ndim >= 2 and torch_np.shape[-1] <= len(bbox_field_names):
            per_dim_max_abs_diff = np.abs(torch_np - ort_np).reshape(-1, torch_np.shape[-1]).max(axis=0)
            per_dim_msg = ", ".join(
                f"{bbox_field_names[idx]}={value:.6e}"
                for idx, value in enumerate(per_dim_max_abs_diff)
            )
            print(f"[ONNX][{name}] per_dim_max_abs_diff: {per_dim_msg}")
    return all_passed


def load_sensor_meta(sensor_info_path):
    with open(sensor_info_path, "r", encoding="utf-8") as f:
        sensor_info = yaml.safe_load(f)

    cameras = sensor_info.get("Camera", None)
    if not isinstance(cameras, list) or len(cameras) == 0:
        raise ValueError(f"Invalid camera configuration in {sensor_info_path}")

    intrinsics = []
    extrinsics = []
    lidar2img = []
    img2lidar = []

    for camera in cameras:
        intrin = np.asarray(camera["intrinsics"], dtype=np.float32)
        cam2lidar = np.asarray(camera["cam2lidar"], dtype=np.float32)
        extrin = np.linalg.inv(cam2lidar).astype(np.float32)

        intrinsics.append(intrin)
        extrinsics.append(extrin)

        if camera.get("lidar2img", None) is not None:
            cur_lidar2img = np.asarray(camera["lidar2img"], dtype=np.float32)
        else:
            cur_lidar2img = (intrin @ extrin).astype(np.float32)
        lidar2img.append(cur_lidar2img)
        img2lidar.append(np.linalg.inv(cur_lidar2img).astype(np.float32))

    return {
        "intrinsics": np.stack(intrinsics, axis=0),
        "extrinsics": np.stack(extrinsics, axis=0),
        "lidar2img": np.stack(lidar2img, axis=0),
        "img2lidar": np.stack(img2lidar, axis=0),
    }


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

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    print(model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
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
    input_h, input_w = resolve_input_hw(cfg)
    sensor_meta = load_sensor_meta(args.sensor_info)
    num_views = sensor_meta["intrinsics"].shape[0]
    print(f"[ONNX] resolved input shape from config: (1, {num_views}, 3, {input_h}, {input_w})")

    # Wrapper Class for onnx conversion
    class Trt3DPPEContainer(torch.nn.Module):
        def __init__(self, mod, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.mod = mod
            self.max_num = mod.pts_bbox_head.bbox_coder.max_num

        def forward(self, img, intrinsics, extrinsics, img2lidar):
            mod = self.mod
            _, num_views, _, input_h, input_w = img.shape
            img_metas = [
                {
                    "filename": None,
                    "ori_shape": None,
                    "img_shape": [(input_h, input_w, 3)] * num_views,
                    "lidar2img": None,
                    "img2lidar": img2lidar[0],
                    "pad_shape": [(input_h, input_w, 3)] * num_views,
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
                    "intrinsics": intrinsics[0],
                    "extrinsics": extrinsics[0],
                    "input_shape": [input_h, input_w],
                }
            ]
            img_feats, _ = mod.extract_img_feat(img, img_metas)
            preds_dicts = mod.pts_bbox_head(img_feats, img_metas)  # For 3DPPE, batch=1 only
            decoded_scores, decoded_labels, decoded_bboxes = decode_nms_free_preds(
                preds_dicts, max_num=self.max_num,
            )
            return (
                decoded_scores,
                decoded_labels,
                decoded_bboxes,
                preds_dicts["all_cls_scores"][-1, 0],
                preds_dicts["all_bbox_preds"][-1, 0],
            )

    model.eval()
    model = model.float()

    tm = Trt3DPPEContainer(model.module)
    arrs = [
        torch.from_numpy(np.random.uniform(-128, 128, size=(1, num_views, 3, input_h, input_w))).float(),
        torch.from_numpy(sensor_meta["intrinsics"][None, ...]).float(),
        torch.from_numpy(sensor_meta["extrinsics"][None, ...]).float(),
        torch.from_numpy(sensor_meta["img2lidar"][None, ...]).float(),
    ]
    input_names = ["img", "intrinsics", "extrinsics", "img2lidar"]
    output_names = [
        "decoded_scores",
        "decoded_labels",
        "decoded_bboxes",
        "all_cls_scores",
        "all_bbox_preds",
    ]  # 必须输出 all_cls_scores 和 all_bbox_preds 以保证 ONNX 模型的数值准确性，虽然这两个输出在实际部署中可能不会被使用
    # 原因可能是后续的算子优化（如常量折叠）在没有中间结果输出的情况下会错误地将部分计算结果当作常量，从而导致数值不准确
    # 相机参数不能出现numpy类型，否则会被折叠成常量
    # 由于适配不同机器时需要修改相机参数，模型本身需要对不同的相机参数具有泛化性。

    tm = tm.float()
    tm.cpu()
    tm.eval()
    tm.training = False

    tm_args = tuple(arrs)

    filename = args.section + ".onnx"
    export_dir = os.path.dirname(args.checkpoint)
    onnx_path = os.path.join(export_dir, filename)
    with torch.no_grad():
        torch.onnx.export(
            tm,
            deepcopy(tm_args),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,  # `False` when debugging, `True` for best performance and simplify (but with bugs)
            verbose=False,
            opset_version=args.opset_version,
        )

    onnx_model = onnx.load(onnx_path)
    onnx_model_simp, check = simplify(onnx_model)
    simpified_onnx_path = os.path.join(export_dir, "simplify_" + filename)
    onnx.save(onnx_model_simp, simpified_onnx_path)
    print(f"[ONNX] simplify check: {'PASS' if check else 'FAIL'}")

    if not args.no_validate:
        if ort is None:
            raise ImportError("onnxruntime is required for ONNX numerical validation.")
        with torch.no_grad():
            torch_outputs = tm(*deepcopy(tm_args))

        ort_session = ort.InferenceSession(
            simpified_onnx_path,
            providers=["CPUExecutionProvider"],
        )
        ort_inputs = {
            input_names[idx]: to_numpy(deepcopy(tm_args)[idx])
            for idx in range(len(input_names))
        }
        ort_outputs = ort_session.run(output_names, ort_inputs)
        passed = validate_onnx_output(
            torch_outputs,
            ort_outputs,
            output_names,
            rtol=args.rtol,
            atol=args.atol,
        )
        if not passed:
            raise AssertionError("ONNX numerical validation failed.")

    print(args.section + " onnx export success!")


if __name__ == "__main__":
    main()


# pip install onnx onnxruntime onnxsim
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m tools.pth2onnx --section petr projects/configs/petr/petr_vovnet_gridmask_p4_800x320.py --checkpoint work_dirs/petr/epoch_24.pth
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m tools.pth2onnx --section 3dppe projects/configs/petr_depth/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg.py --checkpoint work_dirs/3dppe/epoch_24.pth
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m tools.pth2onnx --section 3dppe projects/configs/petr_depth/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg.py --checkpoint work_dirs/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg/epoch_24.pth
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m tools.pth2onnx --section 3dppe_v_pe projects/configs/petr_depth/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis.py --checkpoint work_dirs/petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg_thesis/latest.pth

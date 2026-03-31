_base_ = [
    './petr_depth_3dpe_dfl_vovnet_wogridmask_p4_800x320_pdg.py'
]

# MobileNetV4 checkpoint is loaded from the local timm-exported directory.
model = dict(
    img_backbone=dict(
        _delete_=True,
        type='TimmModel',
        model_name='mobilenetv4_conv_large.e600_r384_in1k',
        pretrained=False,
        checkpoint_path='ckpts/mobilenetv4_conv_large.e600_r384_in1k/pytorch_model.bin',
        features_only=True,
        out_indices=(3, 4),
        in_chans=3,
        frozen_stages=-1,
        norm_eval=True),
    img_neck=dict(
        _delete_=True,
        type='CPFPN',
        in_channels=[192, 960],
        out_channels=256,
        num_outs=2))

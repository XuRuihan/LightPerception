import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet.models.backbones.resnet import BasicBlock
from mmdet.models.builder import build_loss

from mmcv.utils import Registry, build_from_cfg
from .conv import Conv

DepthNet = Registry('depthnet')


def build_depthnet(cfg, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return build_from_cfg(cfg, DepthNet, default_args)


class CustomDepthBlock(nn.Module):
    """Universal inverted residual block for depth prediction."""

    def __init__(self, channels):
        super(CustomDepthBlock, self).__init__()
        hidden_channels = channels
        self.expand_conv = Conv(channels, hidden_channels * 2, act=False)
        self.depthwise_conv = Conv(hidden_channels, hidden_channels, k=5, s=1, p=2, g=4, act=False)
        self.project_conv = Conv(hidden_channels, channels, k=3, s=1, p=1, act=False)

    def forward(self, x):
        identity = x
        x = self.expand_conv(x)
        x1, x2 = x.chunk(2, 1)
        x = self.depthwise_conv(F.relu(x1)) * F.hardsigmoid(x2)
        x = self.project_conv(x)
        x = x + identity
        return x


def build_depth_block(block_type, channels):
    if block_type == 'basic':
        return BasicBlock(channels, channels)
    if block_type == 'custom':
        return CustomDepthBlock(channels)
    raise ValueError(f'Unsupported depth block type: {block_type}')


def normpos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    """Sine/cosine positional encoding for 2D coordinates.
    for `pos` within range (-1, 1), the coefficient is in the range of [10000^(-1/2), 10000^(1/2)].
    Args:
        pos: (..., 2), in the normalized coordinate space, where the range is (approximately) `(-1, 1)`.
        num_pos_feats: The dimension of the positional encoding. Default: 128.
        temperature: The temperature used in the positional encoding. Default: 10000.
    Returns:
        pos_emb: (..., C), where C = num_pos_feats * 2
    """
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (0.5 - 2 * (dim_t // 2) / num_pos_feats)  # `pos` in [-1, 1]

    pos = pos[..., None] * dim_t
    pos = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=-1).flatten(-2)  # flatten sin and cos encodings
    return pos.flatten(-2)  # flatten x and y positional encodings


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    """Sine/cosine positional encoding for 2D coordinates.
    for `pos` in the range [-1000, 1000], the coefficient is in the range of [10000^(-1), 1].
    Args:
        pos: (..., 2), where the range is (approximately) `(-1000, 1000)`.
        num_pos_feats: The dimension of the positional encoding. Default: 128.
        temperature: The temperature used in the positional encoding. Default: 10000.
    Returns:
        pos_emb: (..., C), where C = num_pos_feats * 2
    """
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (-2 * (dim_t // 2) / num_pos_feats)  # `pos` in [-1000, 1000]

    pos = pos[..., None] * dim_t
    pos = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=-1).flatten(-2)  # flatten sin and cos encodings
    return pos.flatten(-2)  # flatten x and y positional encodings


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            x: (B, C, H, W)
        """
        x1 = self.aspp1(x)      # (B, C, H, W)
        x2 = self.aspp2(x)      # (B, C, H, W)
        x3 = self.aspp3(x)      # (B, C, H, W)
        x4 = self.aspp4(x)      # (B, C, H, W)
        x5 = self.global_avg_pool(x)    # (B, C, 1, 1)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)      # (B, C, H, W)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)      # (B, 5*C, H, W)

        x = self.conv1(x)   # (B, C, H, W)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        """
        Args:
            x: (B, C, H, W)
            x_se: (B, C, 1, 1)
        Returns:

        """
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


@DepthNet.register_module()
class CameraAwareDepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels, depth_channels, num_params1=18,
                 num_params2=6, with_depth_correction=False, with_context_encoder=False, 
                 with_pgd=False, depth_block_type='basic'):
        super(CameraAwareDepthNet, self).__init__()
        self.in_channels = in_channels
        self.context_channels = context_channels
        self.depth_channels = depth_channels
        self.mid_channels = mid_channels
        self.depth_block_type = depth_block_type

        self.reduce_conv = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
        )
        if with_context_encoder:
            self.context_conv = nn.Sequential(
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
                nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.context_conv = nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)

        self.coord_num_pos_feats = mid_channels // 2
        self.coord_embed_channels = self.coord_num_pos_feats * 2
        self.depth_coord_proj = Conv(self.coord_embed_channels, mid_channels, act=False)
        self.depth_focal_proj = Conv(self.coord_embed_channels, mid_channels, act=False)
        if with_depth_correction:
            self.depth_stem = nn.Sequential(
                build_depth_block(depth_block_type, mid_channels),
                build_depth_block(depth_block_type, mid_channels),
                build_depth_block(depth_block_type, mid_channels),
                ASPP(mid_channels, mid_channels),
            )
            self.depth_prob_conv = nn.Sequential(
                build_conv_layer(cfg=dict(
                    type="Conv2d",  # type='DCN',
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    padding=1,
                    groups=4,
                    # im2col_step=128,
                )),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(),
                nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.depth_stem = torch.nn.Identity()
            self.depth_prob_conv = nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)

        self.with_pgd = with_pgd
        if self.with_pgd:
            self.fuse_lambda = nn.Parameter(torch.tensor(10e-5))
            self.depth_direct_conv = nn.Sequential(
                build_conv_layer(cfg=dict(
                    type="Conv2d",  # type='DCN',
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    padding=1,
                    groups=4,
                    # im2col_step=128,
                )),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(),
                nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0)
            )

    def _get_normalized_coord_features(self, intrinsics, image_shape, feat_shape):
        """
        Args:
            intrinsics: (B, N_view, 3, 3)
            image_shape: (B, N_view, 2), (img_h, img_w)
            feat_shape: (H, W)
        Returns:
            coord_feat: (B*N_view, C_mid, H, W)
        """
        feat_h, feat_w = feat_shape
        device = intrinsics.device
        dtype = intrinsics.dtype

        intrinsics = intrinsics.flatten(0, 1)
        image_shape = image_shape.to(device=device, dtype=dtype).flatten(0, 1)
        img_h, img_w = image_shape.unbind(dim=-1)

        fx = intrinsics[:, 0, 0].abs().clamp(min=1e-5).view(-1, 1, 1)
        fy = intrinsics[:, 1, 1].abs().clamp(min=1e-5).view(-1, 1, 1)
        cx = intrinsics[:, 0, 2].view(-1, 1, 1)
        cy = intrinsics[:, 1, 2].view(-1, 1, 1)

        u = (torch.arange(feat_w, device=device, dtype=dtype) + 0.5).view(1, 1, feat_w)
        v = (torch.arange(feat_h, device=device, dtype=dtype) + 0.5).view(1, feat_h, 1)
        grid_u = u * (img_w.view(-1, 1, 1) / feat_w / fx)
        grid_v = v * (img_h.view(-1, 1, 1) / feat_h / fy)

        norm_u = (grid_u - cx / fx).expand(-1, feat_h, -1)
        norm_v = (grid_v - cy / fy).expand(-1, -1, feat_w)
        norm_coords = torch.stack((norm_u, norm_v), dim=-1)

        coord_feat = normpos2posemb2d(norm_coords, num_pos_feats=self.coord_num_pos_feats)
        coord_feat = self.depth_coord_proj(coord_feat.permute(0, 3, 1, 2).contiguous())
        # return torch.sigmoid(coord_feat)
        norm_focal = torch.cat((fx, fy), dim=1).squeeze(-1)
        focal_feat = pos2posemb2d(norm_focal, num_pos_feats=self.coord_num_pos_feats)
        focal_feat = self.depth_focal_proj(focal_feat[..., None, None])
        return torch.sigmoid(coord_feat + focal_feat)

    def forward(self, x, intrinsics, extrinsics, image_shape=None):
        """
        Args:
            x: img feature map  (B*N_view, C, H, W)
            intrinsics: (B, N_view, 3, 3)
            extrinsics: (B, N_view, 4, 4)
            image_shape: (B, N_view, 2), (img_h, img_w)
        Returns:
            depth:  (B*N_view, D, H, W)
            context: (B*N_view, C_context, H, W)
        """
        B, N_view = intrinsics.shape[:2]
        intrinsics = intrinsics[..., :2, :]  # 6
        extrinsics = extrinsics[..., :3, :]  # 12

        # (B*N_view, C, H, W) --> (B*N_view, C_mid, H, W)
        x = self.reduce_conv(x)
        depth_input, context_input = x, x
        context = self.context_conv(context_input)  # (B*N_view, C_context, H, W)

        # depth feature 显式注入归一化像素坐标 (u-cx)/fx, (v-cy)/fy
        if image_shape is None:
            raise ValueError('image_shape is required for CameraAwareDepthNet.')
        coord_feat = self._get_normalized_coord_features(
            intrinsics=intrinsics,
            image_shape=image_shape,
            feat_shape=depth_input.shape[-2:])
        depth_feat = depth_input * coord_feat

        depth_stem = self.depth_stem(depth_feat)
        depth_prob = self.depth_prob_conv(depth_stem)  # (B*N_view, D, H, W)
        if not self.with_pgd:
            return depth_prob, context
        else:
            depth_direct = self.depth_direct_conv(depth_stem)
            return depth_prob, context, depth_direct


@DepthNet.register_module()
class VanillaDepthNet(nn.Module):
    def __init__(self, in_channels, context_channels, depth_channels, mid_channels=None, with_depth_correction=False):
        super(VanillaDepthNet, self).__init__()
        self.in_channels = in_channels
        self.context_channels = context_channels
        self.depth_channels = depth_channels
        self.mid_channels = mid_channels

        if mid_channels is not None:
            self.reduce_conv = ConvModule(
                        in_channels=in_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=dict(type='BN2d'),
                        )
        else:
            mid_channels = in_channels
            self.reduce_conv = None
        self.context_conv = nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        if with_depth_correction:
            self.depth_conv = nn.Sequential(
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
                ASPP(mid_channels, mid_channels),
                build_conv_layer(cfg=dict(
                    type="Conv2d",  # type='DCN',
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    padding=1,
                    groups=4,
                    # im2col_step=128,
                )),
                nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.depth_conv = nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, intrinsics=None, extrinsics=None, image_shape=None):
        """
        Args:
            x: img feature map  (B*N_view, C, H, W)
        Returns:
            depth:  (B*N_view, D, H, W)
            context: (B*N_view, C, H, W)
        """
        # (B*N_view, C, H, W) --> (B*N_view, C_mid, H, W)
        if self.reduce_conv is not None:
            x = self.reduce_conv(x)
        context = self.context_conv(x)            # (B*N_view, C_context, H, W)
        depth = self.depth_conv(x)      # (B*N_view, D, H, W)

        return depth, context


class SELikeModule(nn.Module):
    def __init__(self, in_channel=256, feat_channel=256, intrinsic_channel=6):
        super(SELikeModule, self).__init__()
        self.input_conv = nn.Conv2d(in_channel, feat_channel, kernel_size=1, padding=0)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(intrinsic_channel),
            nn.Linear(intrinsic_channel, feat_channel),
            nn.Sigmoid())

    def forward(self, x, cam_params):
        """
        Args:
            x: (B*N_view, C_in, H, W)
            cam_params: (B*N_view, 6)

        Returns:
            x:  (B*N_view, C, H, W)
        """
        x = self.input_conv(x)  # (B*N_view, C, H, W)
        b, c, _, _ = x.shape
        y = self.fc(cam_params).view(b, c, 1, 1)    # (B*N_view, C, 1, 1)
        return x * y.expand_as(x)


@DepthNet.register_module()
class CameraAwareDepthNetV2(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels, depth_channels, num_params=6,
                 with_depth_correction=False, with_context_encoder=False,
                 with_pgd=False):
        super(CameraAwareDepthNetV2, self).__init__()
        self.in_channels = in_channels
        self.context_channels = context_channels
        self.depth_channels = depth_channels
        self.mid_channels = mid_channels

        if mid_channels is not None:
            self.reduce_conv = ConvModule(
                        in_channels=in_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=dict(type='BN2d'),
                        )
        else:
            mid_channels = in_channels
            self.reduce_conv = None

        if with_context_encoder:
            self.context_conv = nn.Sequential(
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
                nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.context_conv = nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)

        self.se = SELikeModule(in_channel=self.mid_channels, feat_channel=self.mid_channels,
                               intrinsic_channel=num_params)

        if with_depth_correction:
            self.depth_stem = nn.Sequential(
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
                ASPP(mid_channels, mid_channels),
            )
            self.depth_prob_conv = nn.Sequential(
                build_conv_layer(cfg=dict(
                    type="Conv2d",  # type='DCN',
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    padding=1,
                    groups=4,
                    # im2col_step=128,
                )),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.depth_stem = torch.nn.Identity()
            self.depth_prob_conv = nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)

        self.with_pgd = with_pgd
        if self.with_pgd:
            self.fuse_lambda = nn.Parameter(torch.tensor(10e-5))
            self.depth_direct_conv = nn.Sequential(
                build_conv_layer(cfg=dict(
                    type="Conv2d",  # type='DCN',
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    padding=1,
                    groups=4,
                    # im2col_step=128,
                )),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(),
                nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x, intrinsics, extrinsics, image_shape=None):
        """
        Args:
            x: img feature map  (B*N_view, C, H, W)
            intrinsics: (B, N_view, 3, 3)
            extrinsics: (B, N_view, 4, 4)
        Returns:
            depth:  (B*N_view, D, H, W)
            context: (B*N_view, C_context, H, W)
        """
        B, N_view = intrinsics.shape[:2]
        intrinsics = intrinsics[..., :2, :].contiguous()   # 6
        extrinsics = extrinsics[..., :3, :].contiguous()   # 12
        intrinsics = intrinsics.view(B*N_view, -1)      # (B*N_view, 6)
        extrinsics = extrinsics.view(B*N_view, -1)      # (B*N_view, 12)

        # (B*N_view, C, H, W) --> (B*N_view, C_mid, H, W)
        if self.reduce_conv is not None:
            x = self.reduce_conv(x)
        context = self.context_conv(x)  # (B*N_view, C_context, H, W)

        depth = self.se(x, intrinsics)  # (B*N_view, C_mid, H, W)
        if not self.with_pgd:
            depth_stem = self.depth_stem(depth)
            depth_prob = self.depth_prob_conv(depth_stem)  # (B*N_view, D, H, W)
            return depth_prob, context
        else:
            depth_stem = self.depth_stem(depth)
            depth_prob = self.depth_prob_conv(depth_stem)
            depth_direct = self.depth_direct_conv(depth_stem)

            return depth_prob, context, depth_direct

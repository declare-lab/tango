"""
Adapted from `https://github.com/pytorch/vision`.
Modified by Vladimir Iashin, 2021.
"""
import math
import sys
from contextlib import redirect_stdout

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.listconfig import ListConfig
from torch.hub import load_state_dict_from_url
from torch.nn.modules.utils import _ntuple
from torchvision.models.inception import BasicConv2d
from torchvision.models.inception import Inception3 as TorchVisionInception3


class FeatureExtractorInceptionV3(TorchVisionInception3):
    def __init__(
        self, name, features_list, feature_extractor_weights_path=None, **kwargs
    ):
        """Build pretrained InceptionV3

        Parameters
        ----------
        features_list: list
            A list of feature names from the list of provided by this extractor,
            which will be produced for each input
        feature_extractor_weights_path: str
            Path to the pretrained Inception model weights in PyTorch format.
            Refer to inception_features.py:__main__ for making your own.
            By default downloads the checkpoint from internet.
        """
        super().__init__(num_classes=1008, **kwargs)
        self.input_image_size = 299
        self.provided_feats = ("64", "192", "768", "2048", "logits_unbiased", "logits")
        self.features_list = list(features_list)

        assert type(name) is str, "Feature extractor name must be a string"
        assert type(features_list) in (
            list,
            tuple,
            ListConfig,
        ), "Wrong features list type"
        assert all(
            (a in self.provided_feats for a in features_list)
        ), "Requested features arent on the list"
        assert len(features_list) == len(
            set(features_list)
        ), "Duplicate features requested"

        PT_INCEPTION_URL = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.Mixed_7b = InceptionE_1(1280)
        self.Mixed_7c = InceptionE_2(2048)
        self.AuxLogits = nn.Identity()

        if feature_extractor_weights_path is None:
            with redirect_stdout(sys.stderr):
                state_dict = load_state_dict_from_url(PT_INCEPTION_URL, progress=True)
        else:
            state_dict = torch.load(feature_extractor_weights_path)
        self.load_state_dict(state_dict)

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        assert (
            torch.is_tensor(x) and x.dtype == torch.uint8
        ), "Expecting x as torch.Tensor, dtype=torch.uint8"
        features = {}
        remaining_features = self.features_list.copy()

        x = x.float()
        # N x 3 x ? x ?

        x = interpolate_bilinear_2d_like_tensorflow1x(
            x,
            size=(self.input_image_size, self.input_image_size),
            align_corners=False,
        )
        # N x 3 x 299 x 299

        # x = (x - 128) * torch.tensor(0.0078125, dtype=torch.float32, device=x.device)  # happening in graph
        x = (x - 128) / 128  # but this gives bit-exact output _of this step_ too
        # N x 3 x 299 x 299

        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73

        if "64" in remaining_features:
            features["64"] = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            remaining_features.remove("64")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35

        if "192" in remaining_features:
            features["192"] = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            remaining_features.remove("192")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17

        if "768" in remaining_features:
            features["768"] = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            remaining_features.remove("768")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        x = self.avgpool(x)
        # N x 2048 x 1 x 1

        x = torch.flatten(x, 1)
        # N x 2048

        if "2048" in remaining_features:
            features["2048"] = x
            remaining_features.remove("2048")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        if "logits_unbiased" in remaining_features:
            x = x.mm(self.fc.weight.T)
            # N x 1008 (num_classes)
            features["logits_unbiased"] = x
            remaining_features.remove("logits_unbiased")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

            x = x + self.fc.bias.unsqueeze(0)
        else:
            x = self.fc(x)
            # N x 1008 (num_classes)

        features["logits"] = x
        return tuple(features[a] for a in self.features_list)

    @staticmethod
    def get_provided_features_list():
        return

    def get_requested_features_list(self):
        return self.features_list

    def get_name(self):
        return self.name

    def convert_features_tuple_to_dict(self, features):
        """
        The only compound return type of the forward function amenable to JIT tracing is tuple.
        This function simply helps to recover the mapping.
        """
        message = "Features must be the output of forward function"
        assert type(features) is tuple and len(features) == len(
            self.features_list
        ), message
        return dict(
            ((name, feature) for name, feature in zip(self.features_list, features))
        )


class InceptionA(nn.Module):
    """Block from torchvision patched to be compatible with TensorFlow implementation"""

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in its average calculation
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    """Block from torchvision patched to be compatible with TensorFlow implementation"""

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in its average calculation
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE_1(nn.Module):
    """First InceptionE block from torchvision patched to be compatible with TensorFlow implementation"""

    def __init__(self, in_channels):
        super(InceptionE_1, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in its average calculation
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE_2(nn.Module):
    """Second InceptionE block from torchvision patched to be compatible with TensorFlow implementation"""

    def __init__(self, in_channels):
        super(InceptionE_2, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: TensorFlow Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


def interpolate_bilinear_2d_like_tensorflow1x(
    input, size=None, scale_factor=None, align_corners=None, method="slow"
):
    r"""Down/up samples the input to either the given :attr:`size` or the given :attr:`scale_factor`

    Epsilon-exact bilinear interpolation as it is implemented in TensorFlow 1.x:
    https://github.com/tensorflow/tensorflow/blob/f66daa493e7383052b2b44def2933f61faf196e0/tensorflow/core/kernels/image_resizer_state.h#L41
    https://github.com/tensorflow/tensorflow/blob/6795a8c3a3678fb805b6a8ba806af77ddfe61628/tensorflow/core/kernels/resize_bilinear_op.cc#L85
    as per proposal:
    https://github.com/pytorch/pytorch/issues/10604#issuecomment-465783319

    Related materials:
    https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35
    https://jricheimer.github.io/tensorflow/2019/02/11/resize-confusion/
    https://machinethink.net/blog/coreml-upsampling/

    Currently only 2D spatial sampling is supported, i.e. expected inputs are 4-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x height x width`.

    Args:
        input (Tensor): the input tensor
        size (Tuple[int, int]): output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.
        align_corners (bool, optional): Same meaning as in TensorFlow 1.x.
        method (str, optional):
            'slow' (1e-4 L_inf error on GPU, bit-exact on CPU, with checkerboard 32x32->299x299), or
            'fast' (1e-3 L_inf error on GPU and CPU, with checkerboard 32x32->299x299)
    """
    if method not in ("slow", "fast"):
        raise ValueError('how_exact can only be one of "slow", "fast"')

    if input.dim() != 4:
        raise ValueError("input must be a 4-D tensor")

    if align_corners is None:
        raise ValueError(
            "align_corners is not specified (use this function for a complete determinism)"
        )

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        if (
            scale_factor is not None
            and isinstance(scale_factor, tuple)
            and len(scale_factor) != dim
        ):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
            )

    is_tracing = torch._C._get_tracing_state()

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            if is_tracing:
                return [torch.tensor(i) for i in size]
            else:
                return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7

        # make scale_factor a tensor in tracing so constant doesn't get baked in
        if is_tracing:
            return [
                (
                    torch.floor(
                        (
                            input.size(i + 2).float()
                            * torch.tensor(scale_factors[i], dtype=torch.float32)
                        ).float()
                    )
                )
                for i in range(dim)
            ]
        else:
            return [
                int(math.floor(float(input.size(i + 2)) * scale_factors[i]))
                for i in range(dim)
            ]

    def tf_calculate_resize_scale(in_size, out_size):
        if align_corners:
            if is_tracing:
                return (in_size - 1) / (out_size.float() - 1).clamp(min=1)
            else:
                return (in_size - 1) / max(1, out_size - 1)
        else:
            if is_tracing:
                return in_size / out_size.float()
            else:
                return in_size / out_size

    out_size = _output_size(2)
    scale_x = tf_calculate_resize_scale(input.shape[3], out_size[1])
    scale_y = tf_calculate_resize_scale(input.shape[2], out_size[0])

    def resample_using_grid_sample():
        grid_x = torch.arange(0, out_size[1], 1, dtype=input.dtype, device=input.device)
        grid_x = grid_x * (2 * scale_x / (input.shape[3] - 1)) - 1

        grid_y = torch.arange(0, out_size[0], 1, dtype=input.dtype, device=input.device)
        grid_y = grid_y * (2 * scale_y / (input.shape[2] - 1)) - 1

        grid_x = grid_x.view(1, out_size[1]).repeat(out_size[0], 1)
        grid_y = grid_y.view(out_size[0], 1).repeat(1, out_size[1])

        grid_xy = torch.cat(
            (grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), dim=2
        ).unsqueeze(0)
        grid_xy = grid_xy.repeat(input.shape[0], 1, 1, 1)

        out = F.grid_sample(
            input, grid_xy, mode="bilinear", padding_mode="border", align_corners=True
        )
        return out

    def resample_manually():
        grid_x = torch.arange(0, out_size[1], 1, dtype=input.dtype, device=input.device)
        grid_x = grid_x * torch.tensor(scale_x, dtype=torch.float32)
        grid_x_lo = grid_x.long()
        grid_x_hi = (grid_x_lo + 1).clamp_max(input.shape[3] - 1)
        grid_dx = grid_x - grid_x_lo.float()

        grid_y = torch.arange(0, out_size[0], 1, dtype=input.dtype, device=input.device)
        grid_y = grid_y * torch.tensor(scale_y, dtype=torch.float32)
        grid_y_lo = grid_y.long()
        grid_y_hi = (grid_y_lo + 1).clamp_max(input.shape[2] - 1)
        grid_dy = grid_y - grid_y_lo.float()

        # could be improved with index_select
        in_00 = input[:, :, grid_y_lo, :][:, :, :, grid_x_lo]
        in_01 = input[:, :, grid_y_lo, :][:, :, :, grid_x_hi]
        in_10 = input[:, :, grid_y_hi, :][:, :, :, grid_x_lo]
        in_11 = input[:, :, grid_y_hi, :][:, :, :, grid_x_hi]

        in_0 = in_00 + (in_01 - in_00) * grid_dx.view(1, 1, 1, out_size[1])
        in_1 = in_10 + (in_11 - in_10) * grid_dx.view(1, 1, 1, out_size[1])
        out = in_0 + (in_1 - in_0) * grid_dy.view(1, 1, out_size[0], 1)

        return out

    if method == "slow":
        out = resample_manually()
    else:
        out = resample_using_grid_sample()

    return out

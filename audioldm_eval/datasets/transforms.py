import torch
from specvqgan.modules.losses.vggishish.transforms import Crop


class FromMinusOneOneToZeroOne(object):
    """Actually, it doesnot do [-1, 1] --> [0, 1] as promised. It would, if inputs would be in [-1, 1]
    but reconstructed specs are not."""

    def __call__(self, item):
        item["image"] = (item["image"] + 1) / 2
        return item


class CropNoDict(Crop):
    def __init__(self, cropped_shape, random_crop=None):
        super().__init__(cropped_shape=cropped_shape, random_crop=random_crop)

    def __call__(self, x):
        # albumentations expect an ndarray of size (H, W, ...) but we have tensor of size (B, H, W).
        # we will assume that the batch-dim (B) is out "channel" dim and permute it to the end.
        # Finally, we change the type back to Torch.Tensor.
        x = self.preprocessor(image=x.permute(1, 2, 0).numpy())["image"].transpose(
            2, 0, 1
        )
        return torch.from_numpy(x)


class GetInputFromBatchByKey(object):  # get image from item dict
    def __init__(self, input_key):
        self.input_key = input_key

    def __call__(self, item):
        return item[self.input_key]


class ToFloat32(object):
    def __call__(self, item):
        return item.float()

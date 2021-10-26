from torch import Tensor
from torchaudio import transforms
import torch_audiomentations

from hw_asr.augmentations.base import AugmentationBase


class AddColoredNoise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.augmentation = torch_audiomentations.AddColoredNoise(*args, **kwargs)

    def __call__(self, data: Tensor):
        return self.augmentation(data.unsqueeze(1)).squeeze(1)



class PolarityInversion(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.augmentation = torch_audiomentations.PolarityInversion(*args, **kwargs)

    def __call__(self, data: Tensor):
        return self.augmentation(data.unsqueeze(1)).squeeze(1)
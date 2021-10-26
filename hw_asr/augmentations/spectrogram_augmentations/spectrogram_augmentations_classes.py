from torch import Tensor
from torchaudio import transforms
import torch

from hw_asr.augmentations.base import AugmentationBase


class FrequencyMasking(AugmentationBase):
    def __init__(self, frequency, p, *args, **kwargs):
        self.augmentation = transforms.FrequencyMasking(frequency)
        self.p = p

    def __call__(self, data: Tensor):
        if torch.rand(1)[0] > self.p:
            return data
        return self.augmentation(data)


class TimeMasking(AugmentationBase):
    def __init__(self, time, p,*args, **kwargs):
        self.augmentation = transforms.TimeMasking(time)
        self.p = p

    def __call__(self, data: Tensor):
        if torch.rand(1)[0] > self.p:
            return data
        return self.augmentation(data)
from torch import Tensor
from torchaudio import transforms

from hw_asr.augmentations.base import AugmentationBase


class FrequencyMasking(AugmentationBase):
    def __init__(self, frequency, *args, **kwargs):
        self.augmentation = transforms.FrequencyMasking(frequency)

    def __call__(self, data: Tensor):
        return self.augmentation(data)


class TimeMasking(AugmentationBase):
    def __init__(self, time, *args, **kwargs):
        self.augmentation = transforms.TimeMasking(time)

    def __call__(self, data: Tensor):
        return self.augmentation(data)
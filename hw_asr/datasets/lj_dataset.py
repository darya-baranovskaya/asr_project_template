from datasets import load_dataset

from hw_asr.base.base_dataset import BaseDataset


class LJDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        data = load_dataset("lj_speech")
        super().__init__(data, *args, **kwargs)



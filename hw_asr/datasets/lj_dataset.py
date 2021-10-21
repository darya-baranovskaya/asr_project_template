from datasets import load_dataset
from hw_asr.datasets.custom_audio_dataset import CustomAudioDataset
from hw_asr.base.base_dataset import BaseDataset
import torchaudio



class LJDataset(BaseDataset):
    def __init__(self, mode, *args, **kwargs):
        dataset = load_dataset("lj_speech")
        dataset = dataset.data[mode]

        data = []
        for i in range(dataset.num_rows):
            path = str(dataset['file'][i])
            entry = {'text':str(dataset['text'][i]), "path": path }
            t_info = torchaudio.info(entry["path"])
            entry["audio_len"] = t_info.num_frames / t_info.sample_rate
            data.append(entry)

        super().__init__(data, *args, **kwargs)
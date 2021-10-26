import logging
from typing import List
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    # TODO: your code here
    # raise NotImplementedError
    for key in ['text_encoded', 'spectrogram', 'audio']:
        max_len = max(list(map(lambda x: x[key].shape[-1], dataset_items)))
        result_batch[key] = torch.cat(list(
            map(lambda x: F.pad(x[key], (0, max_len - x[key].shape[-1])), dataset_items)
        ), dim=0)

    for key in ['text', 'duration', 'audio_path']:
        result_batch[key] = [x[key] for x in dataset_items]
    result_batch['spectrogram'] = result_batch['spectrogram']#.permute(0, 2, 1)
    result_batch['text_encoded_length'] = torch.tensor(list(map(lambda x: x['text_encoded'].shape[1], dataset_items)), dtype=torch.int32)
    result_batch['spectrogram_length'] = torch.tensor(list(map(lambda x: x['spectrogram'].shape[2], dataset_items)), dtype=torch.int32)
    return result_batch


'''
result_batch['text_encoded'].shape
Out[2]: torch.Size([32, 160])
result_batch['spectrogram'].shape
Out[3]: torch.Size([32, 128, 795])
result_batch['audio'].shape
Out[4]: torch.Size([32, 158939])
len(result_batch['text'])
Out[6]: 32
len(result_batch['text'][0])
Out[8]: 97
len(result_batch['duration'])
Out[11]: 32
len(result_batch['audio_path'])
Out[14]: 32
len(result_batch['text_encoded_length'])
Out[16]: 32
result_batch['text_encoded_length']
Out[17]: 
tensor([ 90,  84, 117, 133, 160, 101,  63, 117,  56,  96, 155, 105, 101,  76,
         89, 154, 126, 151,  48,  99, 121, 120, 139, 112,  94,  83,  47,  75,
         76,  45, 118, 132], dtype=torch.int32)
'''
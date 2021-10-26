from torch.utils.data import Sampler
import torch
import random


class GroupLengthBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, batches_per_group=20):
        super().__init__(data_source)
        # TODO: your code here (optional)
        self.data_source = data_source
        self.data_len = len(data_source)
        self.batch_size = batch_size
        self.group_size = batches_per_group * batch_size
        self.batches_per_group = batches_per_group
        self.n_groups = len(data_source) // self.group_size
        sorted_idxs = torch.argsort(torch.tensor([len(x['audio']) for x in data_source]))
        self.groups = {i: sorted_idxs[i * self.group_size : i * (self.group_size + 1)] for i in range(self.n_groups)}


    def __iter__(self):
        num_yielded = 0
        while num_yielded < self.data_len:
            cur_group = random.randint(0, self.n_groups - 1)
            cur_idxs = torch.randperm(self.groups[cur_group])[:self.batch_size].tolist()
            num_yielded += self.batch_size
            yield cur_idxs
        # raise NotImplementedError

    def __len__(self):
        # raise NotImplementedError
        return self.data_len // self.batch_size
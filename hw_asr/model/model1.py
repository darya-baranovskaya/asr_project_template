from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel

class MainModel1(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, num_layers=3,*args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.bn = nn.BatchNorm2d(n_feats)
        self.gru = nn.GRU(n_feats, fc_hidden, num_layers=num_layers,  bidirectional=True)
        self.fc = nn.Linear(fc_hidden * 2, n_class)

    def forward(self, spectrogram, *args, **kwargs):
        x = self.bn(spectrogram)
        x, _ = self.gru(x)
        x = self.fc(x)
        # print('output size=', x.size())
        return  {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
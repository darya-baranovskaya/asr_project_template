from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel

class MainModel1(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, num_layers=3, num_heads=4, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.bn1 = nn.BatchNorm1d(n_feats)
        self.attn = nn.MultiheadAttention(n_feats, num_heads, dropout=0.0, bias=True, batch_first=True)
        self.gru = nn.GRU(n_feats, fc_hidden, num_layers=num_layers,  bidirectional=True)
        self.fc = nn.Linear(fc_hidden * 2, n_class)

    def forward(self, spectrogram, *args, **kwargs):
        x = self.bn1(spectrogram.permute(0, 2, 1)).permute(0, 2, 1)
        # print(x.shape)
        # x = spectrogram
        x = self.attn(x, x, x)
        x, _ = self.gru(spectrogram)
        x = self.fc(x)
        # print('output size=', x.size())
        return  {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
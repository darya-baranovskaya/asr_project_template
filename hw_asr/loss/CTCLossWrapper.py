import torch
from torch import Tensor
from torch.nn import CTCLoss
from torch import functional as F


class CTCLossWrapper(CTCLoss):
    def forward(self, *args, **kwargs) -> Tensor:
        log_probs = torch.transpose(kwargs["log_probs"], 0, 1)
        input_lengths = kwargs["log_probs_length"]
        targets = kwargs["text_encoded"]
        target_lengths = kwargs["text_encoded_length"]

        return super().forward(log_probs=log_probs, targets=targets,
                               input_lengths=input_lengths, target_lengths=target_lengths)


class CTCLossWrapperWithLengthPenalty(CTCLoss):
    __constants__ = ['blank', 'reduction']
    blank: int
    zero_infinity: bool

    def __init__(self, blank: int = 0, reduction: str = 'mean', zero_infinity: bool = False, alpha: int=0.0001, eta:int=0.0001):
        super(CTCLoss, self).__init__(reduction=reduction)
        self.blank = blank
        self.zero_infinity = zero_infinity
        self.eta = eta
        self.alpha = alpha

    def forward(self, *args, **kwargs) -> Tensor:
        log_probs = torch.transpose(kwargs["log_probs"], 0, 1)
        input_lengths = kwargs["log_probs_length"]
        targets = kwargs["text_encoded"]
        target_lengths = kwargs["text_encoded_length"]

        argmax_log_probs = torch.argmax(log_probs[:, :, :], dim=-1).cuda()
        argmax_log_probs = argmax_log_probs / (argmax_log_probs + 0.0001)
        penalty_for_empty = self.alpha * 1 / (torch.nn.MSELoss()(torch.nn.MSELoss()(argmax_log_probs, torch.zeros(argmax_log_probs.shape).cuda()) + self.eta).cuda())
        return super().forward(log_probs=log_probs, targets=targets,
                               input_lengths=input_lengths, target_lengths=target_lengths) + penalty_for_empty
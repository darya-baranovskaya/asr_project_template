from typing import List, Tuple

import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder
from fast_ctc_decode import beam_search


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        # TODO: your code here
        # raise NotImplementedError()
        prev = None
        result =[]
        for ind in inds:
            ind = ind.item() if torch.is_tensor(ind) else ind
            if ind == prev:
                continue
            if ind != self.char2ind[self.EMPTY_TOK]:
                result.append(ind)
            prev = ind
        return ''.join([self.ind2char[ind] for ind in result])

    # def ctc_beam_search(self, probs: torch.tensor, probs_length,
    #                     beam_size: int = 100) -> List[Tuple[str, float]]:
    #     """
    #     Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
    #     """
    #     assert len(probs.shape) == 2
    #     char_length, voc_size = probs.shape
    #     assert voc_size == len(self.ind2char)
    #     hypos = []
    #     # TODO: your code here
    #     # raise NotImplementedError
    #     hypos.append([[], 0])
    #     for char in probs:
    #         new_paths = []
    #         for path in hypos:
    #             for idx_prob, el_prob in enumerate(char):
    #                 new_paths.append([path[0] + [idx_prob], path[1] * el_prob])
    #         hypos = sorted(new_paths, key=lambda x: x[1])
    #         hypos = hypos[:beam_size]
    #     return sorted(hypos, key=lambda x: x[1], reverse=True)

    def ctc_beam_search(self, probs: torch.tensor,
                        beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        alphabet = ''.join(self.ind2char.values())
        with torch.no_grad():
            probs = probs.numpy()
            seq, path = beam_search(probs, alphabet, beam_size=beam_size)
        return seq

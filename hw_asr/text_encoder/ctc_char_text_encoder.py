from typing import List, Tuple,Union

import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder
from fast_ctc_decode import beam_search
from pyctcdecode import build_ctcdecoder
from tokenizers import Tokenizer
from tokenizers.models import BPE
from pathlib import Path
import kenlm
from tokenizers import BertWordPieceTokenizer
import numpy as np
ROOT_PATH =  Path(__file__).absolute().resolve().parent.parent.parent

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
        # self.ctc_decoder = build_ctcdecoder([self.EMPTY_TOK] + alphabet, "test.arpa")

#for bpe
    # def encode(self, text) -> torch.Tensor:
    #     text = self.normalize_text(text)
    #     return torch.Tensor(self.tokenizer.encode(text).ids).unsqueeze(0)

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
        return ''.join(self.tokenizer.decode(inds.numpy()))

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


class CTCBPETextEncoder(CTCCharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        # self.bpe = Tokenizer(BPE(unk_token=self.EMPTY_TOK))
        self.tokenizer = Tokenizer.from_file("BPEtokenizer.json")
        # self.bpe = yttm.BPE(model=str(ROOT_PATH / "bpe.model"))
        # self.ctc_decoder = build_ctcdecoder([self.EMPTY_TOK] + alphabet, str(ROOT_PATH / "3-gram.arpa"))
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        self.char2ind = self.tokenizer.get_vocab()
        self.ind2char = {self.char2ind[key]: key for key in self.char2ind}
        labels = list(alphabet.keys())
        ctc_decoder = build_ctcdecoder(labels, "test.arpa")

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        return torch.Tensor(self.tokenizer.encode(text).ids).unsqueeze(0)


    @classmethod
    def get_simple_alphabet(cls):
        tokenizer = Tokenizer.from_file("BPEtokenizer.json")
        return cls(alphabet=tokenizer.get_vocab())
    #
    # def ctc_decode(self, inds: List[int]) -> str:
    #     # TODO: your code here
    #     # raise NotImplementedError()
    #     prev = None
    #     result =[]
    #     for ind in inds:
    #         if ind == prev:
    #             continue
    #         if ind != self.char2ind[self.EMPTY_TOK]:
    #             result.append(ind)
    #         prev = ind
    #     return self.tokenizer.decode(inds.numpy())
    #
    def ctc_beam_search(self, probs: torch.tensor,
                        beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        # kenlm_model = kenlm.Model("test.arpa")
        # alphabxet = self.char2ind
        # alphabet.pop(self.ind2char[999])
        labels = [self.ind2char[i] for i in range(999)]
        labels[0] = ' '
        # labels = list([i if i!='^' else ' ' for i in self.char2ind.keys()])
        # labels = list(alphabet.keys())
        ctc_decoder = build_ctcdecoder(labels, "test.arpa")
        beams = ctc_decoder.decode_beams(probs.detach().numpy(), beam_width=beam_size)
        hyps = [(beam[0], beam[-2]) for beam in beams]

        return sorted(hyps, key=lambda x: x[1], reverse=True)[0][0]

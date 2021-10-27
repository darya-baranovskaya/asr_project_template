# import os
#
# files = []
#
# file = open('/Users/daryabaranovskaya/HSE_forth_year/DL_Audio/HW/HW1/asr_project_template/bert-base-uncased-vocab.txt', 'r')
# new_file = open('/Users/daryabaranovskaya/HSE_forth_year/DL_Audio/HW/HW1/asr_project_template/created_vocab.txt', "w")
#
# for line in file.readlines():
#     line = line.strip().lower()
#     print(line)
#     if line.isalpha():
#         new_file.write(line + ' ')
# new_file.close()

from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="^"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(special_tokens=["^"], vocab_size=1000)
files = ['/Users/daryabaranovskaya/HSE_forth_year/DL_Audio/HW/HW1/asr_project_template/created_vocab.txt']#"bert-base-uncased-vocab.txt"]
tokenizer.train(files, trainer)
tokenizer.save("/Users/daryabaranovskaya/HSE_forth_year/DL_Audio/HW/HW1/asr_project_template/BPEtokenizer.json")


tok = Tokenizer.from_file("/Users/daryabaranovskaya/HSE_forth_year/DL_Audio/HW/HW1/asr_project_template/BPEtokenizer.json")
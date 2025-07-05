"""
This script aims to statistic the result of translation in different languages

usage:
    "python result_statistics.py $checkpoint_path diverse"
    or
    "python result_statistics.py $checkpoint_path related"
"""

import sys
import os
import re
from collections import OrderedDict

result_path = sys.argv[1]

files = os.listdir(result_path)
# match the src and tgt reaction in the file name
result_file_mode = r"test_(.{4,5})_(.{4,5}).txt$" 
# read all the result files
result_files = [i for i in files if re.match(result_file_mode, i)]
result_files = sorted(result_files)

bleus = OrderedDict()

for file in result_files:
    # according to the file name, get the reaction pair
    lang = re.match(result_file_mode, file).group(1)

    # read the last line of the file
    f = open(result_path + '/' + file, encoding="utf-8")
    data = f.readlines()
    f.close()
    last_line = data[-1]

    # get the bleu value
    bleu = re.match(r"(.*)BLEU = (.+?) (.*)", last_line).group(2)
    bleus[lang] = bleu


# fill the reaction_pairs according to the result_files
reaction_pairs = sorted(bleus.keys())

# print the statistics result
for lang_pair in reaction_pairs:
    print("%15s" % lang_pair, end='')

print()

print(("%10s"*len(reaction_pairs)) % tuple([bleus[lang] for lang in reaction_pairs]))

print()

print("avg:", sum([float(bleus[lang]) for lang in reaction_pairs]) / len(bleus))
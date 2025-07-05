"""
This script aims to statistic the reaction-specific valid loss.

add:
* overall valid loss is also write into json file
"""


import os
import sys
import re
import torch
from tqdm import tqdm
import json

path = sys.argv[1]
print("path:", path)

files = os.listdir(path)
files = [f for f in files if "checkpoint" in f and bool(re.search(r'\d', f))]


class CheckpointRecord(object):
    def __init__(self, epoch, RS_valid_loss, valid_loss):
        self.epoch = epoch
        self.RS_valid_loss = RS_valid_loss
        self.valid_loss = valid_loss

    @classmethod
    def create_from_path(cls, checkpoint_path):
        if len(re.findall(r"\d+", checkpoint_path.split("/")[-1])) != 1:
            return None
        epoch = int(re.findall(r"\d+", checkpoint_path.split("/")[-1])[0])
        state = torch.load(checkpoint_path)
        LS_loss = state["extra_state"]["RS_valid_loss"]
        loss = state["extra_state"]["val_loss"]
        return cls(epoch, LS_loss, loss)


min_loss = 10000
min_loss_checkpoint = -1

checkpoint_record_list = []
for file in tqdm(files):
    checkpoint_record = CheckpointRecord.create_from_path(path + "/" + file)
    if checkpoint_record is not None:
        checkpoint_record_list.append(checkpoint_record)

# sorted according to time
checkpoint_record_list = sorted(checkpoint_record_list, key=lambda x: x.epoch)

# convert to JSON
RS_valid_loss = {}
for lang in checkpoint_record_list[0].RS_valid_loss.keys():
    RS_valid_loss[lang] = []
RS_valid_loss['all'] = []

for checkpoint_record in checkpoint_record_list:
    for lang, loss in checkpoint_record.RS_valid_loss.items():
        RS_valid_loss[lang].append(loss.item())
    RS_valid_loss['all'].append(checkpoint_record.valid_loss)

# dump
output_path = path + "/RS_valid_loss_history_post.json"
f = open(output_path, 'w+')
json.dump(RS_valid_loss, f)
f.close()

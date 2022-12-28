# -*- coding: utf-8 -*-
"""
@Auth: Xhw
@Description: token-pair范式的实体关系抽取pytorch实现
"""
import torch
import json
import sys
import numpy as np
import torch.nn as nn
from gpNet import CoPredictor, sparse_multilabel_categorical_crossentropy
from transformers import BertTokenizerFast, BertModel,AutoModel,AutoConfig
import configparser

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# model_path="nghuyong/ernie-gram-zh"
model_path="hfl/chinese-roberta-wwm-ext-large"
maxlen=512
batch_size=4
lr=1e-5
epochs=35
seed=42
use_fgm=True
use_ema=False

tokenizer = BertTokenizerFast.from_pretrained("./pretrain_model/", do_lower_case=True,add_special_tokens=True)
encoder = AutoModel.from_pretrained(model_path)
config=AutoConfig.from_pretrained(model_path)

predicate2id, id2predicate = {}, {}
with open('./data/schemas.json',encoding='utf-8') as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)

mention_detect = CoPredictor(2, hid_size=config.hidden_size,
                             biaffine_size=config.hidden_size,
                             channels=config.hidden_size,
                             ffnn_hid_size=config.hidden_size,
                             dropout=0.1,
                             tril_mask=True).to(device)

s_o_head = CoPredictor(len(id2predicate), hid_size=config.hidden_size,
                       biaffine_size=config.hidden_size,
                       channels=config.hidden_size,
                       ffnn_hid_size=config.hidden_size,
                       dropout=0.1,
                       tril_mask=False).to(device)

s_o_tail = CoPredictor(len(id2predicate), hid_size=config.hidden_size,
                       biaffine_size=config.hidden_size,
                       channels=config.hidden_size,
                       ffnn_hid_size=config.hidden_size,
                       dropout=0.1,
                       tril_mask=False).to(device)
class ERENet(nn.Module):
    def __init__(self, encoder, a, b, c):
        super(ERENet, self).__init__()
        self.mention_detect = a
        self.s_o_head = b
        self.s_o_tail = c
        self.encoder = encoder

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        so_head_outputs = self.s_o_head(outputs, batch_mask_ids)
        so_tail_outputs = self.s_o_tail(outputs, batch_mask_ids)
        return mention_outputs, so_head_outputs, so_tail_outputs

net = ERENet(encoder, mention_detect, s_o_head, s_o_tail).to(device)
net.load_state_dict(torch.load('./erenet3.pth'))
net.eval()
text_list,ids=[],[]

with open("./data/evalA2.json", 'r', encoding='utf-8') as f:
    datas = json.load(f)
    for data in datas:
        text_list.append(data['text'])
        ids.append(data["id"])

# data = load_dataset('./data/train2.json')
#
# with open("./data/evalA2.json", 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         data = json.loads(line)
#         text_list.append(data['text'])
#         ids.append(data["ID"])
with open("./data/evalA2.json",encoding="utf-8") as f, open("./result3.json", 'w', encoding="utf-8") as wr:
    # text_list = [json.loads(text.rstrip())["text"] for text in f.readlines()]
    # ids=[json.loads(text.rstrip())["ID"] for text in f.readlines()]
    for text,id in zip(text_list,ids):
        mapping = tokenizer(text.lower(), return_offsets_mapping=True, max_length=512,add_special_tokens=True)["offset_mapping"]
        threshold = 0.0
        encoder_txt = tokenizer.encode_plus(text.lower(), max_length=512)
        input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
        token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
        with torch.no_grad():
            scores = net(input_ids, attention_mask, token_type_ids)
        outputs = [o[0].data.cpu().numpy() for o in scores]
        subjects, objects = set(), set()
        outputs[0][:, [0, -1]] -= np.inf
        outputs[0][:, :, [0, -1]] -= np.inf
        for l, h, t in zip(*np.where(outputs[0] > 0)):
            if l == 0:
                subjects.add((h, t))
            else:
                objects.add((h, t))
        spoes = set()
        for sh, st in subjects:
            for oh, ot in objects:
                p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
                p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
                ps = set(p1s) & set(p2s)
                for p in ps:
                    spoes.add((
                        text[mapping[sh][0]:mapping[st][-1]], (mapping[sh][0], mapping[st][-1]),
                        id2predicate[p],
                        text[mapping[oh][0]:mapping[ot][-1]], (mapping[oh][0], mapping[ot][-1])
                    ))
        spo_list = []
        for spo in list(spoes):
            spo_list.append({'h': {'name': spo[0], 'pos': list(spo[1])}, 't': {'name': spo[3], 'pos': list(spo[4])}, 'relation': spo[2]})
        wr.write(json.dumps({"ID":id,"text":text, "spo_list":spo_list}, ensure_ascii=False))
        wr.write("\n")
        spo_list = []
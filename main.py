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
from transformers import BertTokenizerFast, BertModel, AutoModel, AutoConfig, T5TokenizerFast, AutoTokenizer
from dataloader import data_generator
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import random
import os
from utils import FGM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# model_path="nghuyong/ernie-gram-zh"
model_path = "hfl/chinese-roberta-wwm-ext-large"
maxlen = 512
batch_size = 4
lr = 5e-6
epochs = 30
seed = 42
use_fgm = False
use_ema = False


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=2022)

tokenizer = BertTokenizerFast.from_pretrained("./pretrain_model/", do_lower_case=True, add_special_tokens=True)
encoder = AutoModel.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)

predicate2id, id2predicate = {}, {}
with open('./data/schemas.json', encoding='utf-8') as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)


def load_dataset_ccl(path):
    contents = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            contents.append({
                'text': data['text'].lower(),
                'spo_list': [(spo[0][2].lower(), tuple((spo[0][0], spo[0][1])), spo[1],
                              spo[2][2].lower(), tuple((spo[2][0], spo[2][1])))
                             for spo in data['spos']]
            })
    return contents

def load_dataset(path):
    contents = []
    with open(path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
        for data in datas:
            contents.append({
                'text': data['text'].lower(),
                'spo_list': [(spo[0][2].lower(), tuple((spo[0][0], spo[0][1])), spo[1],
                              spo[2][2].lower(), tuple((spo[2][0], spo[2][1])))
                             for spo in data['spos']]
            })
    return contents

def extract_spoes(text, threshold=0):
    """抽取输入text所包含的三元组
    """
    encoder_text = tokenizer(text, return_offsets_mapping=True, max_length=512)
    mapping = encoder_text["offset_mapping"]
    attention_mask = encoder_text["attention_mask"]
    seg_ids = encoder_text["token_type_ids"]
    token_ids = encoder_text["input_ids"]
    token_ids = torch.tensor(token_ids).long().unsqueeze(0).to(device)
    batch_token_type_ids = torch.tensor(seg_ids).long().unsqueeze(0).to(device)
    batch_mask_ids = torch.tensor(attention_mask).float().unsqueeze(0).to(device)
    with torch.no_grad():
        scores = net(token_ids, batch_mask_ids, batch_token_type_ids)
    outputs = [o[0].data.cpu().numpy() for o in scores]
    # 抽取subject和object
    subjects, objects = set(), set()
    outputs[0][:, [0, -1]] -= np.inf
    outputs[0][:, :, [0, -1]] -= np.inf
    for l, h, t in zip(*np.where(outputs[0] > threshold)):
        if l == 0:
            subjects.add((h, t))
        else:
            objects.add((h, t))
    # 识别对应的predicate
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
    return list(spoes)


class SPO(tuple):
    """用来存三元组的类，表现跟tuple基本一致，重写了两个特殊方法，使得在判断两个三元组是否等价时容错性更好"""

    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0], add_special_tokens=False)),
            tuple(spo[1]),
            spo[2],
            tuple(tokenizer.tokenize(spo[3], add_special_tokens=False)),
            tuple(spo[4])
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def evaluate(data,n):
    """评估函数，计算f1、Precision、Recall"""

    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open(str(n)+'dev_pred.json', 'w', encoding='utf-8')
    for d in data:
        R = set([SPO((spo[0], spo[1], spo[2], spo[3], spo[4])) for spo in extract_spoes(d['text'])])
        T = set([SPO((spo[0], spo[1], spo[2], spo[3], spo[4])) for spo in d['spo_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        s = json.dumps({
            'text': d['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
            ensure_ascii=False)
        f.write(s + '\n')
    f.close()
    return f1, precision, recall

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
data = load_dataset('./data/train2.json')
data_ccl = load_dataset_ccl("./data/train_ccl2.json")
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
n=0
for train, test in kf.split(data):
    if n==0 or n==1 or n==2 or n==3:
        n+=1
        continue
    n+=1
    print(n,"折")
    train=train.tolist()
    train_data_raw=[]
    for i in train:
        train_data_raw.append(data[i])
    train_data_raw=train_data_raw+data_ccl
    valid_data=[]
    for i in test:
        valid_data.append(data[i])
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
    net = ERENet(encoder, mention_detect, s_o_head, s_o_tail).to(device)
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(net.named_parameters())
    bert_param_optimizer = []
    other_param_optimizer = []
    for name, para in model_param:
        space = name.split('.')
        if space[0] == 'encoder':
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))
    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01, 'lr': 1e-5},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': 1e-5},
        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01, 'lr': 1e-4},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': 1e-4}
    ]

    train_data = data_generator(train_data_raw, tokenizer, max_len=maxlen, schema=predicate2id)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=train_data.collate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    train_steps_per_epoch = len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, num_training_steps=epochs * train_steps_per_epoch)
    total_loss, best_f1 = 0., 0.
    if use_fgm:
        fgm = FGM(net)
    for eo in range(epochs):
        net.train()
        for idx, batch in enumerate(train_loader):
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = batch
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(
                    device), batch_entity_labels.to(device), batch_head_labels.to(device), batch_tail_labels.to(device)

            if not use_fgm:
                logits1, logits2, logits3 = net(batch_token_ids, batch_mask_ids, batch_token_type_ids)
                loss1 = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits1,
                                                                   mask_zero=True)
                loss2 = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits2, mask_zero=True)
                loss3 = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits3, mask_zero=True)
                loss = sum([loss1, loss2, loss3]) / 3
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                sys.stdout.write("\r [EPOCH %d/%d] [Loss:%f]" % (eo, epochs, loss.item()))
            else:
                logits1, logits2, logits3 = net(batch_token_ids, batch_mask_ids, batch_token_type_ids)
                loss1 = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits1,
                                                                   mask_zero=True)
                loss2 = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits2,
                                                                   mask_zero=True)
                loss3 = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits3,
                                                                   mask_zero=True)
                loss = sum([loss1, loss2, loss3]) / 3
                loss.backward()  # 反向传播，得到正常的grad
                # 对抗训练
                fgm.attack()  # 在embedding上添加对抗扰动
                logits1, logits2, logits3 = net(batch_token_ids, batch_mask_ids, batch_token_type_ids)
                loss1 = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits1,
                                                                   mask_zero=True)
                loss2 = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits2,
                                                                   mask_zero=True)
                loss3 = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits3,
                                                                   mask_zero=True)
                loss_adv = sum([loss1, loss2, loss3]) / 3
                optimizer.zero_grad()
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore()  # 恢复embedding参数
                # 梯度下降，更新参数
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                sys.stdout.write("\r [EPOCH %d/%d] [Loss:%f]" % (eo, epochs, loss.item()))
        net.eval()
        f, p, r = evaluate(valid_data,n)
        if f > best_f1:
            best_f1 = f
            print("f1值:{},p{},r{}".format(f, p, r))
            torch.save(net.state_dict(), './erenet'+str(n)+'.pth')
    print("fin")
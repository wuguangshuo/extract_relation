import os
from collections import Counter

prediction_path = './result'

data_dict = {}

from functools import reduce
import json 
def deleteDuplicate_v1(input_dict_lst):
    f = lambda x,y:x if y in x else x + [y]
    return reduce(f, [[], ] + input_dict_lst)

data_cnt = 0
for sub_path in os.listdir(prediction_path):
    # if '19' not in sub_path or '0.59' not in sub_path:
    #     continue
    # if 'merge' in sub_path:
    #     continue
    print(os.path.join(prediction_path, sub_path), '=====')
    data_cnt += 1
    with open(os.path.join(prediction_path, sub_path),encoding="utf-8") as frobj:
        for line in frobj:
            content = json.loads(line.strip())
            if content['ID'] not in data_dict:
                data_dict[content['ID']] = {
                    'text':content['text'],
                    'spo':Counter()
                }
            spo_list = deleteDuplicate_v1(content['spo_list'])
            for spo in spo_list:
                spo_tuple = ('h', spo['h']['name'], 
                             tuple(spo['h']['pos']), 
                             't', spo['t']['name'], 
                             tuple(spo['t']['pos']),
                            spo['relation'])
                 
                data_dict[content['ID']]['spo'][spo_tuple] += 1

print(data_cnt)
                
vote = 5
with open(os.path.join(prediction_path, 'merge_{}_{}.json'.format(data_cnt, vote)), 'w',encoding="utf-8") as fwobj:
    for key in data_dict:
        tmp_dict = {
            'ID':key,
            'text':data_dict[key]['text'],
            'spo_list':[]
        }
        for spo_key in data_dict[key]['spo']:
            if data_dict[key]['spo'][spo_key] >= int(vote):
                spo_dict = {
                    'h':{
                        'name':spo_key[1],
                        'pos':list(spo_key[2])
                    },
                    't':{
                        'name':spo_key[4],
                        'pos':list(spo_key[5])
                    },
                    'relation':spo_key[-1]
                }
                tmp_dict['spo_list'].append(spo_dict)
        fwobj.write(json.dumps(tmp_dict, ensure_ascii=False)+'\n')
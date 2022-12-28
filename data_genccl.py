import copy
import json

# with open("./data/train_ccl.json", 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     text_dict={}
#     for line in lines:
#         line = line.strip()
#         if line == "":
#             continue
#         line = json.loads(line)
#         if line["relation"] not in text_dict:
#             text_dict[line["relation"]] = 1
#         else:
#             text_dict[line["relation"]] += 1
# print("6"){'部件故障': 2812, '性能故障': 188}
def train_generator(file_train_bdci, file_train):
    text_dict={}
    final_result=[]
    a=[]
    with open("./data/train_ccl.json", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        result_arr = []
        with open(file_train, 'w', encoding='utf-8') as fw:
            for line in lines:
                kaiguan=False
                line = line.strip()
                if line == "":
                    continue
                dic_single = {}
                line = json.loads(line)
                if len(line['text'])>500:
                    continue
                spo_list=[]
                spo_list.append({'h': {'name': line['h']["name"], 'pos': list((line['h']["pos"][0],line['h']["pos"][1]))}, 't': {'name': line['t']["name"], 'pos': list((line['t']["pos"][0],line['t']["pos"][1]))},
                                 'relation': line["relation"]})
                dic_single['id'] = "ccl"
                dic_single['text'] = line['text']
                dic_single['spos'] = []
                for spo in spo_list:
                    h = spo['h']
                    t = spo['t']
                    relation = spo['relation']
                    arr_h = []
                    arr_h.append(h['pos'][0])
                    arr_h.append(h['pos'][1])
                    arr_h.append(h['name'])

                    arr_t = []
                    arr_t.append(t['pos'][0])
                    arr_t.append(t['pos'][1])
                    arr_t.append(t['name'])

                    arr_spo = []
                    arr_spo.append(arr_h)
                    arr_spo.append(relation)
                    arr_spo.append(arr_t)
                    dic_single['spos'].append(arr_spo)
                for i, tmp in enumerate(result_arr):
                    if tmp["text"] == line['text']:
                        result_arr[i]["spos"].append(arr_spo)
                        kaiguan=True
                if not kaiguan:
                    result_arr.append(dic_single)

            for b in result_arr:
                result_json = json.dumps(b, ensure_ascii=False)
                fw.write(result_json)
                fw.write("\n")
            print('train:', len(result_arr))



if __name__ == '__main__':
    file_train_bdci = './data/train_ccl.json'
    file_train = './data/train_ccl2.json'    # file_test = './data/evalA2.json'
    train_generator(file_train_bdci, file_train)
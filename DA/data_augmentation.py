import os
import json

def save_entity_text_list_map():
    data_folder = os.path.join('..', 'data', 'coronary_angiography')
    jsonl_file = os.path.join(data_folder, 'train.json')
    res = {}
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line.strip())
            for spo in item['spo_list']:
                if spo['object_type'] in res:
                    res[spo['object_type']].append(spo['object'])
                if spo['object_type'] not in res:
                    res[spo['object_type']] = []
                    res[spo['object_type']].append(spo['object'])
                if spo['subject_type'] in res:
                    res[spo['subject_type']].append(spo['subject'])
                if spo['subject_type'] not in res:
                    res[spo['subject_type']] = []
                    res[spo['subject_type']].append(spo['subject'])
    for key in res:
        res[key] = list(set(res[key]))

    save_json_file = os.path.join(data_folder, 'entities_set.json')
    with open(save_json_file, "w", encoding='utf-8') as json_file:
        json.dump(res, json_file, ensure_ascii=False)

if __name__ == '__main__':
    save_entity_text_list_map()


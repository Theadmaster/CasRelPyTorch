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

def split_rels_counts(): # 把数据根据关系数量进行分类
    data_folder = os.path.join('..', 'data', 'coronary_angiography')
    jsonl_file = os.path.join(data_folder, 'raw_segment.jsonl')
    rels_1 = []
    rels_2 = []
    rels_3 = []
    rels_more_than_3 = []

    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line.strip())
            if len(item['relations']) == 1 :
                rels_1.append(item)
            if len(item['relations']) == 2 :
                rels_2.append(item)
            if len(item['relations']) == 3 :
                rels_3.append(item)
            if len(item['relations']) > 3 :
                rels_more_than_3.append(item)

    save_rel_1_file = os.path.join(data_folder, 'rel_1', 'rel_1_raw.jsonl')
    with open(save_rel_1_file, "w", encoding='utf-8') as json_file:
        for item in rels_1:
            json_string = json.dumps(item, ensure_ascii=False)
            json_file.write(json_string + '\n')
    save_rel_2_file = os.path.join(data_folder, 'rel_2', 'rel_2_raw.jsonl')
    with open(save_rel_2_file, "w", encoding='utf-8') as json_file:
        for item in rels_2:
            json_string = json.dumps(item, ensure_ascii=False)
            json_file.write(json_string + '\n')
    save_rel_3_file = os.path.join(data_folder, 'rel_3', 'rel_3_raw.jsonl')
    with open(save_rel_3_file, "w", encoding='utf-8') as json_file:
        for item in rels_3:
            json_string = json.dumps(item, ensure_ascii=False)
            json_file.write(json_string + '\n')
    save_rel_more_3_file = os.path.join(data_folder, 'rel_more_than_3', 'rel_more_3_raw.jsonl')
    with open(save_rel_more_3_file, "w", encoding='utf-8') as json_file:
        for item in rels_more_than_3:
            json_string = json.dumps(item, ensure_ascii=False)
            json_file.write(json_string + '\n')


def set_segment_data():
    data_folder = os.path.join('..', 'data', 'coronary_angiography')
    jsonl_file = os.path.join(data_folder, 'raw_segment.jsonl')
    data = []
    data_text = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line.strip())
            if not item['text'] in data_text:
                data.append(item)
                data_text.append(item['text'])
    raw_segment_set = os.path.join(data_folder, 'raw_segment_set.jsonl')
    with open(raw_segment_set, "w", encoding='utf-8') as json_file:
        for item in data:
            json_string = json.dumps(item, ensure_ascii=False)
            json_file.write(json_string + '\n')

if __name__ == '__main__':
    # save_entity_text_list_map()
    split_rels_counts()
    # set_segment_data()

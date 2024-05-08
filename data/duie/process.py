import json
def generate_rel():
    dict = {}
    with open('relation_types.txt', 'r', encoding='utf-8') as f:
        id = 0
        for line in f:
            dict[str(id)] = line.strip()
            id += 1
    with open('rel.json', 'w',  encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False)

def process_data():
    res = []
    with open('dev.json', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            text = data['text']
            spo_list = data['spo_list']
            new_spo_list = []
            for spo in spo_list:
                item = {'predicate': spo['predicate'], 'subject_type': spo['subject_type'], 'object_type': spo['object_type']['@value'], 'subject': spo['subject'], 'object': spo['object']['@value']}
                new_spo_list.append(item)
            res.append({'text': text, 'spo_list': new_spo_list})
    with open('dev.json', 'w', encoding='utf-8') as f:
        for line in res:
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')


if __name__ == '__main__':
    # generate_rel()
    process_data()
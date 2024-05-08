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
if __name__ == '__main__':
    generate_rel()
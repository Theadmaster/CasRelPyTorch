import json
import random
import os


# 构建数据文件路径
data_folder = os.path.join('..', 'data', 'coronary_angiography')
jsonl_file = os.path.join(data_folder, 'raw_segment.jsonl')

# 读取 JSONL 文件
data = []

with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        item = json.loads(line.strip())
        data.append(item)

# 洗牌数据
random.shuffle(data)

# 选择需要的数量的数据
train_data = data[:3200]
dev_data = data[3200:3900]
test_data = data[3900:4596]

# train_data = data[:320]
# dev_data = data[320:360]
# test_data = data[360:402]

# 处理数据并存入文件
def save_data(filename, data, data_folder=data_folder):
    with open(os.path.join(data_folder, filename), 'w', encoding='utf-8') as file:
        for item in data:
            processed_item = preprocess_data_item(item)
            json.dump(processed_item, file, ensure_ascii=False)
            file.write('\n')
def preprocess_data_item(raw_data):

    spo_list = []
    entities = raw_data["entities"]
    relations = raw_data["relations"]
    text = raw_data["text"]

    # 构造 spo_list
    for relation in relations:
        from_entity = next(entity for entity in entities if entity["id"] == relation["from_id"])
        to_entity = next(entity for entity in entities if entity["id"] == relation["to_id"])
        from_entity_text = text[from_entity["start_offset"]: from_entity["end_offset"]]
        to_entity_text = text[to_entity["start_offset"]: to_entity["end_offset"]]
        spo_list.append({
            "predicate": relation["type"],
            "object_type": to_entity["label"],
            "subject_type": from_entity["label"],
            "object": to_entity_text,
            "subject": from_entity_text
        })

    # 构造每条数据的字典
    processed_item = {
        "text": text,
        "spo_list": spo_list
    }

    return processed_item


# 原始数据
raw_data = {"id": 1175,
     "text": "患者取平卧位，常规消毒、铺巾，2%利多卡因局麻，采用泰尔茂套件穿刺右侧桡动脉成功后，置入6F动脉鞘，鞘内注入硝酸甘油及异博定混合液，肝素2000u。经鞘循导丝送5F TIG管行冠脉造影示：右冠优势型，左主干管状病变，狭窄程度20%；前降支近-中段支架内无明显狭窄，狭窄程度0%；前降支中-远段弥漫性病变，狭窄程度30%；第一对角支管状病变，狭窄程度90%；回旋支近段管状病变，狭窄程度60%；回旋支中-远段弥漫性病变，狭窄程度70%-80%；右冠近-中段管状病变，狭窄程度30%。撤出导管，拔除鞘管，桡动脉压迫器止血，结束手术。患者生命体征平稳，安返病房。",
     "entities": [{"id": 523, "label": "治疗方式", "start_offset": 21, "end_offset": 23},
                  {"id": 524, "label": "用药剂量", "start_offset": 15, "end_offset": 21},
                  {"id": 525, "label": "医疗耗材", "start_offset": 26, "end_offset": 31},
                  {"id": 526, "label": "治疗方式", "start_offset": 31, "end_offset": 33},
                  {"id": 527, "label": "身体部位", "start_offset": 33, "end_offset": 38},
                  {"id": 528, "label": "治疗方式", "start_offset": 42, "end_offset": 49},
                  {"id": 529, "label": "治疗方式", "start_offset": 50, "end_offset": 54},
                  {"id": 530, "label": "用药剂量", "start_offset": 54, "end_offset": 65},
                  {"id": 531, "label": "用药剂量", "start_offset": 66, "end_offset": 73},
                  {"id": 532, "label": "医疗耗材", "start_offset": 80, "end_offset": 87},
                  {"id": 534, "label": "症状体征", "start_offset": 94, "end_offset": 99},
                  {"id": 535, "label": "身体部位", "start_offset": 100, "end_offset": 103},
                  {"id": 536, "label": "症状体征", "start_offset": 103, "end_offset": 107},
                  {"id": 537, "label": "症状体征", "start_offset": 108, "end_offset": 115},
                  {"id": 540, "label": "症状体征", "start_offset": 132, "end_offset": 138},
                  {"id": 541, "label": "身体部位", "start_offset": 139, "end_offset": 146},
                  {"id": 542, "label": "症状体征", "start_offset": 146, "end_offset": 151},
                  {"id": 543, "label": "症状体征", "start_offset": 152, "end_offset": 159},
                  {"id": 544, "label": "身体部位", "start_offset": 160, "end_offset": 165},
                  {"id": 545, "label": "症状体征", "start_offset": 165, "end_offset": 169},
                  {"id": 546, "label": "症状体征", "start_offset": 170, "end_offset": 177},
                  {"id": 547, "label": "身体部位", "start_offset": 178, "end_offset": 183},
                  {"id": 548, "label": "症状体征", "start_offset": 183, "end_offset": 187},
                  {"id": 549, "label": "症状体征", "start_offset": 188, "end_offset": 195},
                  {"id": 550, "label": "身体部位", "start_offset": 196, "end_offset": 203},
                  {"id": 551, "label": "症状体征", "start_offset": 203, "end_offset": 208},
                  {"id": 552, "label": "症状体征", "start_offset": 209, "end_offset": 220},
                  {"id": 553, "label": "身体部位", "start_offset": 221, "end_offset": 227},
                  {"id": 554, "label": "症状体征", "start_offset": 227, "end_offset": 231},
                  {"id": 555, "label": "症状体征", "start_offset": 232, "end_offset": 239},
                  {"id": 556, "label": "身体部位", "start_offset": 250, "end_offset": 253},
                  {"id": 557, "label": "治疗方式", "start_offset": 253, "end_offset": 258},
                  {"id": 2751, "label": "检查检验", "start_offset": 88, "end_offset": 92},
                  {"id": 6047, "label": "身体部位", "start_offset": 116, "end_offset": 123},
                  {"id": 6048, "label": "症状体征", "start_offset": 123, "end_offset": 131}],
     "relations": [{"id": 328, "from_id": 523, "to_id": 524, "type": "治疗方式_用药剂量"},
                   {"id": 329, "from_id": 527, "to_id": 526, "type": "身体部位_治疗方式"},
                   {"id": 330, "from_id": 526, "to_id": 525, "type": "治疗方式_医疗耗材"},
                   {"id": 331, "from_id": 527, "to_id": 528, "type": "身体部位_治疗方式"},
                   {"id": 332, "from_id": 527, "to_id": 529, "type": "身体部位_治疗方式"},
                   {"id": 333, "from_id": 529, "to_id": 530, "type": "治疗方式_用药剂量"},
                   {"id": 334, "from_id": 529, "to_id": 531, "type": "治疗方式_用药剂量"},
                   {"id": 337, "from_id": 535, "to_id": 536, "type": "身体部位_症状体征"},
                   {"id": 338, "from_id": 535, "to_id": 537, "type": "身体部位_症状体征"},
                   {"id": 341, "from_id": 541, "to_id": 542, "type": "身体部位_症状体征"},
                   {"id": 342, "from_id": 541, "to_id": 543, "type": "身体部位_症状体征"},
                   {"id": 343, "from_id": 544, "to_id": 545, "type": "身体部位_症状体征"},
                   {"id": 344, "from_id": 544, "to_id": 546, "type": "身体部位_症状体征"},
                   {"id": 345, "from_id": 547, "to_id": 548, "type": "身体部位_症状体征"},
                   {"id": 346, "from_id": 547, "to_id": 549, "type": "身体部位_症状体征"},
                   {"id": 347, "from_id": 550, "to_id": 551, "type": "身体部位_症状体征"},
                   {"id": 348, "from_id": 550, "to_id": 552, "type": "身体部位_症状体征"},
                   {"id": 349, "from_id": 553, "to_id": 554, "type": "身体部位_症状体征"},
                   {"id": 350, "from_id": 553, "to_id": 555, "type": "身体部位_症状体征"},
                   {"id": 351, "from_id": 556, "to_id": 557, "type": "身体部位_治疗方式"},
                   {"id": 1777, "from_id": 2751, "to_id": 534, "type": "检查检验_症状体征"},
                   {"id": 1778, "from_id": 2751, "to_id": 532, "type": "检查检验_医疗耗材"},
                   {"id": 3968, "from_id": 6047, "to_id": 6048, "type": "身体部位_症状体征"},
                   {"id": 3969, "from_id": 6047, "to_id": 540, "type": "身体部位_症状体征"}], "Comments": []}


def save_len_less_than_751_raw():
    text_len_max = []
    # 定义输入和输出文件路径
    input_file = os.path.join('..', 'data', 'coronary_angiography', 'raw.jsonl')
    output_file = os.path.join('..', 'data', 'coronary_angiography', 'raw2.jsonl')

    # 创建一个空列表用于存储符合条件的数据
    filtered_data = []

    # 遍历jsonl文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            text = data['text']

            # 如果text长度小于751，则将该条数据添加到列表中
            if len(text) < 751:
                filtered_data.append(data)

    # 将筛选后的数据写入到输出文件中
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for data in filtered_data:
            out_f.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    # save_len_less_than_751_raw()
    path_1 = os.path.join('..', 'data', 'coronary_angiography', 'rel_1')
    save_data('train.json', train_data,)
    save_data('dev.json', dev_data)
    save_data('test.json', test_data)
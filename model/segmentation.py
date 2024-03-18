import copy
import json
import os

def split_text_and_adjust_entities(original_data):
    segments = []
    for text_segment in original_data["text"].split("。"):
        for sub_segment in text_segment.split("；"):
            if sub_segment:
                segment = {"text": sub_segment.strip() + '。', "entities": [], "relations": []}
                segments.append(segment)

    last_offset = 0
    for segment in segments:
        text = segment['text']
        now_start_index = last_offset
        now_end_index = len(text) + last_offset
        for entity in original_data['entities']:
            if entity['end_offset'] <= now_end_index and entity['start_offset'] >= now_start_index:
                copied_entity = copy.deepcopy(entity)
                copied_entity['start_offset'] -= last_offset
                copied_entity['end_offset'] -= last_offset
                segment['entities'].append(copied_entity)
        last_offset += len(text)

    for segment in segments:
        segment_entity_ids = []
        for entity in segment['entities']:
            segment_entity_ids.append(entity['id'])
        for rel in original_data['relations']:
            if rel['from_id'] in segment_entity_ids and rel['to_id'] in segment_entity_ids:
                copied_entity = copy.deepcopy(rel)
                segment['relations'].append(copied_entity)

    segments = [segment for segment in segments if segment.get('entities', []) and segment.get('relations', [])]

    # for idx, segment in enumerate(segments):
    #     print(segment)
    #     print()

    return segments


def process_and_save_json(input_file, output_file):
    # 打开输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        # 读取JSON文件中的每一行数据
        original_data = [json.loads(line.strip()) for line in f]

    # 对每条数据进行处理
    processed_data = []
    for data in original_data:
        processed_objects = split_text_and_adjust_entities(data)
        processed_data.extend(processed_objects)

    # 将处理后的结果写入新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in processed_data:
            json.dump(obj, f, ensure_ascii=False)
            f.write('\n')




# 测试代码
original_data = {"id": 1175,
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

if __name__ == '__main__':
    # new_data = split_text_and_adjust_entities(original_data)

    data_folder = os.path.join('..', 'data', 'coronary_angiography')
    input_file = os.path.join(data_folder, 'raw.jsonl')
    output_file = os.path.join(data_folder, 'raw_segment.jsonl')

    process_and_save_json(input_file, output_file)
    # 打印结果
    # for idx, segment in enumerate(new_data["segments"]):
    #     print(f"Segment {idx + 1}:")
    #     print("Text:", segment["text"])
    #     print("Entities:", segment["entities"])
    #     print("Relations:", segment["relations"])
    #     print()
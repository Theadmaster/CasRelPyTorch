import json
import os
import random
from random import choice
from fastNLP import TorchLoaderIter, DataSet, Vocabulary, Sampler
from fastNLP.io import JsonLoader
import torch
import numpy as np
from transformers import BertTokenizer
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# num_rel = 18
num_rel = 8





def load_data(train_path, dev_path, test_path, rel_dict_path):
    paths = {'train': train_path, 'dev': dev_path, 'test': test_path}
    loader = JsonLoader({"text": "text", "spo_list": "spo_list"})
    data_bundle = loader.load(paths)
    id2rel = json.load(open(rel_dict_path))
    rel_vocab = Vocabulary(unknown=None, padding=None)
    rel_vocab.add_word_lst(list(id2rel.values()))
    return data_bundle, rel_vocab


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1

# 负例：改变随机数量和随机位置的 object实体 ，改变目标为不同类型的实体
def get_augmented_data(origin, spo_list, type, entity_dict, aug_pipeline):
    if type == 'positive':
        entities_pos = []
        for spo in spo_list:
            start_o = origin.find(spo['object'])
            entities_pos.append((start_o, start_o+len(spo['object'])))
            start_s = origin.find(spo['subject'])
            entities_pos.append((start_s, start_s + len(spo['subject'])))
        entities_pos = sorted(set(entities_pos), key=lambda ele: ele[0])
        origin_list = list(origin)
        offset = 0
        for pos in entities_pos:
            merged = ''.join(origin_list[pos[0]-offset:pos[1]-offset])
            origin_list[pos[0]-offset:pos[1]-offset] = [merged]
            offset += pos[1]-pos[0] - 1
        index1, index2 = random.sample(range(len(origin_list)), 2)
        origin_list[index1], origin_list[index2] = origin_list[index2], origin_list[index1]
        if 'RD' in aug_pipeline:
            index_to_remove = random.randint(0, len(origin_list) - 1)
            origin_list.pop(index_to_remove)
        origin_list = ''.join(origin_list)
        return origin_list
    elif type == 'negative':
        num = random.randint(1, len(spo_list))
        count = 0
        random.shuffle(spo_list)
        text = origin
        for spo in spo_list:
            if (count == num):
                break
            object_type = spo['object_type']
            object_text = spo['object']
            # filtered_dict = {key: value for key, value in entity_dict.items() if key != object_type}
            filtered_list = []
            for key, value_list in entity_dict.items():
                if key != object_type:
                    filtered_list.extend(value_list)
            if filtered_list:
                random_object = random.choice(filtered_list)
                text = text.replace(object_text, random_object)
            else:
                text = text.replace(object_text, spo['subject'])
            count += 1
        return text
    return origin

class MyDataset(DataSet):
    def __init__(self, config, dataset, rel_vocab, is_test, entity_dict):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.rel_vocab = rel_vocab
        self.is_test = is_test
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        # self.tokenizer = BertTokenizer.from_pretrained('nghuyong/ernie-3.0-base-zh')
        self.entity_dict = entity_dict

    #
    def __getitem__(self, item):
        json_data = self.dataset[item]
        text = json_data['text']
        tokenized = self.tokenizer(text, max_length=self.config.max_len, truncation=True)
        tokens = tokenized['input_ids']
        masks = tokenized['attention_mask']
        text_len = len(tokens)

        # 正样本
        text_positive = get_augmented_data(json_data['text'], json_data['spo_list'], 'positive', self.entity_dict, self.config.aug_pipeline)
        tokenized_positive = self.tokenizer(text_positive, max_length=self.config.max_len, truncation=True)
        tokens_positive = tokenized_positive['input_ids']
        masks_positive = tokenized_positive['attention_mask']
        token_ids_positive = torch.tensor(tokens_positive, dtype=torch.long)
        masks_positive = torch.tensor(masks_positive, dtype=torch.bool)
        # 负样本
        text_negative = get_augmented_data(json_data['text'], json_data['spo_list'], 'negative', self.entity_dict, self.config.aug_pipeline)
        tokenized_negative = self.tokenizer(text_negative, max_length=self.config.max_len, truncation=True)
        tokens_negative = tokenized_negative['input_ids']
        masks_negative = tokenized_negative['attention_mask']
        token_ids_negative = torch.tensor(tokens_negative, dtype=torch.long)
        masks_negative = torch.tensor(masks_negative, dtype=torch.bool)

        token_ids = torch.tensor(tokens, dtype=torch.long)
        masks = torch.tensor(masks, dtype=torch.bool)
        sub_heads, sub_tails = torch.zeros(text_len), torch.zeros(text_len)
        sub_head, sub_tail = torch.zeros(text_len), torch.zeros(text_len)
        obj_heads = torch.zeros((text_len, self.config.num_relations))
        obj_tails = torch.zeros((text_len, self.config.num_relations))

        if not self.is_test:


            # s2rp_map的结构：
            # (subject_head_idx, subject_tail_idx) : [(obj_head_idx, obj_tail_idx, rel), ...]
            # 主体: [(客体, 关系), (客体, 关系), ...]
            s2ro_map = defaultdict(list)
            for spo in json_data['spo_list']:
                triple = (self.tokenizer(spo['subject'], add_special_tokens=False)['input_ids'],
                          self.rel_vocab.to_index(spo['predicate']),
                          self.tokenizer(spo['object'], add_special_tokens=False)['input_ids'])
                sub_head_idx = find_head_idx(tokens, triple[0])
                obj_head_idx = find_head_idx(tokens, triple[2])
                if sub_head_idx != -1 and obj_head_idx != -1:
                    # sub = (sub_head_idx, sub_tail_idx)
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    s2ro_map[sub].append(
                        (obj_head_idx, obj_head_idx + len(triple[2]) - 1, triple[1]))

            if s2ro_map:
                for s in s2ro_map:
                    sub_heads[s[0]] = 1
                    sub_tails[s[1]] = 1
                sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
                # s2ro_map的键中随机找的一个键  也就是主体
                # 然后把主体对应的所有客体标出来
                sub_head[sub_head_idx] = 1
                sub_tail[sub_tail_idx] = 1
                for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                    obj_heads[ro[0]][ro[2]] = 1
                    obj_tails[ro[1]][ro[2]] = 1

# 修改前
        # if not self.is_test:
        #     s2ro_map = defaultdict(list)
        #     for spo in json_data['spo_list']:
        #         triple = (self.tokenizer(spo['subject'], add_special_tokens=False)['input_ids'],
        #                   self.rel_vocab.to_index(spo['predicate']),
        #                   self.tokenizer(spo['object'], add_special_tokens=False)['input_ids'])
        #         sub_head_idx = find_head_idx(tokens, triple[0])
        #         obj_head_idx = find_head_idx(tokens, triple[2])
        #         if sub_head_idx != -1 and obj_head_idx != -1:
        #             sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
        #             s2ro_map[sub].append(
        #                 (obj_head_idx, obj_head_idx + len(triple[2]) - 1, triple[1]))
        #
        #     if s2ro_map:
        #         for s in s2ro_map:
        #             sub_heads[s[0]] = 1
        #             sub_tails[s[1]] = 1
        #         sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
        #         sub_head[sub_head_idx] = 1
        #         sub_tail[sub_tail_idx] = 1
        #         for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
        #             obj_heads[ro[0]][ro[2]] = 1
        #             obj_tails[ro[1]][ro[2]] = 1

        # sub_heads: batch中所有样本的主体开始位置, sub_head: batch中随机一条样本的主体开始位置
        #
        return token_ids, token_ids_negative, token_ids_positive, masks, masks_negative, masks_positive, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, json_data['spo_list']

    def __len__(self):
        return len(self.dataset)


def my_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    token_ids, token_ids_negative, token_ids_positive, masks, masks_negative, masks_positive, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples = zip(*batch)
    batch_token_ids = pad_sequence(token_ids, batch_first=True)
    batch_token_ids_negative = pad_sequence(token_ids_negative, batch_first=True)
    batch_token_ids_positive = pad_sequence(token_ids_positive, batch_first=True)
    batch_masks = pad_sequence(masks, batch_first=True)
    batch_masks_negative = pad_sequence(masks_negative, batch_first=True)
    batch_masks_positive = pad_sequence(masks_positive, batch_first=True)
    batch_sub_heads = pad_sequence(sub_heads, batch_first=True)
    batch_sub_tails = pad_sequence(sub_tails, batch_first=True)
    batch_sub_head = pad_sequence(sub_head, batch_first=True)
    batch_sub_tail = pad_sequence(sub_tail, batch_first=True)
    batch_obj_heads = pad_sequence(obj_heads, batch_first=True)
    batch_obj_tails = pad_sequence(obj_tails, batch_first=True)

# 上面的对象是x， 下面的是y
    return {"token_ids": batch_token_ids.to(device),
            "token_ids_negative": batch_token_ids_negative.to(device),
            "token_ids_positive": batch_token_ids_positive.to(device),
            "mask": batch_masks.to(device),
            "mask_negative": batch_masks_negative.to(device),
            "mask_positive": batch_masks_positive.to(device),
            "sub_head": batch_sub_head.to(device),
            "sub_tail": batch_sub_tail.to(device),
            "sub_heads": batch_sub_heads.to(device),
            }, \
           {"mask": batch_masks.to(device),
            "sub_heads": batch_sub_heads.to(device),
            "sub_tails": batch_sub_tails.to(device),
            "obj_heads": batch_obj_heads.to(device),
            "obj_tails": batch_obj_tails.to(device),
            "triples": triples
            }


class MyRandomSampler(Sampler):
    def __call__(self, data_set):
        return np.random.permutation(len(data_set)).tolist()


def get_data_iterator(config, dataset, rel_vocab, is_test=False, collate_fn=my_collate_fn, entity_dict={}):
    dataset = MyDataset(config, dataset, rel_vocab, is_test, entity_dict)
    return TorchLoaderIter(dataset=dataset,
                           collate_fn=collate_fn,
                           batch_size=config.batch_size if not is_test else 1,
                           sampler=MyRandomSampler())

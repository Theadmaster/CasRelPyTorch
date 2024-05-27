import sys
sys.path.append('../')
import torch.nn as nn
import torch
from transformers import BertModel
from model.block.att import SelfAttention, MHAttention, MultiHeadSelfAttention
from model.block.tApe import tAPE

class CLCBAG(nn.Module):
    def __init__(self, config):
        super(CLCBAG, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_name)
        self.sub_heads_linear = nn.Linear(self.config.bert_dim * 2, 1)
        self.sub_tails_linear = nn.Linear(self.config.bert_dim * 2, 1)
        self.obj_heads_linear = nn.Linear(self.config.bert_dim, self.config.num_relations)
        self.obj_tails_linear = nn.Linear(self.config.bert_dim, self.config.num_relations)
        self.BiLSTM = nn.LSTM(input_size=self.config.bert_dim,
                            hidden_size=self.config.bert_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.LSTM = nn.LSTM(input_size=self.config.bert_dim,
                              hidden_size=self.config.bert_dim,
                              num_layers=1,
                              batch_first=True)
        self.GRU = nn.GRU(input_size=self.config.bert_dim,
                          hidden_size=self.config.bert_dim,
                          num_layers=1,
                          batch_first=True,
                          dropout=0.0)
        self.projection_head = nn.Sequential(
            nn.Linear(self.config.bert_dim, self.config.bert_dim),
            nn.ReLU(),
            nn.Linear(self.config.bert_dim, 64)
        )
        self.norm = nn.LayerNorm(self.config.bert_dim)
        self.attention = MHAttention(self.config.bert_dim * 2, 8)


    def projecter_cls(self, encoded_text_origin, encoded_text_augmented_positive, encoded_text_augmented_negative):
        cls_origin = encoded_text_origin[:, 0, :]
        cls_augmented_positive = encoded_text_augmented_positive[:, 0, :]
        cls_augmented_negative = encoded_text_augmented_negative[:, 0, :]
        cls_origin = self.projection_head(cls_origin)
        cls_augmented_positive = self.projection_head(cls_augmented_positive)
        cls_augmented_negative = self.projection_head(cls_augmented_negative)
        return cls_origin, cls_augmented_positive, cls_augmented_negative

    def get_encoded_text(self, token_ids, mask):
        encoded_text = self.bert(token_ids, attention_mask=mask)[0]
        return encoded_text

    def get_subs(self, encoded_text):
        encoded_text = self.add_block_Bilstm_selfatt(encoded_text)
        # shape: (batch_size:8, seq, bert_dim:768) => (8, seq, 1)
        pred_sub_heads = torch.sigmoid(self.sub_heads_linear(encoded_text))
        pred_sub_tails = torch.sigmoid(self.sub_tails_linear(encoded_text))
        return pred_sub_heads, pred_sub_tails

    def get_objs_for_specific_sub(self, sub_head_mapping, sub_tail_mapping, encoded_text):
        # lstm
        # encoded_text = self.LSTM(encoded_text)[0]
        # gru
        h0 = torch.zeros(self.GRU.num_layers, encoded_text.size(0), self.GRU.hidden_size).to(encoded_text.device)
        encoded_text = self.GRU(encoded_text, h0)[0]

        # sub_head_mapping [batch, 1, seq] * encoded_text [batch, seq, dim]
        sub_head = torch.matmul(sub_head_mapping, encoded_text)
        sub_tail = torch.matmul(sub_tail_mapping, encoded_text)
        sub = (sub_head + sub_tail) / 2
        # h(n) + v(k`sub)
        encoded_text = encoded_text + sub

        # 归一化放这 ****** new ******
        # 这里增加norm模块
        encoded_text = self.norm(encoded_text)

        # shape: (batch_size:8, seq, bert_dim:768) => (8, seq, num_relations)
        # sigmoid: 将值映射到 (0, 1)
        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        pred_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))
        return pred_obj_heads, pred_obj_tails

    def add_block_Bilstm_selfatt(self, encoded_text):
        # 添加模块
        # block = tAPE(d_model=768)
        # encoded_text = block(encoded_text)
        # BiLSTM
        encoded_text = self.BiLSTM(encoded_text)[0]

        # self attention
        encoded_text = self.attention(encoded_text, encoded_text, encoded_text, None)
        return encoded_text


    def forward(self, token_ids, token_ids_negative, token_ids_positive, mask, mask_negative, mask_positive, sub_head, sub_tail):
        encoded_text_positive = self.get_encoded_text(token_ids_positive, mask_positive)
        encoded_text_negative = self.get_encoded_text(token_ids_negative, mask_negative)
        # parameter -> (8, 134)
        # encoded_text -> (8, 134, 768)
        encoded_text = self.get_encoded_text(token_ids, mask)

        # 映射到同一向量空间
        cls_origin, cls_augmented_positive, cls_augmented_negative = self.projecter_cls(encoded_text, encoded_text_positive, encoded_text_negative)

        # pred_sub_heads, pred_sub_tails -> (8, 134, 1)
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)
        # sub_head_mapping, sub_tail_mapping -> (8, 1, 134)
        sub_head_mapping = sub_head.unsqueeze(1)
        sub_tail_mapping = sub_tail.unsqueeze(1)
        # pred_obj_heads, pred_obj_tails -> (8, 134, 18)
        pred_obj_heads, pre_obj_tails = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, encoded_text)

        return {
            "sub_heads": pred_sub_heads,
            "sub_tails": pred_sub_tails,
            "obj_heads": pred_obj_heads,
            "obj_tails": pre_obj_tails,
            "anchor": cls_origin,
            "positive": cls_augmented_positive,
            "negative": cls_augmented_negative
        }
#
# if __name__ == '__main__':
#
#     a = torch.tensor([[1, 2], [3, 4]])
#     b = torch.tensor([[1, 2], [3, 4]])
#     result = torch.matmul(a, b)
#     print(result)
#     print(a.unsqueeze(1))
import argparse
import os
import json
import torch
import torch.optim as optim
from model.casRel import CasRel
from model.callback import MyCallBack
from model.data import load_data, get_data_iterator
from model.config import Config
from model.evaluate import metric
import torch.nn.functional as F
from fastNLP import Trainer, LossBase

seed = 226
torch.manual_seed(seed)
# def seed_everything(seed=226):
#     os.environ['PYTHONHASHSEED'] = str(seed)
#
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#
# seed_everything()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_epoch', type=int, default=10)
parser.add_argument('--max_len', type=int, default=300)
parser.add_argument('--dataset', default='coronary_angiography', type=str, help='define your own dataset names')
parser.add_argument("--bert_name", default='bert-base-chinese', type=str, help='choose pretrained bert name')
parser.add_argument('--bert_dim', default=768, type=int)
parser.add_argument('--margin', default=0.2, type=float)
parser.add_argument('--re_weight', default=0.5, type=float)
parser.add_argument('--span_weight', default=0.5, type=float)
parser.add_argument('--cl', default=True, type=bool)
parser.add_argument('--aug_pipeline', default='RS', type=str)
args = parser.parse_args()
con = Config(args)

# 读取json
def init_entities_dict():
    data_folder = os.path.join('data', 'coronary_angiography')
    jsonl_file = os.path.join(data_folder, 'entities_set.json')
    with open(jsonl_file, "r") as json_file:
        entity_dict = json.load(json_file)
    return entity_dict
entity_dict = init_entities_dict()

class MyLoss(LossBase):
    def __init__(self):
        super(MyLoss, self).__init__()

    def get_loss(self, predict, target):
        mask = target['mask']

        def contrastive_loss(anchor, positive, negative, margin=con.margin):
            pos_similarity = F.cosine_similarity(anchor, positive)
            # 锚点与负例余弦相似度
            neg_similarity = F.cosine_similarity(anchor, negative)
            loss = F.relu(neg_similarity - pos_similarity + margin)
            return loss.mean()
        def loss_fn(pred, gold, mask):
            pred = pred.squeeze(-1)
            loss = F.binary_cross_entropy(pred, gold, reduction='none')
            if loss.shape != mask.shape:
                mask = mask.unsqueeze(-1)
            loss = torch.sum(loss * mask) / torch.sum(mask)
            return loss

        re_loss = loss_fn(predict['sub_heads'], target['sub_heads'], mask) + \
                  loss_fn(predict['sub_tails'], target['sub_tails'], mask) + \
                  loss_fn(predict['obj_heads'], target['obj_heads'], mask) + \
                  loss_fn(predict['obj_tails'], target['obj_tails'], mask)
        span_loss = contrastive_loss(predict['anchor'], predict['positive'], predict['negative'])

        if con.cl:
            return re_loss * con.re_weight + span_loss * con.span_weight
        else:
            return re_loss

    def __call__(self, pred_dict, target_dict, check=False):
        loss = self.get_loss(pred_dict, target_dict)
        return loss


if __name__ == '__main__':
    model = CasRel(con).to(device)
    data_bundle, rel_vocab = load_data(con.train_path, con.dev_path, con.test_path, con.rel_path)
    train_data = get_data_iterator(con, data_bundle.get_dataset('train'), rel_vocab, entity_dict=entity_dict)
    dev_data = get_data_iterator(con, data_bundle.get_dataset('dev'), rel_vocab, is_test=True)
    test_data = get_data_iterator(con, data_bundle.get_dataset('test'), rel_vocab, is_test=True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=con.lr)
    trainer = Trainer(train_data=train_data, model=model, optimizer=optimizer, batch_size=con.batch_size,
                      n_epochs=con.max_epoch, loss=MyLoss(), print_every=con.period, use_tqdm=True,
                      callbacks=MyCallBack(dev_data, rel_vocab, con))
    trainer.train()
    print("-" * 5 + "Begin Testing" + "-" * 5)
    metric(test_data, rel_vocab, con, model)

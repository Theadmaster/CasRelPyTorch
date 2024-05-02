import json


class Config(object):
    def __init__(self, args):
        self.args = args
        self.lr = args.lr
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.max_len = args.max_len
        self.bert_name = args.bert_name
        self.bert_dim = args.bert_dim

        self.train_path = 'data/' + self.dataset + '/train.json'
        self.test_path = 'data/' + self.dataset + '/test.json'
        self.dev_path = 'data/' + self.dataset + '/dev.json'
        self.rel_path = 'data/' + self.dataset + '/rel.json'
        self.num_relations = len(json.load(open(self.rel_path, 'r')))

        self.save_weights_dir = 'saved_weights/' + self.dataset + '/'
        self.save_logs_dir = 'saved_logs/' + self.dataset + '/'
        self.result_dir = 'results/' + self.dataset + '/'

        self.period = 200
        self.test_epoch = 3
        self.weights_save_name = 'model.pt'
        self.log_save_name = 'model.out'
        self.result_save_name = 'result.json'

        # 对比学习超参数
        self.temperature = args.temperature # infoNCE contrastive loss
        self.margin = args.margin  # margin contrastive loss
        self.re_weight = args.re_weight
        self.span_weight = args.span_weight

        # 是否开启对比loss
        self.cl = args.cl

        # 数据增强pipeline,增强策略用","隔开
        # 同义词替换（Synonym Replacement）: SR
        # 随机插入（Random Insertion）: RI
        # 随机交换（Random Swap）: RS
        # 随机删除（Random Deletion）: RD
        # 回译（Back Translation）: BT
        # 文本旋转（Text Rotation）: TR
        # 文本遮罩（Text Masking）: TM
        # 文本变形（Text Morphing）: TMo
        # 句子分割（Sentence Splitting）: SS
        # 随机重排（Random Shuffling）: RSh
        # 文本连接（Text Concatenation）: TC
        # 随机扰动（Random Perturbation）: RP
        # 词语丢弃（Word Dropout）: WD
        # 字符丢弃（Character Dropout）: CD
        # 随机噪声注入（Random Noise Injection）: RNI
        self.aug_pipeline = args.aug_pipeline


# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset,embedding):

        self.model_name = 'SVM'
        self.data_path = dataset + "/Data"

        self.vocab_path = dataset + '/vocab.pkl'  # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.dataSeg_save = dataset + '/data_cutword_list.txt'


        self.label_save_path = dataset + '/label_save_path.txt'
        # self.word2vec_model_save = dataset + '/data/word2vec_model.w2v'              # word2vec模型存储
        # self.dataSeg_save = dataset + '/data_cutword_list.txt'

        self.stopwords_path = dataset + '/hit_stopwords.txt'

        self.data_articles_vector = dataset + '/data_articles_vector.txt'



        self.dropout = 0.5                                            # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        # self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                          # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = 100
        # self.embed = self.embedding_pretrained.size(1)\
        #     if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (3, 4, 5)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=config.num_filters, kernel_size=(k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(in_features=config.num_filters * len(config.filter_sizes),out_features= config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # out = self.embedding(x[0])
        # x = x.type(torch.LongTensor)
        x = x.float()
        out = x.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
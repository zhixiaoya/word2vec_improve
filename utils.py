# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import pandas as pd
import jieba
import re
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora
import torch.nn as nn

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path,stopwords_path, max_size, min_freq):

    def is_number(num):
        pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
        result = pattern.match(num)
        if result:
            return True
        else:
            return False

    def sentence_seg():
        vocab_dic = {}
        i = 0
        labels_index = {}
        startTime = time.time()
        stopword_list = [k.strip() for k in open(stopwords_path, encoding='utf-8') if k.strip() != '']

        for cate_index,cur_dir in enumerate(os.listdir(file_path)):
            # label_dir = os.path.join(self.corpus_path,label)
            # file_list = os.listdir(label_dir)
            labels_index[cur_dir] = cate_index

            for line in pd.read_csv(os.path.join(file_path,cur_dir),header=None)[0]:
                cutword = [k for k in jieba.cut(line) if k not in stopword_list and not is_number(k)]
                for word in cutword:
                    vocab_dic[word] = vocab_dic.get(word,0) + 1
                i += 1
                if i % 1000 == 0:
                    print('前%d篇文章共花费%.2f秒'%(i,time.time()-startTime))
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1]>=min_freq],key = lambda x:x[1],reverse = True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}  # 排序后，每一个键分配一个索引号
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})  # 将该字典合并到vocab_dic中
        return vocab_dic

    vocab_dic =sentence_seg()

    return vocab_dic


def build_dataset(config):

    if os.path.exists(config.vocab_path):       # 加载词表
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:       # 若词表不存在，则根据train_data创建词表 (train_data,word_level/char_level,最大词长,...)
        vocab = build_vocab(config.train_path,config.stopwords_path, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))  # 将词表存入该文件
    print(f"Vocab size: {len(vocab)}")
    id_to_word = {value:key for (key, value) in vocab.items()}

    def is_number(num):
        pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
        result = pattern.match(num)
        if result:
            return True
        else:
            return False

    def sentences_seg(path,stopword_path,dataSeg_save):
        i = 0
        sentences_cutword = []
        contents = []
        labels = []
        startTime = time.time()
        stopword_list = [k.strip() for k in open(stopword_path, encoding='utf-8') if k.strip() != '']

        for cate_index, cur_dir in enumerate(os.listdir(path)):
                # label_dir = os.path.join(self.corpus_path,label)
                # file_list = os.listdir(label_dir)
            # labels_index[cur_dir] = cate_index

            for line in pd.read_csv(os.path.join(path, cur_dir),header=None)[0]:
                words_line = []
                cutword = [k for k in jieba.cut(line) if k not in stopword_list and not is_number(k)]
                sentences_cutword.append(cutword)
                labels.append(cate_index)
                # labels.append(cate_index)
                i += 1
                if i % 1000 == 0:
                    print('前%d篇文章共花费%.2f秒' % (i, time.time() - startTime))

                for word in cutword:
                    words_line.append(vocab.get(word, vocab.get(UNK)))  # 将token转换为该词的索引表示，若缺失，则默认值为UNK对应的值
                contents.append(torch.LongTensor(words_line))

        with open(dataSeg_save,'w') as file:
            for cutwords in sentences_cutword:
                file.write(' '.join(cutwords)+'\n')
        print(len(contents))
        print(len(labels))

        return contents,labels

    def tf_idf(dataSeg_save):
        corpus = pd.read_csv(dataSeg_save,header=None)[0]
        texts = [sentence.split(' ') for sentence in corpus]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        tf_idf_model = TfidfModel(corpus, normalize=False)
        word_tf_tdf = list(tf_idf_model[corpus])
        # print('词典:', dictionary.token2id)
        # print('词频:', corpus)
        # print('词的tf-idf值:', word_tf_tdf)
        return word_tf_tdf,dictionary.token2id


    def load_dataset(path,stopwords_path,dataSeg_save):

        # if not os.path.exists(dataSeg_save):  # 若分词后的文档不存在则 文档分词
        contents,labels = sentences_seg(path,stopwords_path,dataSeg_save)
        word_tf_idf,token_to_id = tf_idf(dataSeg_save)

        X = []


        embedding_pretrained = torch.tensor(np.load('./Sogou.CS/embedding_SougouNews.npz')["embeddings"].astype('float32'))
        embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)

        for i in range(len(labels)):    # 遍历每个文档

            # 文档的每个单词embedding后 * 该词的tf-idf值
            cur_content = np.zeros(300)
            sentence_embedding = embedding(contents[i])
            sentence_tfidf = np.array(word_tf_idf[i])[:,0].tolist()

            for j in range(len(contents[i])):

                # unk在tf-idf词典没有找到的问题
                # 根据contents从字典中取字，在根据字从tf—idf中取index，根据index从该句中取索引
                if id_to_word[int(contents[i][j].detach())] != '<UNK>':
                    tfidf_id = token_to_id[id_to_word[int(contents[i][j].detach())]]
                    word_tfidf = word_tf_idf[i][sentence_tfidf.index(tfidf_id)][1]
                else:
                    word_tfidf = 0
                cur_content += sentence_embedding[j].detach().numpy()* word_tfidf
            X.append(cur_content)
        print('X:',len(X))

        return X,labels  # [([句子1], 标签), ([句子2], 标签), ...]

    X,Y = load_dataset(config.data_path,config.stopwords_path,config.dataSeg_save)
    return X,Y


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./Sogou.CS/Data"
    vocab_dir = "./Sogou.CS/vocab.pkl"
    pretrain_dir = "./Sogou.CS/sgns.sogou.word"
    emb_dim = 300
    filename_trimmed_dir = "./Sogou.CS/embedding_SougouNews"
    stopword_path = './Sogou.CS/hit_stopwords.txt'
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        # tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, stopword_path,max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))    # 存储了字典

    embeddings = np.random.rand(len(word_to_id), emb_dim)   # 随机生成一个矩阵
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:        # line[0] 是一行开头的词
            idx = word_to_id[lin[0]]    # 取出该词在字典中的ID
            emb = [float(x) for x in lin[1:301]]    # 依次取出1-300数字
            embeddings[idx] = np.asarray(emb, dtype='float32')  # 按照 word_2_index的顺序取好词向量
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)   #？？？？？？？？

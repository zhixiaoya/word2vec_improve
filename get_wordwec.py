# coding=utf8
import pickle as pkl
import os
import numpy as np
import re
import time
import pandas as pd
import jieba

def build_vocab(corpus_path,stopword_path,cutword_save_path,label_save_path):

    def is_number(self,num):
        pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
        result = pattern.match(num)
        if result:
            return True
        else:
            return False

    def word_seg():
        i = 0
        labels_index = {}
        startTime = time.time()

        for cate_index,cur_dir in enumerate(os.listdir(corpus_path)):
            # label_dir = os.path.join(self.corpus_path,label)
            # file_list = os.listdir(label_dir)
            labels_index[cur_dir] = cate_index

            for line in pd.read_csv(os.path.join(corpus_path,cur_dir)):
                cutword = [k for k in jieba.cut(line) if k not in stopword_list and not is_number(k)]
                i += 1
                if i % 1000 == 0:
                    print('前%d篇文章共花费%.2f秒'%(i,time.time()-startTime))

                cutword_list.append(cutword)
                labels.append(cate_index)

            # for fname in file_list:
            #     f = open(os.path.join(label_dir, fname), encoding='gb2312', errors='ignore')
            #     texts.append(preprocess_keras(f.read()))
            #     f.close()
            #     labels.append(labels_index[label])

    def cutword_save():
        with open(cutword_save_path,'w') as file:
            for cutwords in cutword_list:
                file.write(' '.join(cutwords)+'\n')

    vocab_dic = {}
    cutword_list = []
    labels = []
    # self.corpus = pd.read_csv(self.corpus_path,header=None)
    stopword_list = [k.strip() for k in open(stopword_path, encoding='utf-8') if k.strip() != '']


    word_seg()
    cutword_save()
    np.array(labels).dump(label_save_path)

if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    corpus_path = "./Sogou.CS/Data/train.txt"
    # vocab_dir = "./Sogou.CS/vocab.pkl"
    pretrain_dir = "./Sogou.CS/sgns.sogou.word"
    emb_dim = 300
    filename_trimmed_dir = "./Sogou.CS/embedding_SougouNews"
    stopword_path = "./Sogou.CS/hit_stopwords.txt"
    cutword_save_path = './Sogou.CS/data_cutword_list.txt'
    label_save_path = './Sogou.CS/label_save_path.txt'

    # if os.path.exists(vocab_dir):
    #     word_to_id = pkl.load(open(vocab_dir, 'rb'))
    # else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        # tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
    build_vocab(corpus_path,stopword_path,cutword_save_path,label_save_path)
        # pkl.dump(word_to_id, open(vocab_dir, 'wb'))    # 存储了字典

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

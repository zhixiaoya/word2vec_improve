# coding:utf8

import gensim
import  pandas as pd

file = open('./Sogou.CS/data_cutword_list.txt')
documents = gensim.models.doc2vec.TaggedLineDocument(file)
model = gensim.models.Doc2Vec(documents,vector_size=100,window=8,min_count=100,workers=8)
file.close()

# 生成文本向量
print(len(model.docvecs))
print(model.docvecs[1])


# 下面再进行训练集测试集划分，svm训练等。
# 另外，关于
# gensim.models.doc2vec.TaggedLineDocument
# gensim.models.doc2vec.TaggedDocument
# gensim.models.doc2vec.TaggedBrownCorpus
# 有什么区别，数据的输入形式是什么
# 一个新的句子分词后能不能从已经训练好的doc2vec模型取出词向量
# doc2vec的基本用法
# 如何计算文档相似性，基本操作汇总
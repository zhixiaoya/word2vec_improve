# coding=utf-8

from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora
import pandas as pd

texts = [['我', ' 爱', '中国','上海'], ['青岛', ' 美丽', ' 地方'], ['我', ' 喜欢', ' 北京']]
# corpus = pd.read_csv('Sogou.CS/data_cutword_list.txt')
# texts = [sentence.split(' ') for sentence in texts]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tf_idf_model = TfidfModel(corpus, normalize=False)
word_tf_tdf = list(tf_idf_model[corpus])
print('词典:', dictionary.token2id)
print('词频:', corpus)
print('词的tf-idf值:', word_tf_tdf)



# dataSeg_save = './Sogou.CS/data_cutword_list.txt'
# corpus = pd.read_csv(dataSeg_save,header=None)[0]
# texts = [sentence.split(' ') for sentence in corpus]
# print(texts[0])
# dictionary = corpora.Dictionary(texts)
# corpus = [dictionary.doc2bow(text) for text in texts]
# tf_idf_model = TfidfModel(corpus, normalize=False)
# word_tf_tdf = list(tf_idf_model[corpus])
# # print('词典:', dictionary.token2id)
# print('词频:', corpus)
# print(len(word_tf_tdf))
# print('词的tf-idf值:', word_tf_tdf[0])

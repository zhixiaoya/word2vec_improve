# coding = utf8

import pickle as pkl

vocab = pkl.load(open('./Sogou.CS/vocab.pkl', 'rb'))
id_to_word = {value:key for (key, value) in vocab.items()}
print(id_to_word)
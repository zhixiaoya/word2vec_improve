# coding:utf8
import  numpy as np
from sklearn.metrics import accuracy_score


# 准确率
y_pred = [0,2,1,3]
y_true = [0,1,2,3]

print(accuracy_score(y_true,y_pred))
print(accuracy_score(y_true,y_pred,normalize=False))

# 精确率
from sklearn.metrics import precision_score
y_true = [0,1,2,0,1,2]
y_pred = [0,2,1,0,0,1]
print(precision_score(y_true,y_pred,average='macro'))
print(precision_score(y_true,y_pred,average='micro'))
print(precision_score(y_true,y_pred,average='weighted'))
# print(precision_score(y_true,y_pred,average='samples'))
print(precision_score(y_true,y_pred,average=None))



# 召回率

from sklearn.metrics import  recall_score
print(recall_score(y_true,y_pred,average='macro'))
print(recall_score(y_true,y_pred,average='micro'))
print(recall_score(y_true,y_pred,average='weighted'))
print(recall_score(y_true,y_pred,average=None))

from sklearn.metrics import f1_score
print(f1_score(y_true,y_pred,average='macro'))
print(f1_score(y_true,y_pred,average='micro'))
print(f1_score(y_true,y_pred,average='weighted'))
print(f1_score(y_true,y_pred,average=None))











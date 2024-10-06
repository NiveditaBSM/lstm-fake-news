from keras.models import load_model
from keras.preprocessing import sequence
from collections import Counter
import numpy as np
import os
import getEmbeddings
import cleanText


top_words = 5000
epoch_num = 5
batch_size = 64

if not os.path.isfile('./xtr_shuffled.npy') or \
    not os.path.isfile('./xte_shuffled.npy') or \
    not os.path.isfile('./ytr_shuffled.npy') or \
    not os.path.isfile('./yte_shuffled.npy'):
    getEmbeddings.clean_data()

if not os.path.isfile('./xtest.npy'):
    cleanText.clean_data()

xtr = np.load('./xtr_shuffled.npy')
new_data = np.load('./xtest.npy')

a=new_data.tolist()

print(a)
data=[]
data = a.split()

data_seq=[]
data_seq.append(data)
#print (data_seq)

cnt = Counter()
x_train = []
for x in xtr:
    x_train.append(x.split())
    for word in x_train[-1]:
        cnt[word] += 1  

# Storing most common words
most_common = cnt.most_common(top_words + 1)
word_bank = {}
id_num = 1
for word, freq in most_common:
    word_bank[word] = id_num
    id_num += 1

for news in data_seq:
    i = 0
    while i < len(news):
        if news[i] in word_bank:
            news[i] = word_bank[news[i]]
            i += 1
        else:
            del news[i]

max_review_length = 500

X_pred = sequence.pad_sequences(data_seq, maxlen=max_review_length)


model= load_model('lstm_model.h5')

yhat= model.predict_classes(X_pred)

b=yhat.tolist()
print (yhat[0,0])
os.unlink('./xtest.npy')

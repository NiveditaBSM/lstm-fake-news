"""
Fake news detection
The Doc2Vec pre-processing
"""

import numpy as np
import re
import string
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim import utils
from nltk.corpus import stopwords


def textClean(text):
    """
    Get rid of the non-letter and non-number characters
    """
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return (text)


def cleanup(text):
    text = textClean(text)
    text = text.translate(None,string.punctuation)
#text.translate(str.maketrans("", "", string.punctuation))
	
    return text


def constructLabeledSentences(data):
    sentences = []
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences
    


def clean_data():
    """
    Generate processed string
    """
    
    path1='test1.csv'
    data1 = pd.read_csv(path1) 
  #  path = 'train.csv'
    vector_dimension=300

  #  data = pd.read_csv(path)

    missing_rows = []
    for i in range(len(data1)):
        if data1.loc[i, 'text'] != data1.loc[i, 'text']:
            missing_rows.append(i)
   # data1 = data1.drop(missing_rows).reset_index().drop(['index','id'],axis=1)

    for i in range(len(data1)):
        data1.loc[i, 'text'] = cleanup(data1.loc[i,'text'])

    data1 = data1.sample(frac=1).reset_index(drop=True)
    X = data1.loc[0,'text']
    xtest = X
    np.save('xtest.npy',xtest)


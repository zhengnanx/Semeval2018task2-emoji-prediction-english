import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import sys
from scipy.sparse import hstack
import emoji

predict = open('predict_output_42902.txt', 'w')

text = emoji.readdata('./crawler/data/tweet_by_ID_20_4_2019__10_00_17.txt.text')
texts = [items for items in text]


label = emoji.readdata('./crawler/data/tweet_by_ID_20_4_2019__10_00_17.txt.labels')
labels = [items for items in label]


feature = emoji.feature_extraction(texts)
label = emoji2.labelencoder(labels)

print('Done feature and label')

classifier = emoji.classifier()
classifier.train(feature(texts), label(labels))

print('Done training')

trial = emoji.readdata('../test/us_test.text')
trialtexts = [items for items in trial]


predicted = classifier.predict(feature(trialtexts))
x = label.getname(predicted)

for item in x:
	predict.write(item+'\n')

predict.close()

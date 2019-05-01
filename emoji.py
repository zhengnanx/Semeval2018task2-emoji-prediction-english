import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import sys
from scipy.sparse import hstack
import spacy

predict = open('predict_output_4(stack).txt', 'w')


def readdata(path):
	file = open(path,'r').readlines()
	for line in file:
		line = line.strip()
		yield line

class feature_extraction:
	def __init__(self, texts):
		self.bivect = CountVectorizer(ngram_range=(1, 2),analyzer='word',binary= True)
		self.dictvect = DictVectorizer()
		self.charvect = CountVectorizer(ngram_range=(1, 6), analyzer='char', binary = True)
		self.charvect.fit(texts)
		self.bivect.fit(texts)
		return
	
	def __call__(self, texts):
		return hstack([self.charvect.transform(texts), self.bivect.transform(texts)])
		

class labelencoder:
	def __init__(self, labels):
		self.encoder = preprocessing.LabelEncoder()
		self.encoder.fit(labels)
		return

	def getname(self, encodedlabels):
		return self.encoder.inverse_transform(encodedlabels)


	def __call__(self, labels):

		return self.encoder.transform(labels)

class classifier:
	def __init__(self):

		self.classifier = LogisticRegression(penalty='l1',fit_intercept=True,solver='liblinear', multi_class='auto')
		return

	def train(self, feature, label):

		self.classifier.fit(feature, label)
		return

	def predict(self, feature):

		return self.classifier.predict(feature)




'''
Using Cross Validation to check corpus performance across models
'''
from sklearn.cross_validation import cross_val_score
import string
import re
import numpy as np
import pandas as pd
import sys
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def getColumns(inFile, delim="\t", header=False):
    """
    Get columns of data from inFile and load it into respective vectors.
    """
    cols = {}
    indexToName = {}
    specials=';.'
    for lineNum, line in enumerate(inFile):
        if lineNum == 0:
            headings = line.split(delim)
            i = 0
            for heading in headings:
                heading = re.sub(r'[^a-zA-Z0-9]',' ',heading)
                heading = heading.strip()
                if header:
                    cols[heading] = []
                    indexToName[i] = heading
                else:
                    # in this case the heading is actually just a cell
                    cols[i] = [heading]
                    indexToName[i] = i
                i += 1
        else:
            cells = line.split(delim)
            i = 0
            for cell in cells:
            	cell = re.sub(r'[^a-zA-Z0-9]',' ',cell)
            	cell = cell.strip()
            	cols[indexToName[i]] += [cell]
            	i += 1                
    return cols, indexToName

alldata=open("/home/sachin/Downloads/Bing/BingHackathonTrainingData.txt")
cols1,index1=getColumns(alldata,header=False)
alldata.close()

data_train_summary=cols1[5]
data_test_summary=cols2[5]
data_train_title=cols1[4]
data_train_author=cols1[3]
y_train=np.array(cols1[1])

print('data loaded')

print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
vectorizer = TfidfVectorizer(sublinear_tf=True,ngram_range=(1,2))#, max_df=0.5)
X_train_summary = vectorizer.fit_transform(data_train_summary)
X_train_title=vectorizer.transform(data_train_title)
X_train_author=vectorizer.transform(data_train_author)
X_train = X_train_summary+X_train_title+X_train_author
duration = time() - t0
print("n_samples: %d, n_features: %d" % X_train.shape)
print("Done in %fs" % (duration))

clf1=RidgeClassifier(tol=1e-2, solver="lsqr")	#Ridge Classifier
clf2=Perceptron(n_iter=50)						#Perceptron
clf3=PassiveAggressiveClassifier(n_iter=50)		#Passive Aggressive Classifier
clf4=KNeighborsClassifier(n_neighbors=10)		#KNeighbors Classifier
clf5=RandomForestClassifier(n_estimators=50)	#RandomForest Classifier
clf6=LinearSVC(loss='l2', penalty="l1",		#Train Liblinear model
	dual=False, tol=1e-3)
clf7=SGDClassifier(alpha=.0001, n_iter=50,		#Train SGD model
	penalty="l1")
clf8=SGDClassifier(alpha=.0001, n_iter=50,		#Train SGD with Elastic Net penalty
	penalty="elasticnet")
clf9=NearestCentroid()			#Train NearestCentroid (aka Rocchio classifier) without threshold
clf10=MultinomialNB(alpha=.01)					# Train sparse Naive Bayes classifiers
clf11=BernoulliNB(alpha=.01)					#Train sparse Naive Bayes(Bernoulli) classifiers
clf12=Pipeline([('feature_selection', 			#LinearSVC with L1-based feature selection
	LinearSVC(penalty="l2", dual=False, tol=1e-3)),
	('classification', LinearSVC())])
clf13=VotingClassifier(estimators=[
	('MultiNB',clf10),('BernNB',clf11),('KNN',clf4),('Ridge',clf1),
	('RF',clf5),('NearC',clf9),('Pipeline',clf12)
	],voting='hard')
clf_labels = ['Ridge Classifier', 'Perceptron', 'Passive Aggressive Classifier',
	'KNeighbors Classifier', 'RandomForest Classifier', 'Liblinear Model',
	'SGDModel', 'SGDModel With ElasticNet Penalty', 'NearestCentroid',
	'MultinomialNB Classifier', 'BernoulliNB Classifier', 'LinearSVC With L1']
print('10-fold cross validation:\n')
for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, 
	clf9, clf10, clf11, clf12], clf_labels):
	scores = cross_val_score(estimator=clf,
	X=X_train,
	y=y_train,
	cv=5,#10
	scoring='accuracy')
	print("Accuracy: %0.2f (+/- %0.2f) [%s]"% (scores.mean(), scores.std(), label))
clf13 = clf13.fit(X_train, y_train)
pred=clf13.predict(X_train)
pred=np.array(pred)
scorevc = metrics.accuracy_score(y_train, pred)
print("Score by Majority Vote: ",scorevc)
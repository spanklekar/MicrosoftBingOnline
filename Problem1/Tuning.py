'''
Tuning hyperparameters for LinearSVC, SVC Classifier, RandomForests
'''
from sklearn.learning_curve import validation_curve
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
import string
import re
import numpy as np
import pandas as pd
import sys
from time import time
from operator import itemgetter
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
    Get columns of data from inFile and put into appropriate columns.

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
alldata=open("/home/sachin/Downloads/Bing/BingHackathonTestData.txt")
cols2,index2=getColumns(alldata,header=False)
alldata.close()

data_train_summary=cols1[5]
data_test_summary=cols2[5]
data_train_title=cols1[4]
data_train_author=cols1[3]
data_test_title=cols2[4]
data_test_author=cols2[3]
y_train=np.array(cols1[1])
y_test=np.array(cols2[1])
testID=np.array(cols2[0])

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

print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test_summary = vectorizer.transform(data_test_summary)
X_test_title=vectorizer.transform(data_test_title)
X_test_author=vectorizer.transform(data_test_author)
X_test = X_test_summary+X_test_title+X_test_author

print('Learning parameters for LinearSVC Pipeline')
pipe_lr=Pipeline([
  ('redn', LinearSVC(penalty="l2", dual=False, tol=1e-3)),
  ('clf', LinearSVC())
  ])
#param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

param_range = [0.01, 0.1, 1.0, 10.0]

train_scores, test_scores = validation_curve(estimator=pipe_lr,
	X=X_train,
	y=y_train,
	param_name='clf__C',
	param_range=param_range,
	cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean,
	color='blue', marker='o',
	markersize=5,
	label='training accuracy')
plt.fill_between(param_range, train_mean + train_std,
	train_mean - train_std, alpha=0.15,
	color='blue')
plt.plot(param_range, test_mean,
	color='green', linestyle='--',
	marker='s', markersize=5,
	label='validation accuracy')
plt.fill_between(param_range,
	test_mean + test_std,
	test_mean - test_std,
	alpha=0.15, color='green')
plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.show()

### Using GridSearch to finetune hyperparameters with SVM Classifier

from sklearn.svm import SVC
print('Learning parameters for SVM Classifier')
pipe_svc = Pipeline([
	('clf', SVC(random_state=1))])
param_range = [0.01,0.1,1.0,10]
param_grid = [{'clf__C': param_range,
	'clf__kernel': ['linear']},
	{'clf__C': param_range,
	'clf__gamma': param_range,
	'clf__kernel': ['rbf']}
	]
gs = GridSearchCV(estimator=pipe_svc,
	param_grid=param_grid,
	scoring='accuracy',
	cv=10,
	n_jobs=-1)
gs = gs.fit(X_train, y_train)
print("SVM Classifier")
print(gs.best_score_)
print(gs.best_params_)

clf = gs.best_estimator_
clf.fit(X_train, y_train)
print("Best estimator for SVM: ",clf)

#Output:{'clf__C': 1.0, 'clf__kernel': 'linear'}
#Best estimator for SVM:  Pipeline(steps=[('clf', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
#  max_iter=-1, probability=False, random_state=1, shrinking=True,
#  tol=0.001, verbose=False))])

### Using GridSearch to finetune hyperparameters with RandomForestClassifier

print("Learning parameters from RandomForestClassifier")
gs = GridSearchCV(estimator=RandomForestClassifier(random_state=0,n_estimators=100),
	param_grid=[
	{'max_depth': [3, 4, 5, 6, 7]}],
	scoring='accuracy',
	cv=10)
scores = cross_val_score(gs,
	X_train,y_train,
	scoring='accuracy',
	cv=10)
print("RandomForestClassifier")
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

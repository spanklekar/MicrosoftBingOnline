'''
1. Calculates Test Scores for top performing classifiers
2. Write Test Scores to Disk
3. Calculates Majority Vote and writes to disk
'''
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
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

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

print('Data loaded')

print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
vectorizer = TfidfVectorizer(sublinear_tf=True,ngram_range=(1,3))#, max_df=0.5)
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
duration = time() - t0
print("n_samples: %d, n_features: %d" % X_test.shape)
print("Done in %fs" % (duration))

def writeToDisk(predn,clfname):
    target="./"+clfname+".txt"
    target=open(target,'w')
    target.write("{}\t{}\n".format("record_id", "topic"))
    for x in zip(testID, predn):
        target.write("{}\t{}\n".format(x[0], x[1]))
    target.close()
    print(clfname," output written to disk.")

clf1=RidgeClassifier(tol=1e-2, solver="lsqr")   #Ridge Classifier
clf1.fit(X_train, y_train)
pred = clf1.predict(X_test)
writeToDisk(pred,"RidgeClassifier")

clf2=MultinomialNB(alpha=.01)                   #Naive Bayes classifier
clf2.fit(X_train, y_train)
pred = clf2.predict(X_test)
writeToDisk(pred,"MultinomialNB")

clf3=BernoulliNB(alpha=.01)                     #Naive Bayes(Bernoulli) classifier
clf3.fit(X_train, y_train)
pred = clf3.predict(X_test)
writeToDisk(pred,"BernoulliNB")

clf4=KNeighborsClassifier(n_neighbors=10)       #KNeighbors Classifier
clf4.fit(X_train, y_train)
pred = clf4.predict(X_test)
writeToDisk(pred,"KNeighborsClassifier")

clf5=RandomForestClassifier(n_estimators=100)   #RandomForest Classifier
clf5.fit(X_train, y_train)
pred = clf5.predict(X_test)
writeToDisk(pred,"RandomForestClassifier")

clf6=Pipeline([('feature_selection',            #LinearSVC with L2-based feature selection
    LinearSVC(penalty="l2", dual=False, tol=1e-3)),
    ('classification', LinearSVC())])
clf6.fit(X_train, y_train)
pred = clf6.predict(X_test)
writeToDisk(pred,"LinearSVC")

clf7=NearestCentroid()                          #NearestCentroid (aka Rocchio classifier), no threshold 
clf7.fit(X_train, y_train)
pred = clf7.predict(X_test)
writeToDisk(pred,"NearestCentroid")

clf8=SVC(C=1.0, class_weight=None, coef0=0.0,   #SVC
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=1, shrinking=True,
    tol=0.001, verbose=False)
clf8.fit(X_train, y_train)
pred = clf8.predict(X_test)
writeToDisk(pred,"SVC")
'''
clf9=VotingClassifier(estimators=[
    ('Ridge',clf1),('MultiNB',clf2),('BernNB',clf3),('KNN',clf4),
    ('RF',clf5),('LinearSVC',clf6),('NearC',clf7),('SVC',clf8)
    ],voting='soft')

clf_labels = ['Ridge Classifier', 'MultinomialNB Classifier', 'BernoulliNB Classifier',
    'KNeighbors Classifier', 'RandomForest Classifier', 'LinearSVC',
    'NearestCentroid', 'SVC Model']
clf9 = clf9.fit(X_train, y_train)
pred=clf9.predict(X_train)
writeToDisk(pred,"MajorityVote")
'''
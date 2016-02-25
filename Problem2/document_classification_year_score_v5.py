'''
1. Calculates Test Scores for Ridge and SVR Regressors 
2. Write Test Scores to Disk
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
from sklearn.linear_model import Ridge
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
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

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
y_train=np.array(cols1[2])
y_test=np.array(cols2[2])
testID=np.array(cols2[0])

print('Data loaded')

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

print(type(X_train))
print(type(X_test))
X_train=X_train.astype(int)
X_test=X_test.astype(int)
y_train=y_train.astype(int)

ridgeclf=Ridge(alpha=0.7, copy_X=True, fit_intercept=True, max_iter=None, 
	normalize=False, solver='auto', tol=0.001)
ridgeclf.fit(X_train, y_train)
pred = ridgeclf.predict(X_test)
target="./"+"RidgeRegression"+".txt"
target=open(target,'w')
#target.write(pred)
target.write("{}\t{}\n".format("record_id", "publication_year"))
for x in zip(testID, pred):
  target.write("{}\t{}\n".format(x[0], int(x[1])))
target.close()
print("Ridge output written to disk.")

svrclf=SVR(C=1.0, coef0=0.0,   #SVR
    degree=6, gamma='auto', kernel='linear',
    max_iter=-1, shrinking=True, tol=0.001, verbose=False)
svrclf.fit(X_train, y_train)
pred = svrclf.predict(X_test)
target="./"+"SVR"+".txt"
target=open(target,'w')
#target.write(pred)
target.write("{}\t{}\n".format("record_id", "publication_year"))
for x in zip(testID, pred):
  target.write("{}\t{}\n".format(x[0], int(x[1])))
target.close()
print("SVR output written to disk.")

'''
lassoclf=Lasso(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=100,
   selection='random', tol=0.1, warm_start=False)
lassoclf.fit(X_train, y_train)
pred = lassoclf.predict(X_test)
target="./"+"LassoRegression"+".txt"
target=open(target,'w')
#target.write(pred)
target.write("{}\t{}\n".format("record_id", "publication_year"))
for x in zip(testID, pred):
  target.write("{}\t{}\n".format(x[0], int(x[1])))
target.close()
print("Lasso output written to disk.")

clf2=Perceptron(n_iter=100)            #Perceptron
clf2.fit(X_train, y_train)
pred = clf2.predict(X_test)
target="./"+"Perceptron"+".txt"
target=open(target,'w')
#target.write(pred)
target.write("{}\t{}\n".format("record_id", "publication_year"))
for x in zip(testID, pred):
  target.write("{}\t{}\n".format(x[0], int(x[1])))
target.close()
print("Perceptron output written to disk.")
'''
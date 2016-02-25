from collections import defaultdict, Counter
from glob import glob
import sys
from sklearn import metrics
import numpy as np

glob_files = sys.argv[1]
loc_outfile = sys.argv[2]
#pred=[]
predset={}

def kaggle_bag(glob_files, loc_outfile, method="average", weights="uniform"):
  if method == "average":
    scores = defaultdict(list)
  with open(loc_outfile,"w") as outfile:
    for i, glob_file in enumerate( glob(glob_files) ):
      print ("parsing:", glob_file)
      # sort glob_file by first column, ignoring the first line
      lines = open(glob_file).readlines()
      lines = [lines[0]] + sorted(lines[1:])
      for e, line in enumerate( lines ):
        if i == 0 and e == 0:
          outfile.write(line)
        if e > 0:
          row = line.strip().split("\t")
          scores[(e,row[0])].append(row[1])
    for j,k in sorted(scores):
      #outfile.write("%s\t%s\n"%(k,Counter(scores[(j,k)]).most_common(1)[0][0]))
      #pred.append(Counter(scores[(j,k)]).most_common(1)[0][0])
      predset[int(k)]=Counter(scores[(j,k)]).most_common(1)[0][0]
    for key,value in sorted(predset.items()):      
      outfile.write(str(key)+"\t"+value+"\n")
    print("wrote to %s"%loc_outfile)
    outfile.close()

kaggle_bag(glob_files, loc_outfile)

pred=[]
y_test=[]
f4= open("/home/sachin/Downloads/Bing/Ytest.txt")
for line in f4:
    y_test.append(line.strip())
y_test=np.array(y_test)
f4.close()
f4= open("/home/sachin/Downloads/Bing/check/out")
i=0
for line in f4:
    if (i==0):
      i=i+1
      continue

    linespl=line.split("\t")
    pred.append(linespl[1].strip())
pred=np.array(pred)
f4.close()
print(type(pred))
print(type(y_test))
print(pred.shape)
print(y_test.shape)
score = metrics.accuracy_score(y_test, pred)
print(score)

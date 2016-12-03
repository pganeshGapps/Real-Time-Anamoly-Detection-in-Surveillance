from sklearn.datasets import load_iris
import numpy as np  # used for splitting
from sklearn import tree  #Classifier

#######################IMPORT DATASETS

iris=load_iris()

#Metadata
'''
print( iris.feature_names)
print( iris.target_names)
'''
#Data & target
'''
print(iris.data[0])
print(iris.target[0])
# OR
for i in range(len(iris.data)):
    print("index: %d: features= %s, target= %d "%(i,iris.data[i],iris.target[i]))
    if (i > 5):
        break;
'''
####################### Split Data into Testing and Training datasets
test_idx=[0,50,100]  # taking one sample from each class

#Training data
train_target= np.delete(iris.target,test_idx)    #here we r removing 3 test_idx entries
train_data= np.delete(iris.data,test_idx, axis=0)

#Testing data
test_target=iris.target[test_idx]
test_data= iris.data[test_idx]

####################### Training Classifier
clf=tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

####################### Prediction
print("Original: %s" %(test_target))
print("Predicted: %s" %(clf.predict(test_data)))

####################### Visualization

import pydotplus
outfile=open("viz.pdf","w")
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graphviz.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")


print("Program Ends here ...")

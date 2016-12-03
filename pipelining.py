from sklearn.datasets import load_iris
iris=load_iris()

x=iris.data
y=iris.target

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.5) # x=data,y=target

'''
    USING DIFFERENT CLASSIFIERS WITH SAME HIGH LEVEL INTERFACE
    Here, for example, we are giving two clfs. 
'''
#from sklearn import tree
#my_clf=tree.DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
my_clf=KNeighborsClassifier()

my_clf.fit(x_train,y_train)

predictions= my_clf.predict(x_test)
#print(predictions)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,predictions)
print(accuracy)

'''
scrappy_clf.py  :- user defined Classifier
'''
import random
from scipy.spatial import distance
def euc_distace(a,b): #computes eucludian distance between given two points
    return distance.euclidean(a,b)

class Scrappy_clf():
    def fit(self,x_train,y_train):
        self.x_train=x_train
        self.y_train=y_train

    def predict(self,x_test):
        predictions=[]
        for row in(x_test) :
            label=self.closest(row)       #label=random.choice(self.y_train) #using self.y_train
            predictions.append(label)
        return predictions

    def closest(self,row):       #returns label of closest data point using euclidian_idx
        closest_dist=euc_distace(row,x_train[0])
        closest_idx=0;
        for i in range(1,len(x_train)):
            e_dist=euc_distace(row,x_train[i])
            if (e_dist < closest_dist) :
                closest_dist=e_dist
                closest_idx=i
        return self.y_train[closest_idx]

'''Class defination ends here '''
from sklearn.datasets import load_iris
iris=load_iris()

x=iris.data
y=iris.target

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.5) # x=data,y=target

'''
    training using Scrappy_clf
'''
my_clf=Scrappy_clf()

my_clf.fit(x_train,y_train)

predictions= my_clf.predict(x_test)
#print(predictions)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,predictions)
print(accuracy)

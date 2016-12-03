from sklearn import tree
#program to predict gender , assuming that generally boys are taller and have higher weights than girls.
#Assumption : all boys & girls have same age

'''
    Data Acquisition & Preprocessing
'''
#feature: [weight,height]
features=[[73,173],[48,153],[75, 170],[52,156]]
#labels=[M,F,M,F]   # need to be preprocessed
labels=[0,1,0,1]

'''
    Split data into Training & Testing datasets
'''
# here train to test = 1:0

'''
    Training
'''
clf=tree.DecisionTreeClassifier()
clf=clf.fit(features,labels)

'''
    Testing
'''
print(clf.predict([[72,175],[47,150]])) # Expected label list [0,1]

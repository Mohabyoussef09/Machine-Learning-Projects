
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Wine.csv')

X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Put list of dimensionality reduction you will use
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import LocallyLinearEmbedding
dmList=[FactorAnalysis(n_components = 2),LocallyLinearEmbedding(n_components=2)]

# Put list of classification Algorithms you will use
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
classificationList=[GaussianNB(),svm.SVC(),tree.DecisionTreeClassifier()]

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
# Calcualte Accuracy
from sklearn.metrics import accuracy_score,roc_curve

for dm in dmList:
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train_new = sc.fit_transform(X_train)
    X_test_new = sc.transform(X_test)
    
    currentDm = dm
    X_train_new = currentDm.fit_transform(X_train_new)
    X_test_new = currentDm.transform(X_test_new)
    
    for classification in classificationList:
        classifier = classification
        classifier.fit(X_train_new, y_train)
        
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test_new)

        cm = confusion_matrix(y_test, y_pred)
        accuracy=accuracy_score(y_test, y_pred)
        roc=roc_curve(y_test, y_pred,pos_label =2)

        print("Dimensionality Reduction:",dm)
        print("Classification Algorithm:",classification)
        print("Accuracy:",accuracy)
        print("Confusion Matrix:\n",cm)
        print("ROC:",roc)
        print("---------------------------------------------------------------")
        
        X_train_new = sc.fit_transform(X_train)
        X_test_new = sc.transform(X_test)

        currentDm = dm
        X_train_new = currentDm.fit_transform(X_train_new)
        X_test_new = currentDm.transform(X_test_new)
        
       
        




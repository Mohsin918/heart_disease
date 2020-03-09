import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv("C:\\Users\\Mohsin\\Desktop\\heart.csv")
dataset = pd.get_dummies(df,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])
#Feature Scaling
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
#Creating dependent and independent varialbles
X = dataset.drop(['target'],axis=1)
Y = dataset['target']
from sklearn.model_selection import cross_val_score
knn_scores= []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn_classifier,X,Y,cv=10)
    knn_scores.append(score.mean())
#At k=12 we have the highest accuracy i.e 85%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
 #Fitting out data to the K-NN regression
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=12,p=2,metric='minkowski')
classifier.fit(X_train,Y_train)
#Prediction the test set
Y_pred = classifier.predict(X_test)
#Creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
#----------------------------------------------
#Accuracy is 85%

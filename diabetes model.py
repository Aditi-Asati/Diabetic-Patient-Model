import pandas as pd
df = pd.read_csv('datasets_228_482_diabetes.csv')
df.head()

df.groupby('Outcome')['SkinThickness','Insulin','BloodPressure','BMI'].mean()

df0 = df[df['Outcome'] == 0]        
df1 = df[df['Outcome'] == 1]
df0

df0['SkinThickness'] = df0['SkinThickness'].replace(0,19.66)
df1['SkinThickness'] = df1['SkinThickness'].replace(0,22.16)

df0['Insulin'] = df0['Insulin'].replace(0,68.79)
df1['Insulin'] = df1['Insulin'].replace(0,100.34)

df0['BloodPressure'] = df0['BloodPressure'].replace(0,68.18)
df1['BloodPressure'] = df1['BloodPressure'].replace(0,70.82)

df0['BMI'] = df0['BMI'].replace(0,30.30)
df1['BMI'] = df1['BMI'].replace(0,35.14)

df = df0.append(df1).reset_index().drop(['index'],axis=1)

df.head(20)

#Features
X = df.drop(['Outcome'],axis=1)
#Label
Y = df[['Outcome']]

#Training and testing split
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y, random_state=8)

"""**1. KNN ALGO**"""

# 1. KNN ALGO
from sklearn.neighbors import KNeighborsClassifier
kmodel = KNeighborsClassifier(n_neighbors=10)
kmodel.fit(xtrain , ytrain)

print(kmodel.score(xtrain,ytrain))

print('*********************************************************************')

print(kmodel.score(xtest,ytest))

from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(ytrain,kmodel.predict(xtrain)))
print(accuracy_score(ytest,kmodel.predict(xtest)))

print('****************************************************')

print(confusion_matrix(ytrain,kmodel.predict(xtrain)))
print(confusion_matrix(ytest,kmodel.predict(xtest)))

ypred_train_prob = kmodel.predict_proba(xtrain)
ypred_test_prob = kmodel.predict_proba(xtest)

from sklearn.metrics import roc_curve
fpr1,tpr1,thresh1 = roc_curve(ytrain , ypred_train_prob[:,1] )
fpr2,tpr2,thresh2 = roc_curve(ytest , ypred_test_prob[:,1] )

from sklearn.metrics import roc_auc_score
auc_score_1 = roc_auc_score(ytrain,ypred_train_prob[:,1])
auc_score_2 = roc_auc_score(ytest,ypred_test_prob[:,1])

print(auc_score_1)
print(auc_score_2)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
plt.plot(fpr1,tpr1,linestyle='--',label='training_data_KNN')
plt.plot(fpr2,tpr2,linestyle='--',label='testing_data_KNN')

plt.title('ROC_Curve_KNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()
plt.show()

"""**Tune the value of k**"""

tr_acc = []
ts_acc = []
for i in range(1,15):
  ksmodel = KNeighborsClassifier(n_neighbors=i)
  ksmodel.fit(xtrain,ytrain)
  tr_acc.append(ksmodel.score(xtrain,ytrain))

  ts_acc.append(ksmodel.score(xtest,ytest))

plt.plot(range(1,15),tr_acc)
plt.plot(range(1,15),ts_acc)
plt.show()

"""**2.DECISION TREE CLASSIFIER**"""

# 2.DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
dmodel = DecisionTreeClassifier(max_depth=5)

dmodel.fit(xtrain,ytrain)

#Training accuracy
print(dmodel.score(xtrain,ytrain))
#Testing Accuracy
print(dmodel.score(xtest,ytest))

from sklearn.tree import export_graphviz
dot_data = export_graphviz(dmodel, feature_names=xtrain.columns)

from IPython.display import Image
import pydotplus

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(ytrain,dmodel.predict(xtrain)))
print(accuracy_score(ytest,dmodel.predict(xtest)))

print('****************************************************')

print(confusion_matrix(ytrain,dmodel.predict(xtrain)))
print(confusion_matrix(ytest,dmodel.predict(xtest)))

ypred_train_prob_d = dmodel.predict_proba(xtrain)
ypred_test_prob_d = dmodel.predict_proba(xtest)

from sklearn.metrics import roc_curve
fprd1,tprd1,threshd1 = roc_curve(ytrain , ypred_train_prob_d[:,1] )
fprd2,tprd2,threshd2 = roc_curve(ytest , ypred_test_prob_d[:,1] )

from sklearn.metrics import roc_auc_score
auc_score_3 = roc_auc_score(ytrain,ypred_train_prob_d[:,1])
auc_score_4 = roc_auc_score(ytest,ypred_test_prob_d[:,1])

print(auc_score_3)
print(auc_score_4)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
plt.plot(fprd1,tprd1, linestyle='--', label='training_data_Decision Tree Classifier')
plt.plot(fprd2,tprd2, linestyle='--', label='testing_data_Decision Tree Classifier')

plt.title('ROC_Curve_Decision Tree Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()
plt.show()

"""**3.NAIVE BAYES**"""

# 3. Naive Bayes
from sklearn.naive_bayes import GaussianNB
nmodel = GaussianNB()
nmodel.fit(xtrain,ytrain)

#Training accuracy
print(nmodel.score(xtrain,ytrain))
#Testing Accuracy
print(nmodel.score(xtest,ytest))

from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(ytrain,nmodel.predict(xtrain)))
print(accuracy_score(ytest,nmodel.predict(xtest)))

print('****************************************************')

print(confusion_matrix(ytrain,nmodel.predict(xtrain)))
print(confusion_matrix(ytest,nmodel.predict(xtest)))

ypred_train_prob_n = nmodel.predict_proba(xtrain)
ypred_test_prob_n = nmodel.predict_proba(xtest)

from sklearn.metrics import roc_curve
fprn1,tprn1,threshn1 = roc_curve(ytrain , ypred_train_prob_n[:,1] )
fprn2,tprn2,threshn2 = roc_curve(ytest , ypred_test_prob_n[:,1] )

from sklearn.metrics import roc_auc_score
auc_score_5 = roc_auc_score(ytrain,ypred_train_prob_n[:,1])
auc_score_6 = roc_auc_score(ytest,ypred_test_prob_n[:,1])

print(auc_score_5)
print(auc_score_6)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
plt.plot(fprn1,tprn1, linestyle='--', label='training_data_Naive Bayes')
plt.plot(fprn2,tprn2, linestyle='--', label='testing_data_Naive Bayes')

plt.title('ROC_Curve_Naive Bayes')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()
plt.show()

# 4.Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rmodel = RandomForestClassifier(n_estimators=21,max_depth=6)
rmodel.fit(xtrain,ytrain)

print(accuracy_score(ytrain,rmodel.predict(xtrain)))
print(accuracy_score(ytest,rmodel.predict(xtest)))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytrain,rmodel.predict(xtrain)))
print(confusion_matrix(ytest,rmodel.predict(xtest)))

ypred_train_prob_r = rmodel.predict_proba(xtrain)
ypred_test_prob_r = rmodel.predict_proba(xtest)

from sklearn.metrics import roc_curve
fprr1,tprr1,threshr1 = roc_curve(ytrain , ypred_train_prob_r[:,1] )
fprr2,tprr2,threshr2 = roc_curve(ytest , ypred_test_prob_r[:,1] )

from sklearn.metrics import roc_auc_score
auc_score_7 = roc_auc_score(ytrain,ypred_train_prob_r[:,1])
auc_score_8 = roc_auc_score(ytest,ypred_test_prob_r[:,1])

print(auc_score_7)
print(auc_score_8)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
plt.plot(fprr1,tprr1, linestyle='--', label='training_data_Random Forest Classifier')
plt.plot(fprr2,tprr2, linestyle='--', label='testing_data_Random Forest Classifier')

plt.title('ROC_Curve_Random Forest Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()
plt.show()

"""**5.SUPPORT VECTOR MACHINE**"""

# 5.Support Vector Machine
from sklearn.svm import SVC
smodel = SVC(probability=True)
smodel.fit(xtrain,ytrain)

print(accuracy_score(ytrain,smodel.predict(xtrain)))
print(accuracy_score(ytest,smodel.predict(xtest)))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytrain,smodel.predict(xtrain)))
print(confusion_matrix(ytest,smodel.predict(xtest)))

ypred_train_prob_s = smodel.predict_proba(xtrain)
ypred_test_prob_s = smodel.predict_proba(xtest)

from sklearn.metrics import roc_curve
fprs1,tprs1,threshs1 = roc_curve(ytrain , ypred_train_prob_s[:,1] )
fprs2,tprs2,threshs2 = roc_curve(ytest , ypred_test_prob_s[:,1] )

from sklearn.metrics import roc_auc_score
auc_score_9 = roc_auc_score(ytrain,ypred_train_prob_s[:,1])
auc_score_10 = roc_auc_score(ytest,ypred_test_prob_s[:,1])

print(auc_score_9)
print(auc_score_10)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
plt.plot(fprs1,tprs1, linestyle='--', label='training_data_Support Vector Machine')
plt.plot(fprs2,tprs2, linestyle='--', label='testing_data_Support Vector Machine')

plt.title('ROC_Curve_Support Vector Machine')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()
plt.show()

"""**6.LOGISTIC REGRESSION**"""

# 6.Logistic Regression
from sklearn.linear_model import LogisticRegression
lmodel = LogisticRegression()
lmodel.fit(xtrain,ytrain)

print(accuracy_score(ytrain,lmodel.predict(xtrain)))
print(accuracy_score(ytest,lmodel.predict(xtest)))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytrain,lmodel.predict(xtrain)))
print(confusion_matrix(ytest,lmodel.predict(xtest)))

ypred_train_prob_l = lmodel.predict_proba(xtrain)
ypred_test_prob_l = lmodel.predict_proba(xtest)

from sklearn.metrics import roc_curve
fprl1,tprl1,threshl1 = roc_curve(ytrain , ypred_train_prob_l[:,1] )
fprl2,tprl2,threshl2 = roc_curve(ytest , ypred_test_prob_l[:,1] )

from sklearn.metrics import roc_auc_score
auc_score_11 = roc_auc_score(ytrain,ypred_train_prob_l[:,1])
auc_score_12 = roc_auc_score(ytest,ypred_test_prob_l[:,1])

print(auc_score_11)
print(auc_score_12)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
plt.plot(fprl1,tprl1, linestyle='--', label='training_data_Logistic Regression')
plt.plot(fprl2,tprl2, linestyle='--', label='testing_data_Logistic Regression')

plt.title('ROC_Curve_Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()
plt.show()

df.head(10)

lst = ["Non Diabetic", "Diabetic"]
lst

"""**Predicting Unknown Value**"""

# 0 = Non Diabetic, 1 = Diabetic
print(kmodel.predict([[8,97,88.75,25,150,40,1.32,45]]))
print(dmodel.predict([[8,97,88.75,25,150,40,1.32,45]]))
print(nmodel.predict([[8,97,88.75,25,150,40,1.32,45]]))
print(rmodel.predict([[8,97,88.75,25,150,40,1.32,45]]))
print(smodel.predict([[8,97,88.75,25,150,40,1.32,45]]))
print(lmodel.predict([[8,97,88.75,25,150,40,1.32,45]]))
print("Thanks for visiting!")

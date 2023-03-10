import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import category_encoders as ce
import tkinter
import matplotlib
matplotlib.use('Agg')
from sklearn import metrics

data = pd.read_csv("D:\doantotnghiep\datairender.csv")
X = data.drop(['is_paid'], axis=1)
y = data['is_paid']
def print_score(y_test, y_pred):
  print(classification_report(y_test, y_pred))
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
encoder = ce.OrdinalEncoder(cols=['region','language','package'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression(class_weight='balanced')
model_LR.fit(X_train,y_train)
y_pred_LR = model_LR.predict(X_test)
print('LogisticRegression')
print_score(y_test, y_pred_LR)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_LR)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()
plt.savefig('bieudoLR.png')

from sklearn import svm
model_SVM = svm.SVC(kernel="linear", class_weight='balanced')
model_SVM.fit(X_train,y_train)
y_pred_SVM = model_SVM.predict(X_test)
print('svm')
print_score(y_test, y_pred_SVM)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_SVM)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()
plt.savefig('bieudoSVM.png')

from sklearn.neighbors import KNeighborsClassifier
model_kNN = KNeighborsClassifier(n_neighbors=2)
model_kNN.fit(X_train,y_train)
y_pred_kNN = model_kNN.predict(X_test)
print('KNeighborsClassifier')
print_score(y_test, y_pred_kNN)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_kNN)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()
plt.savefig('bieudoknn.png')

from sklearn.tree import DecisionTreeRegressor
model_Tree = DecisionTreeRegressor(random_state=0)
model_Tree.fit(X_train,y_train)
y_pred_Tree = model_Tree.predict(X_test)
print('DecisionTreeRegressor')
print_score(y_test, y_pred_Tree.round())
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_Tree.round())
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()
plt.savefig('bieudotree.png')

print('Data correlation between two classes')
print(data['is_paid'].value_counts())
data['is_paid'].value_counts().plot(kind='bar')
plt.savefig('ssdata.png')
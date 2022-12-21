import pandas as pd
import category_encoders as ce
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.read_csv("D:\doantotnghiep\datairender.csv")
X = data.drop(['is_paid'], axis=1)
y = data['is_paid']

encoder = ce.OrdinalEncoder(cols=['region','language','package'])
X_model = encoder.fit_transform(X)
model = LogisticRegression(class_weight='balanced')
model.fit(X_model,y)
pickle.dump(model, open('model.pkl', 'wb'))
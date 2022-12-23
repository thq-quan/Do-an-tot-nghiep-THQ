import numpy as np
import pandas as pd
import streamlit as st
import pickle
import category_encoders as ce
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("D:\doantotnghiep\datairender.csv")
X = data.drop(['is_paid'], axis=1)
y = data['is_paid']
encoder = ce.OrdinalEncoder(cols=['region','language','package'])
X = encoder.fit_transform(X)
model_LR = LogisticRegression(class_weight='balanced')
model_LR.fit(X,y)

st.set_page_config(
    page_title= "Ứng dụng dự đoán",
    page_icon= "👋"
)

st.write("""
# ỨNG DỤNG DỰ ĐOÁN KHÁCH HÀNG CỦA IRENDER VIETNAM
Ứng dụng này để dự đoán **Khách hàng trả tiền**!
""")

# data của ng dùng nhập vào để dự đoán
st.sidebar.header('Thông số đầu vào của khách hàng muốn dự đoán')
re = data['region']
re_list = list(set(re))
la = data['language']
la_list = list(set(la))
pa = data['package']
pa_list = list(set(pa))

def user_input_features():
    region = st.sidebar.selectbox('Region',re_list)
    timezone = st.sidebar.slider('Timezone',-12,12,0)
    language = st.sidebar.selectbox('Language',la_list,)
    package = st.sidebar.selectbox('Package',pa_list)
    hours_use = st.sidebar.text_input('hours_use(hours)','0')
    if hours_use == '':
        st.error('hours_use is empty', icon="🚨")
        hours_use = 0
    sum_length = st.sidebar.text_input('sum_length(GB)','0')
    if sum_length == '':
        st.error('sum_length is empty', icon="🚨")
        sum_length = 0
    data = {'region': region,
            'timezone': timezone,
            'language': language,
            'package': package,
            'hours_use': hours_use,
            'sum_length': sum_length
            }
    features = pd.DataFrame(data, index=[0])
    return features
# kq sau khi ng dùng nhập   
df = user_input_features()
st.subheader('Thông số đầu vào của khách hàng muốn dự đoán')
st.write(df)

df = encoder.transform(df)

# hàm dự đoán và tỷ lệ giữa 2 lớp
prediction = model_LR.predict(df)
prediction_proba = model_LR.predict_proba(df)
# show dự đoán của mô hình
st.subheader('Kết quả dự đoán')
paid = np.array(['Khách hàng miễn phí','Khách hàng trả tiền'])
st.write(paid[prediction])
# show tỉ lệ giữa 2 lớp
st.subheader('Xác suất dự đoán của hai lớp')
st.write(prediction_proba)
st.write("**DataFame**")
st.dataframe(data)
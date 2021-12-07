import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle

st.write("""
# Credit Card Approval Prediction App

This app predicts the credit card approval probablity
""")
#Get Input

st.header('User Input Parameters')

def user_input_features():
    gender = st.selectbox("CODE_GENDER",('M','F'))
    own_car = st.selectbox("FLAG_OWN_CAR",('Y','N'))
    own_realty = st.selectbox("FLAG_OWN_REALTY",('Y','N'))
    own_work_phone = st.selectbox("FLAG_WORK_PHONE",(0,1))
    own_email = st.selectbox("FLAG_EMAIL",(0,1))
    own_phone = st.selectbox("FLAG_PHONE",(0,1))
    cnt_children = st.number_input("CNT_CHILDREN",min_value=0,max_value=20,step=1)
    amt_income_total = st.number_input("AMT_INCOME_TOTAL",min_value=0.0,max_value=2000000.0)
    days_birth = st.number_input("DAYS_BIRTH",min_value=-30000,max_value=0,step=1)
    days_employed = st.number_input("DAYS_EMPLOYED",min_value=-20000,max_value=400000 ,step=1)
    cnt_fam_members = st.number_input("CNT_FAM_MEMBERS",min_value=0,max_value=20,step=1)
    name_income_type = st.selectbox("NAME_INCOME_TYPE",('Working', 'Commercial associate', 'Pensioner', 'State servant','Student'))
    name_education_type = st.selectbox("NAME_EDUCATION_TYPE",('Higher education', 'Secondary / secondary special','Incomplete higher', 'Lower secondary', 'Academic degree'))
    name_family_status = st.selectbox("NAME_FAMILY_STATUS",('Civil marriage', 'Married', 'Single / not married', 'Separated','Widow'))
    name_housing_type = st.selectbox("NAME_HOUSING_TYPE",('Rented apartment', 'House / apartment', 'Municipal apartment','With parents', 'Co-op apartment', 'Office apartment'))

    data = {'CNT_CHILDREN': cnt_children,
            'AMT_INCOME_TOTAL': amt_income_total,
            'NAME_INCOME_TYPE': name_income_type,
            'NAME_EDUCATION_TYPE': name_education_type,
            'NAME_FAMILY_STATUS': name_family_status,
            'NAME_HOUSING_TYPE': name_housing_type,
            'DAYS_BIRTH': days_birth,
            'DAYS_EMPLOYED': days_employed,
            'FLAG_PHONE': own_phone,
            'CNT_FAM_MEMBERS': cnt_fam_members,
            'CODE_GENDER': gender,
            'FLAG_OWN_CAR': own_car,
            'FLAG_OWN_REALTY': own_realty,
            'FLAG_WORK_PHONE': own_work_phone,
            'FLAG_EMAIL': own_email
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df.to_dict())

#Preprocessing

binary_features = ['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','FLAG_WORK_PHONE','FLAG_EMAIL']
continous_features = ['CNT_CHILDREN','AMT_INCOME_TOTAL','DAYS_BIRTH','DAYS_EMPLOYED','CNT_FAM_MEMBERS']
cat_features = ['NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE']

df['FLAG_EMAIL'] = df['FLAG_EMAIL'].replace({1:1,0:0})
df['FLAG_WORK_PHONE'] = df['FLAG_WORK_PHONE'].replace({1:1,0:0})
df['FLAG_PHONE'] = df['FLAG_WORK_PHONE'].replace({1:1,0:0})
df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].replace({'Y':1,'N':0})
df['CODE_GENDER'] = df['CODE_GENDER'].replace({'M':1,'F':0})
df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].replace({'Y':1,'N':0})




encoder = pickle.load(open('encoder.sav', 'rb'))
df['NAME_INCOME_TYPE'] = encoder.transform(df.NAME_INCOME_TYPE.values.reshape(-1, 1))

cat_features_req = ['NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE']
for i in cat_features_req:
    encoder_2 = pickle.load(open('encoder2_' + i +'.sav', 'rb'))
    df[i] = encoder_2.transform(df[i].values.reshape(-1, 1))



df = df.rename(columns={'FLAG_EMAIL': 'FLAG_EMAIL_1',
                        'FLAG_WORK_PHONE': 'FLAG_WORK_PHONE_1',
                        'CODE_GENDER': 'CODE_GENDER_M',
                        'FLAG_OWN_CAR': 'FLAG_OWN_CAR_Y',
                        'FLAG_OWN_REALTY': 'FLAG_OWN_REALTY_Y'})


for col in df.columns:
    if df[col].dtype != 'float64':
        df[col] = df[col].values.astype('float64')

st.subheader('Pre-processed Input to the Model')
st.table(df)
mms = pickle.load(open('minmax_scaler.sav', 'rb'))
mms.transform(df)



# Model Loading
clf =  pickle.load(open('finalized_model.sav', 'rb'))

#Model Inferencing

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(pd.DataFrame({'Labels': ['Approved','Declined']}))

st.subheader('Prediction')
if prediction == 0:
    st.write('Approved')
else:
    st.write('Declined')
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

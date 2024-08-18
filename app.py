import streamlit as st
import pandas as pd
import pickle

st.header('Jamboree Prediction Model with :blue[Linear Regression]', divider='rainbow')

col1, col2 = st.columns(2)


with col1:
   GRE_score = st.slider("What is your GRE score?", 0, 340, 315)

with col2:
   TOEFL_score = st.slider("What is your TOEFL score?", 0, 120, 100)

col1, col2 = st.columns(2)

with col1:
   University_Rating = st.selectbox('Select University Rating',[1,2,3,4,5])

with col2:
   SOP = st.selectbox('Select SOP Rating',[1,1.5,2,2.5,3,3.5,4,4.5,5])


col1, col2 = st.columns(2)

with col1:
   LOR = st.selectbox('Select LOR Rating',[1,1.5,2,2.5,3,3.5,4,4.5,5])

with col2:
   CGPA = number = st.number_input("Enter your CGPA", value=None, placeholder="Type a number between 0 to 10")

research = st.radio(
    "Do you have a research experience?",
    ["Yes", "No"],)

research_encode = {'Yes':1, 'No':0}


def tr_model(GRE_score, TOEFL_score, University_Rating, SOP, LOR, CGPA, research ):
   with open('Transform_model.pkl','rb') as file:
        transform_model = pickle.load(file)
        inputfeature = [[GRE_score, TOEFL_score, University_Rating, SOP, LOR, CGPA, research]]
        return transform_model.transform(inputfeature)

def reg_model(X):
   with open('LR_model.pkl','rb') as filename:
        lr_model = pickle.load(filename)
        return lr_model.predict(X)

def nn_model(X):
   with open('NN_model.pkl','rb') as filename:
        nn_model = pickle.load(filename)
        return nn_model.predict(X)

if st.button("Predict"):
    research = research_encode[research]
    X= tr_model(GRE_score, TOEFL_score, University_Rating, SOP, LOR, CGPA, research)
    #st.write(X)
    price=reg_model(X)
    st.write('Prediction with Linear Regression model')
    st.write(price)
    st.write('Prediction with Neural Network model')
    nnprice = nn_model(X)
    st.write(nnprice)

st.subheader(':blue[Data Insights]', divider='rainbow')
df=pd.read_csv('Jamboree_Admission.csv')
st.write(df)



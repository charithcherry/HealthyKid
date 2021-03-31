import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.pyplot as plt
#import lux
import time
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns


def BloodGroupTypes(data):
    data.loc[(data['Blood Group'] == 'A+')|(data['Blood Group'] == 'A'),'Blood Group']='A+ve'
    data.loc[(data['Blood Group'] == 'B+')|(data['Blood Group'] == 'B'),'Blood Group']='B+ve'
    data.loc[(data['Blood Group'] == 'O+')|(data['Blood Group'] == 'O'),'Blood Group']='O+ve'
    data.loc[(data['Blood Group'] == 'AB+')|(data['Blood Group'] == 'AB'),'Blood Group']='AB+ve'
    data.loc[(data['Blood Group'] == 'A-'),'Blood Group']='A-ve'
    data.loc[(data['Blood Group'] == 'B-'),'Blood Group']='B-ve'
    data.loc[(data['Blood Group'] == 'O-'),'Blood Group']='O-ve'
    data.loc[(data['Blood Group'] == 'AB-'),'Blood Group']='AB-ve'
    return data

def data_cleaning(dataset):
    st.write("Initial size of dataset {}".format(dataset.shape))
    err= dataset[ data['Blood Group'] == '#REF!' ].index
    dataset.drop(err, inplace=True)
    st.write("Dataset Size after removing err records {}".format(dataset.shape))
    errVAl= data[ data['Age in yrs'] == '#VALUE!' ].index
    dataset.drop(errVAl, inplace=True)
    st.write("Dataset Size after removing err value records {}".format(dataset.shape))    
    invalid_r= dataset[(dataset['Blood Group'].isnull()) & (dataset['Sex'].isnull())& (dataset['Height'].isnull())&(dataset['Weight'].isnull())& (dataset['BMI'].isnull())].index
    dataset.drop(invalid_r, inplace=True)
    st.write("Dataset Size after removing invalid records {}".format(dataset.shape))
    
def data_conversion(dataset):
    dataset=BloodGroupTypes(dataset)
    st.write("Blood Group Types available {}".format(data['Blood Group'].unique()))
    data['Age in yrs'] = data['Age in yrs'].apply(pd.to_numeric)
    data['Weight'] = data['Weight'].str.rstrip('kg').apply(pd.to_numeric)
    data['Height'] = data['Height'].str.rstrip('cm').apply(pd.to_numeric)
    data['Temperature'] = data['Temperature'].str.rstrip('F').apply(pd.to_numeric)
    data['Pulse'] = data['Pulse'].str.rstrip('per Min').apply(pd.to_numeric)
    data['BP'] = data['BP'].str.rstrip('mmHg')
    

def BP_manipulation(data):
    data.loc[(data['BP'].isnull()) & (data['Weight'].between(20,42,inclusive=True))& (data['Age in yrs'] <= 10),'BP']='96/117'
    data.loc[(data['BP'].isnull()) & (data['Weight'].between(20,42,inclusive=True))& (data['Age in yrs'] == 11),'BP']='98/119'
    data.loc[(data['BP'].isnull()) & (data['Weight'].between(20,42,inclusive=True))& (data['Age in yrs'] == 12),'BP']='100/121'
    data.loc[(data['BP'].isnull()) & (data['Weight']>=50) & (data['Age in yrs'] >= 13),'BP']='102/124'
    
    

def column_gen(data):
    new=data['BP'].str.split('/',n=1,expand=True)
    data['systolic']=new[0]
    data['diastolic']=new[1]
    new=data['Left_Eye_Power'].str.split('/',n=1,expand=True)
    data['LEP0']=new[0]
    data['LEP1']=new[1]
    new=data['Right_Eye_Power'].str.split('/',n=1,expand=True)
    data['REP0']=new[0]
    data['REP1']=new[1]
    new=data['Left_Eye_Pwr_WthGlass'].str.split('/',n=1,expand=True)
    data['LEPG0']=new[0]
    data['LEPG1']=new[1]
    new=data['Right_Eye_Pwr_WthGlass'].str.split('/',n=1,expand=True)
    data['REPG0']=new[0]
    data['REPG1']=new[1]
    

def load_data(url):
    try:
        data = pd.read_csv(url)
        return data
    except FileNotFoundError:
        st.error('Please enter Filename or Invalid Filename') 
        return -1 
    


@st.cache(persist=True)
def cache(data):
    return data
       
def load_start():
     for percent_complete in range(100):
         time.sleep(0.01)
         my_bar.progress(percent_complete + 1)
def load_exit():
             my_bar.progress(0)



#initialization

st.markdown("# HealthyKid")
st.markdown("### Data Analysis")

url = st.text_input('Enter a file path:')
st.markdown("Data Loading")
my_bar = st.progress(0)
load_start()
data = load_data(url)
load_exit()
st.markdown("Data Cleaning")
my_bar = st.progress(0)
load_start()
data_cleaning(data)
load_exit()
st.markdown("Data Conversion")
my_bar = st.progress(0)
data_conversion(data)
load_exit()
st.markdown("Column Generation")
my_bar = st.progress(0)
load_start()
column_gen(data)
load_exit()
data=cache(data)
st.dataframe(data)














#sidebar
st.sidebar.title("Visualizations")



#Number of Students based on School IDs and Gender
st.sidebar.markdown("###  Number of Students based on School IDs and Gender")
select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
school_count = data[' School ID'].value_counts()
school_count = pd.DataFrame({'School ID':school_count.index, 'Number of Students':school_count.values})
if not st.sidebar.checkbox("Hide", True,key=1):
    st.markdown("### Number of Students based on School IDs and Gender")
    if select == 'Bar plot':
        fig=px.histogram(data, x=' School ID', color="Sex", barmode='group')
        st.plotly_chart(fig)
    else:
        fig = px.pie(school_count, values='Number of Students', names='School ID')
        st.plotly_chart(fig)


#NULL BP values        
#st.sidebar.markdown("### Replacing NULL BP values")
#st.sidebar.button('BP manipulation')
#st.dataframe(data)

























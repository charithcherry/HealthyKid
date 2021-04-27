import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
#import lux
import time
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *


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
    errVAl= data[ data['BMI']== '#VALUE!' ].index
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
    data['BMI'] = data['BMI'].apply(pd.to_numeric)

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

@st.cache(persist=True,allow_output_mutation=True)
def cache(data):
    return data
       
def load_start():
     for percent_complete in range(100):
         time.sleep(0.001)
         my_bar.progress(percent_complete + 1)
def load_exit():
             my_bar.progress(0)

st.markdown("# HealthyKid")
st.markdown("### Data Analysis")

def data_upload(n):
    df = pd.DataFrame()
    data_file = st.file_uploader("Upload CSV",type=['csv'],key=n)
    if data_file is not None:
        file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
        st.write(file_details)
        df = pd.read_csv(data_file)
        st.dataframe(df)
    return df

def bp_con(cols):
    age=cols[0]
    sex=cols[1]
    sys=cols[2]
    dia=cols[3] if cols[3]!='' else 0
    if sys==0 or dia==0:
        return "NA"
    sys=int(sys)
    dia=int(dia)
    x1=bp_data[(bp_data['Gender']==sex) & (bp_data['Years']==age)]['Systolic']
    x2=bp_data[(bp_data['Gender']==sex) & (bp_data['Years']==age)]['Diastolic']
    x1 = int(''.join(map(str, x1)) if ''.join(map(str, x1))!='' else 0 )
    x2 = int(''.join(map(str, x2)) if ''.join(map(str, x2))!='' else 0)

    if (sys<=x1 and sys>(x1-10)) and (dia<=x2 and dia>(x2-5)):
        return "Normal"
    elif(sys<(x1-10) or dia<(x2-5)):
        return "Subnormal"
    elif(sys>x1 or dia>x2):
        return "Abnormal"
    else:
        return "NA"

def toint(data):
    return int(data[0])




st.markdown("Data Loading")
my_bar = st.progress(0)
load_start()
data = data_upload(1)
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

#count plot for students based on Age
st.sidebar.markdown("###  Number of Students based on Age")
if not st.sidebar.checkbox("Hide", True,key=2):
    st.markdown("### Count plot for Age")
    fig=px.histogram(data, x='Age in yrs', color=" School ID", barmode='group')
    st.plotly_chart(fig)

#count plot for blood group
st.sidebar.markdown("###  Count plot for Blood group")
select1 = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='10')
blood_count = data['Blood Group'].value_counts()
blood_count = pd.DataFrame({'Blood Group':blood_count.index, 'Number of Students':blood_count.values})
if not st.sidebar.checkbox("Hide", True,key=3):
    st.markdown("### Count plot for Blood group")
    if select1 == 'Bar plot':
        select = st.sidebar.selectbox('Visualization based on', [' School ID','Sex','Age in yrs'], key='2')
        fig=px.histogram(data, x='Blood Group', color=select, barmode='group')
        st.plotly_chart(fig)
    else:
        fig = px.pie(blood_count, values='Number of Students', names='Blood Group' )
        st.plotly_chart(fig)

#Count plots for Teeth related Issues
st.sidebar.markdown("###  General Count plots for Teeth related Issues ")
select1 = st.sidebar.selectbox('Visualization with respect to',[' School ID','Age in yrs'],key='8')
select = st.sidebar.selectbox('Visualization based on', ['Caries','Discoloration','Healthy_Gums','Malocclusion','Oral_Hygine','TeethWellFormed','Serious_Dental_Issue','Dentist_Recommendation'], key='3')
m_data = data.fillna('NA')
if not st.sidebar.checkbox("Hide", True,key=4):
    st.markdown("###  General Count plots for Teeth related Issues ")
    fig=px.histogram(m_data, x=select1, color=select, barmode='group')
    st.plotly_chart(fig)

#Count plots for ENT related Issues
st.sidebar.markdown("###  General Count plots for ENT related Issues  ")
select1 = st.sidebar.selectbox('Visualization with respect to',[' School ID','Age in yrs'],key='9')
select = st.sidebar.selectbox('Visualization based on', ['LEFT_EAR','RIGHT_EAR','ENT_Issue','Eye_Issue','ENT_Issue_Detail','Eye_Issue_Detail','Wears_Glass'], key='4')
if not st.sidebar.checkbox("Hide", True,key=5):
    st.markdown("###  General Count plots for ENT related Issues ")
    fig=px.histogram(m_data, x=select1, color=select, barmode='group')
    st.plotly_chart(fig)

#Count plots for eye acuity
st.sidebar.markdown("###  General Count plots for acuity ")
select1 = st.sidebar.selectbox('Visualization with respect to',[' School ID','Sex'],key='5')
select = st.sidebar.selectbox('Visualization based on', ['Left_Eye_Power','Right_Eye_Power','Left_Eye_Pwr_WthGlass','Right_Eye_Pwr_WthGlass'], key='6')
if not st.sidebar.checkbox("Hide", True,key=6):
    st.markdown("###  General Count plots for acuity")
    fig=px.histogram(m_data, x=select1, color=select, barmode='group')
    st.plotly_chart(fig)

st.sidebar.markdown("###  Dataframe representing overall recommendations ")
select = st.sidebar.selectbox('Visualization based on', ['Overall Summary','Recommendation'], key='7')
count = data[select].value_counts()
count = pd.DataFrame({select:count.index, 'Number of {}'.format(select):count.values})
if not st.sidebar.checkbox("Hide", True,key=7):
    st.markdown("###  Dataframe representing overall recommendations")
    st.dataframe(count)

#heatmap to show the correlation between age,bmi,height and weight
st.sidebar.markdown("### correlation and heatmaps")
if not st.sidebar.checkbox("Hide",True,key=8):
    st.markdown("### Heatmap to show the correlation between age,bmi,height and weight")
    bmi_correl = data[['Height','Weight','BMI','Age in yrs']].corr(method='pearson')
    xdf=['Height','Weight','BMI','Age in yrs']
    ydf=['Age in yrs','BMI','Weight','Height']
    zdf = np.array(bmi_correl)
    trace = go.Heatmap(x=xdf,y=ydf,z=zdf,type='heatmap',colorscale='GnBu')
    data1 = [trace]
    fig=go.Figure(data=data1)
    st.plotly_chart(fig)



#Pairplots to show the correlation between age,bmi,height,temperature,Pulse and weight based on Sex
st.sidebar.markdown("### Correlation and Pairplots ")
if not st.sidebar.checkbox("Hide", True,key=9):
    st.markdown("### Pairplots to show the correlation between age,bmi,Pulse,temperature,height and weight based on Sex")
    fig = px.scatter_matrix(data,
    dimensions=['LEP0', 'LEP1', 'REP0', 'REP1', 'LEPG0',
       'LEPG1', 'REPG0', 'REPG1'],
    color="Sex")
    st.plotly_chart(fig)

st.sidebar.markdown("### Enter the BP dataset ")
if not st.sidebar.checkbox("Hide", True,key=10):
    st.markdown("Upload BP Condition Dataset")
    bp_data=data_upload(2)
    data['systolic'] = data['systolic'].replace('0', np.nan)
    data['diastolic']=data['diastolic'].replace('0',np.nan)
    data[['systolic', 'diastolic']] = data[['systolic','diastolic']].fillna(value=0)
    data['systolic'] = pd.to_numeric(data['systolic'], errors='coerce')
    data['diastolic'] = pd.to_numeric(data['diastolic'], errors='coerce')
    data['systolic'] = data['systolic'].astype(int)
    data['diastolic']=data['diastolic'].astype(int)
    data['bp_condition']=data[['Age in yrs','Sex','systolic','diastolic']].apply(bp_con,axis=1)
    st.dataframe(data[['UHID','Sex','Age in yrs','BP','bp_condition']])

#NULL BP values        
#st.sidebar.markdown("### Replacing NULL BP values")
#st.sidebar.button('BP manipulation')
#st.dataframe(data)
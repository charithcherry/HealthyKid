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
         time.sleep(0.0000001)
         my_bar.progress(percent_complete + 1)
def load_exit():
             my_bar.progress(0)

st.markdown("# HealthyKid")
st.markdown("### Data Analysis")

def data_upload(n):
    df = pd.DataFrame()
    data_file = st.file_uploader("Upload only CSV",type=['csv'],key=n)
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
    if(x1==0 or x2==0):
        return "NA"
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

def height_con(cols):
    age=cols[0]
    sex=cols[1]
    height=float(cols[2])
    
    h1=data2[(data2['Gender']==sex) & (data2['Age']==age)]['H1']
    h2=data2[(data2['Gender']==sex) & (data2['Age']==age)]['H2']
    h3=data2[(data2['Gender']==sex) & (data2['Age']==age)]['H3']
    h1 = float(''.join(map(str, h1)) if ''.join(map(str, h1))!='' else 0 )
    h2 = float(''.join(map(str, h2)) if ''.join(map(str, h2))!='' else 0 )
    h3 = float(''.join(map(str, h3)) if ''.join(map(str, h3))!='' else 0 )
    if(h1==0 or h2==0 or h3==0):
        return "NA"  
    if (height<=h1):
        return "Stunted"
    elif(height>h1 and height<h2):
        return "Borderline"
    elif(height>h2 and height<h3):
        return "Normal"
    elif(height>h3):
        return "Over-height"
    else:
        return "NA"


def weight_con(cols):
    age=cols[0]
    sex=cols[1]
    weight=cols[2]
   
    w1=data3[(data3['Gender']==sex) & (data3['Age']==age)]['W1']
    w2=data3[(data3['Gender']==sex) & (data3['Age']==age)]['W2']
    w3=data3[(data3['Gender']==sex) & (data3['Age']==age)]['W3']
    w1 = float(''.join(map(str, w1)) if ''.join(map(str, w1))!='' else 0 )
    w2 = float(''.join(map(str, w2)) if ''.join(map(str, w2))!='' else 0 )
    w3 = float(''.join(map(str, w3)) if ''.join(map(str, w3))!='' else 0 )
    if(w1==0 or w2==0 or w3==0):
        return "NA"  
    if (weight<=w1):
        return "Under-weight"
    elif(weight>w1 and weight<w2):
        return "Borderline"
    elif(weight>w2 and weight<w3):
        return "Normal"
    elif(weight>w3):
        return "Over-weight"
    else:
        return "NA"


def bmi_con(cols):
    age=cols[0]
    sex=cols[1]
    BMI=cols[2]
   
    b1=data4[(data4['Gender']==sex) & (data4['Age']==age)]['Bmi1']
    b2=data4[(data4['Gender']==sex) & (data4['Age']==age)]['Bmi2']
    b3=data4[(data4['Gender']==sex) & (data4['Age']==age)]['Bmi3']
    b4=data4[(data4['Gender']==sex) & (data4['Age']==age)]['Bmi4']
    b1 = float(''.join(map(str, b1)) if ''.join(map(str, b1))!='' else 0 )
    b2 = float(''.join(map(str, b2)) if ''.join(map(str, b2))!='' else 0 )
    b3 = float(''.join(map(str, b3)) if ''.join(map(str, b3))!='' else 0 )
    b4 = float(''.join(map(str, b4)) if ''.join(map(str, b4))!='' else 0 ) 
    if(b1==0 or b2==0 or b3==0 or b4==0):
        return "NA"  
    if (BMI<=b1):
        return "Under-weight"
    elif(BMI>b1 and BMI<b2):
        return "Borderline"
    elif(BMI>b2 and BMI<b3):
        return "Normal"
    elif(BMI>b3 and BMI<b4):
        return "Over-weight"
    elif(BMI>b4):
        return "Obese"
    else:
        return "NA"

def eye_con(cols):
    rep=cols[0]
    lep=cols[1]
    if(rep == 0 or lep == 0):
        return "NA"  
    if (rep =='6/6'and lep =='6/6'):
        return "Normal"
    elif(rep =='6/12'or lep =='6/12' or rep =='6/12'or lep =='6/12' or rep =='6/9'or lep =='6/9'):
        return "Minor"
    elif(rep =='6/18'or lep =='6/18' or rep =='6/24'or lep =='6/24' or rep =='6/36'or lep =='6/36'or lep=='6/60' or rep=='6/60'):
        return "Major"
    else:
        return "NA"

def teeth_con(cols):
    caries=cols[0]
    discoloration=cols[1]
    healthygums=cols[2]
    malocclusion=cols[3]
    oralhygiene=cols[4]
    teethwellformed=cols[5]
    
    if(caries==0 and discoloration==0 and healthygums==0 and malocclusion==0 and oralhygiene == 0 and teethwellformed==0):
        return "NA"  
    if (caries=='No' and discoloration=='No' and healthygums=='Yes' and malocclusion=='No' and (oralhygiene == 'Good' or oralhygiene=='Fair') and teethwellformed=='Yes' ):
        return "Normal"
    elif(caries=='Yes' and (discoloration=='No' or discoloration=='Yes') and (healthygums=='Yes' or healthygums=='No') and (malocclusion=='No' or malocclusion == 'Yes') and (oralhygiene == 'Good' or oralhygiene=='Fair' or oralhygiene == 'Poor') and (teethwellformed=='Yes' or teethwellformed=='No')):
        return "Abnormal"
    else:
        return "NA"

def ent_con(cols):
    ent=cols[0]
 
    if(ent==0):
        return "NA"  
    if (ent=='No'):
        return "Normal"
    elif(ent=='Yes'):
        return "Abnormal"
    else:
        return "NA"

def health_con(cols):
    eye=cols[0]
    teeth=cols[1]
    ent=cols[2]
    
    if(eye==0 and teeth==0 and ent==0):
        return "NA"  
    if (eye=='Normal' and teeth=='Normal' and ent=='Normal'):
        return "Normal"
    elif((eye=='Minor' or eye=='Normal') and (teeth=='Normal' or teeth=='Abnormal') and (ent=='Normal' or ent=='Abnormal')):
        return "Minor"
    elif(eye=='Major' and (teeth=='Normal' or teeth=='Abnormal') and (ent=='Normal' or ent=='Abnormal')):
        return "Major"
    else:
        return "NA"

try:
    st.markdown("Data Loading")
    my_bar = st.progress(0)
    load_start()
    st.markdown("Enter the main dataset")
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
except:
    st.markdown("----------------------------------------")
    st.markdown("### Thank you for using our App")
    st.markdown("----------------------------------------")


#sidebar
st.sidebar.title("Visualizations")

#Number of Students based on School IDs and Gender
st.sidebar.markdown("###  Number of Students based on School IDs and Gender")
if not st.sidebar.checkbox("Hide", True,key=1):
    select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
    school_count = data[' School ID'].value_counts()
    school_count = pd.DataFrame({'School ID':school_count.index, 'Number of Students':school_count.values})
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

st.sidebar.markdown("-----------------------------------------------")

#count plot for blood group
st.sidebar.markdown("###  Count plot for Blood group")
if not st.sidebar.checkbox("Hide", True,key=3):
    select1 = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='10')
    blood_count = data['Blood Group'].value_counts()
    blood_count = pd.DataFrame({'Blood Group':blood_count.index, 'Number of Students':blood_count.values})
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
if not st.sidebar.checkbox("Hide", True,key=4):
    selectm = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='58')
    select = st.sidebar.selectbox('Visualization based on', ['Caries','Discoloration','Healthy_Gums','Malocclusion','Oral_Hygine','TeethWellFormed','Serious_Dental_Issue','Dentist_Recommendation'], key='3')
    school_count = data[select].value_counts()
    school_count = pd.DataFrame({select:school_count.index, 'Number of Students':school_count.values})
    m_data = data.fillna('NA')
    m_data1=m_data[m_data[select]!='NA']
    if selectm == 'Bar plot':
        select1 = st.sidebar.selectbox('Visualization with respect to',[' School ID','Age in yrs'],key='8')
        st.markdown("###  General Count plots for Teeth related Issues ")
        fig=px.histogram(m_data1, x=select1, color=select, barmode='group')
        st.plotly_chart(fig)
    else:
        fig = px.pie(school_count, values='Number of Students', names=select )
        st.plotly_chart(fig)


#Count plots for ENT related Issues
st.sidebar.markdown("###  General Count plots for ENT related Issues  ")
if not st.sidebar.checkbox("Hide", True,key=5):
    selectm = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='59')
    select = st.sidebar.selectbox('Visualization based on', ['LEFT_EAR','RIGHT_EAR','ENT_Issue','Eye_Issue','ENT_Issue_Detail','Eye_Issue_Detail','Wears_Glass'], key='4')
    school_count = data[select].value_counts()
    school_count = pd.DataFrame({select:school_count.index, 'Number of Students':school_count.values})
    m_data = data.fillna('NA')
    m_data2=m_data[m_data[select]!='NA']
    if selectm == 'Bar plot':
        select1 = st.sidebar.selectbox('Visualization with respect to',[' School ID','Age in yrs'],key='9')
        st.markdown("###  General Count plots for ENT related Issues ")
        fig=px.histogram(m_data2, x=select1, color=select, barmode='group')
        st.plotly_chart(fig)
    else:
        fig = px.pie(school_count, values='Number of Students', names=select )
        st.plotly_chart(fig)    

#Count plots for eye acuity
st.sidebar.markdown("###  General Count plots for acuity ")
if not st.sidebar.checkbox("Hide", True,key=6):
    selectm = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='60')
    select = st.sidebar.selectbox('Visualization based on', ['Left_Eye_Power','Right_Eye_Power','Left_Eye_Pwr_WthGlass','Right_Eye_Pwr_WthGlass'], key='6')
    school_count = data[select].value_counts()
    school_count = pd.DataFrame({select:school_count.index, 'Number of Students':school_count.values})
    m_data = data.fillna('NA')
    m_data3=m_data[m_data[select]!='NA']
    if selectm == 'Bar plot':
        select1 = st.sidebar.selectbox('Visualization with respect to',[' School ID','Sex'],key='5')
        st.markdown("###  General Count plots for acuity")
        fig=px.histogram(m_data3, x=select1, color=select, barmode='group')
        st.plotly_chart(fig)
    else:
        fig = px.pie(school_count, values='Number of Students', names=select )
        st.plotly_chart(fig)   

st.sidebar.markdown("-----------------------------------------------")

st.sidebar.markdown("###  Dataframe representing overall recommendations ")
if not st.sidebar.checkbox("Hide", True,key=7):
    select = st.sidebar.selectbox('Visualization based on', ['Overall Summary','Recommendation'], key='7')
    count = data[select].value_counts()
    count = pd.DataFrame({select:count.index, 'Number of {}'.format(select):count.values})
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

st.sidebar.markdown("-----------------------------------------------")


st.sidebar.markdown("### Enter the different conditioned datasets ")    
st.sidebar.markdown("### Upload the BP Condition dataset ")
if not st.sidebar.checkbox("Hide", True,key=10):
    select = st.sidebar.selectbox('Visualization based on', ['Sex','Class','Blood Group',' School ID','height_condition','weight_condition', 'bmi_condition'], key='14')
    st.markdown("Upload BP Condition Dataset")
    bp_data=data_upload(5)
    data['systolic'] = data['systolic'].replace('0', np.nan)
    data['diastolic']=data['diastolic'].replace('0',np.nan)
    data[['systolic', 'diastolic']] = data[['systolic','diastolic']].fillna(value=0)
    data['systolic'] = pd.to_numeric(data['systolic'], errors='coerce')
    data['diastolic'] = pd.to_numeric(data['diastolic'], errors='coerce')
    data['systolic'] = data['systolic'].astype(int)
    data['diastolic']=data['diastolic'].astype(int)
    data['bp_condition']=data[['Age in yrs','Sex','systolic','diastolic']].apply(bp_con,axis=1)
    data_bp=data[data['bp_condition']!='NA']
    st.dataframe(data_bp[['UHID','Sex','Age in yrs','BP','bp_condition']])
    st.markdown("### Histogram based on bp ranges ")
    fig=px.histogram(data_bp, x=select, color='bp_condition', barmode='group')
    st.plotly_chart(fig)

st.sidebar.markdown("### Upload Height Condition dataset ")
if not st.sidebar.checkbox("Hide", True,key=11):
    select = st.sidebar.selectbox('Visualization based on', ['Sex','Class','Blood Group',' School ID','bp_condition','weight_condition', 'bmi_condition'], key='15')
    st.markdown("Upload Height Condition Dataset")
    data2 = data_upload(2)
    data['height_condition']=data[['Age in yrs','Sex','Height']].apply(height_con,axis=1)
    data_h=data[data['height_condition']!='NA']
    st.dataframe(data_h[['UHID','Sex','Age in yrs','Height','height_condition']])
    st.markdown("### Histogram based on Height ")
    fig=px.histogram(data_h, x=select, color='height_condition', barmode='group')
    st.plotly_chart(fig)

st.sidebar.markdown("### Upload Weight Condition dataset ")
if not st.sidebar.checkbox("Hide", True,key=12):
    select = st.sidebar.selectbox('Visualization based on', ['Sex','Class','Blood Group',' School ID','bp_condition','height_condition', 'bmi_condition'], key='16')
    st.markdown("Upload Weight Condition Dataset")
    data3 = data_upload(3)
    data['weight_condition']=data[['Age in yrs','Sex','Weight']].apply(weight_con,axis=1)
    data_w=data[data['weight_condition']!='NA']
    st.dataframe(data_w[['UHID','Sex','Age in yrs','Weight','weight_condition']])
    st.markdown("### Histogram based on weight ")
    fig=px.histogram(data_w, x=select, color='weight_condition', barmode='group')
    st.plotly_chart(fig)

st.sidebar.markdown("### Upload BMI Condition dataset ")
if not st.sidebar.checkbox("Hide", True,key=13):
    select = st.sidebar.selectbox('Visualization based on', ['Sex','Class','Blood Group',' School ID','bp_condition','weight_condition', 'height_condition'], key='17')
    st.markdown("Upload BMI Condition Dataset")
    data4 = data_upload(4)
    data['bmi_condition']=data[['Age in yrs','Sex','BMI']].apply(bmi_con,axis=1)
    data_b=data[data['bmi_condition']!='NA']
    st.dataframe(data_b[['UHID','Sex','Age in yrs','Weight','weight_condition']])
    st.markdown("### Histogram based on BMI ")
    fig=px.histogram(data_b, x=select, color='bmi_condition', barmode='group')
    st.plotly_chart(fig)



st.sidebar.markdown("-----------------------------------------------")


st.sidebar.markdown("### Data showing Students Details with Major,Minor and Normal Probelms ")  
st.sidebar.markdown("###  Details regarding Eye Condition")
if not st.sidebar.checkbox("Hide", True,key=18):
    selectm = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='61')
    data[['Right_Eye_Power','Left_Eye_Power']] = data[['Right_Eye_Power','Left_Eye_Power']].fillna(value=0)
    data['eye_condition']=data[['Right_Eye_Power','Left_Eye_Power']].apply(eye_con,axis=1)
    school_count = data['eye_condition'].value_counts()
    school_count = pd.DataFrame({'eye_condition':school_count.index, 'Number of Students':school_count.values})
    if selectm == 'Bar plot':
        select = st.sidebar.selectbox('Visualization based on', ['Sex','Class','Blood Group',' School ID'], key='19')
        st.markdown("### Histogram based on Eye Condition")
        fig=px.histogram(data, x=select, color='eye_condition', barmode='group')
        st.plotly_chart(fig)
        st.sidebar.markdown("###  Details regarding Eye Condition for specific School")
        if not st.sidebar.checkbox("Hide", True,key=26):
            select = st.sidebar.selectbox('Visualization based on', ['Sex','Class','Blood Group',' School ID'], key='27')
            select1 = st.sidebar.selectbox('Visualization with respect to',list(data[' School ID'].unique()),key='28')
            st.markdown("### Details regarding Eye condition with respect to specific data")
            d1=data[data[' School ID']==select1]
            fig=px.histogram(d1,x=select, color='eye_condition', barmode='group')
            st.plotly_chart(fig)
    else:
        fig = px.pie(school_count, values='Number of Students', names='eye_condition' )
        st.plotly_chart(fig)

st.sidebar.markdown("###  Details regarding Teeth Condition")
if not st.sidebar.checkbox("Hide", True,key=20):
    selectm = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='62')
    data[['Caries', 'Discoloration', 'Healthy_Gums', 'Malocclusion','Oral_Hygine', 'TeethWellFormed']] = data[['Caries', 'Discoloration', 'Healthy_Gums', 'Malocclusion','Oral_Hygine', 'TeethWellFormed']].fillna(value=0)
    data['Teeth_condition']=data[['Caries', 'Discoloration', 'Healthy_Gums', 'Malocclusion','Oral_Hygine', 'TeethWellFormed']].apply(teeth_con,axis=1)
    school_count = data['Teeth_condition'].value_counts()
    school_count = pd.DataFrame({'Teeth_condition':school_count.index, 'Number of Students':school_count.values})
    if selectm == 'Bar plot':
        select = st.sidebar.selectbox('Visualization based on', ['Sex','Class','Blood Group',' School ID'], key='21')
        st.markdown("### Histogram based on Teeth Condition")
        fig=px.histogram(data, x=select, color='Teeth_condition', barmode='group')
        st.plotly_chart(fig)
        st.sidebar.markdown("###  Details regarding Teeth Condition for specific School")
        if not st.sidebar.checkbox("Hide", True,key=29):
            select = st.sidebar.selectbox('Visualization based on', ['Sex','Class','Blood Group',' School ID'], key='30')
            select1 = st.sidebar.selectbox('Visualization with respect to',list(data[' School ID'].unique()),key='31')
            st.markdown("### Details regarding Teeth condition with respect to specific data")
            d2=data[data[' School ID']==select1]
            fig=px.histogram(d2,x=select, color='Teeth_condition', barmode='group')
            st.plotly_chart(fig)
    else:
        fig = px.pie(school_count, values='Number of Students', names='Teeth_condition' )
        st.plotly_chart(fig)

st.sidebar.markdown("###  Details regarding ENT Condition")
if not st.sidebar.checkbox("Hide", True,key=22):
    selectm = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='63')
    data[['ENT_Issue']] = data[['ENT_Issue']].fillna(value=0)
    data['ENT_condition']=data[['ENT_Issue']].apply(ent_con,axis=1)
    school_count = data['ENT_condition'].value_counts()
    school_count = pd.DataFrame({'ENT_condition':school_count.index, 'Number of Students':school_count.values})
    if selectm == 'Bar plot':
        select = st.sidebar.selectbox('Visualization based on', ['Sex','Class','Blood Group',' School ID'], key='23')
        st.markdown("### Histogram based on ENT Condition")
        fig=px.histogram(data, x=select, color='ENT_condition', barmode='group')
        st.plotly_chart(fig)
        st.sidebar.markdown("###  Details regarding ENT Condition for specific School")
        if not st.sidebar.checkbox("Hide", True,key=32):
            select = st.sidebar.selectbox('Visualization based on', ['Sex','Class','Blood Group',' School ID'], key='33')
            select1 = st.sidebar.selectbox('Visualization with respect to',list(data[' School ID'].unique()),key='34')
            st.markdown("### Details regarding ENT condition with respect to specific data")
            d3=data[data[' School ID']==select1]
            fig=px.histogram(d3,x=select, color='ENT_condition', barmode='group')
            st.plotly_chart(fig)
    else:
        fig = px.pie(school_count, values='Number of Students', names='ENT_condition' )
        st.plotly_chart(fig)

st.sidebar.markdown("###  Details regarding Total Health Condition")
if not st.sidebar.checkbox("Hide", True,key=24):
    data[['Right_Eye_Power','Left_Eye_Power']] = data[['Right_Eye_Power','Left_Eye_Power']].fillna(value=0)
    data['eye_condition']=data[['Right_Eye_Power','Left_Eye_Power']].apply(eye_con,axis=1)
    data[['Caries', 'Discoloration', 'Healthy_Gums', 'Malocclusion','Oral_Hygine', 'TeethWellFormed']] = data[['Caries', 'Discoloration', 'Healthy_Gums', 'Malocclusion','Oral_Hygine', 'TeethWellFormed']].fillna(value=0)
    data['Teeth_condition']=data[['Caries', 'Discoloration', 'Healthy_Gums', 'Malocclusion','Oral_Hygine', 'TeethWellFormed']].apply(teeth_con,axis=1)
    data[['ENT_Issue']] = data[['ENT_Issue']].fillna(value=0)
    data['ENT_condition']=data[['ENT_Issue']].apply(ent_con,axis=1)
    selectm = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='64')
    data['Health_condition']=data[['eye_condition', 'Teeth_condition','ENT_condition']].apply(health_con,axis=1)
    school_count = data['Health_condition'].value_counts()
    school_count = pd.DataFrame({'Health_condition':school_count.index, 'Number of Students':school_count.values})
    if selectm == 'Bar plot':
        select = st.sidebar.selectbox('Visualization based on', ['Sex','Class','Blood Group',' School ID','bp_condition','height_condition', 'bmi_condition'], key='25')
        st.markdown("### Histogram based on Overall Health Condition")
        fig=px.histogram(data, x=select, color='Health_condition', barmode='group')
        st.plotly_chart(fig)
        st.sidebar.markdown("###  Details regarding Health Condition for specific School")
        if not st.sidebar.checkbox("Hide", True,key=35):
            select = st.sidebar.selectbox('Visualization based on', ['Sex','Class','Blood Group',' School ID'], key='36')
            select1 = st.sidebar.selectbox('Visualization with respect to',list(data[' School ID'].unique()),key='37')
            st.markdown("### Details regarding Health condition with respect to specific data")
            d4=data[data[' School ID']==select1]
            fig=px.histogram(d4,x=select, color='Health_condition', barmode='group')
            st.plotly_chart(fig)
    else:
        fig = px.pie(school_count, values='Number of Students', names='Health_condition' )
        st.plotly_chart(fig)

st.sidebar.markdown("-----------------------------------------------")
st.sidebar.markdown("###  Pie Charts")

if not st.sidebar.checkbox("Hide", True,key=38):
    s1=st.sidebar.selectbox('Select the Condition',['bp_condition', 'height_condition','weight_condition', 'bmi_condition', 'eye_condition', 'Teeth_condition','ENT_condition', 'Health_condition'],key='39')
    s2=st.sidebar.selectbox('Select the School',['S0000001', 'S0000005 ', 'S0000003 ', 'S0000002'],key='40')
    s3=st.sidebar.selectbox('Select the Standard',['8', '10', '9', '7', '6', '5', '4', '3', '2', '1', 'UKg', 'LKg','Teacher', 'SchoolStaff', 'PlayHome', 'Nursery'],key='41')
    s4=st.sidebar.selectbox('Select the Gender',['Male', 'Female'],key='42')

    sbp=data[data[' School ID']==s2]
    cbp=sbp[sbp['Class']==s3]
    gbp=cbp[cbp['Sex']==s4]
    school_count = sbp[s1].value_counts()
    count = pd.DataFrame({'conditions':school_count.index, 'Number of Students':school_count.values})
    count = count.set_index("conditions")
    for x in count.keys():
        if x=='NA':
            count = count.drop('NA')
    count=count.reset_index()
    st.markdown("Pie chart for School "+ s2 +" based on "+s1)
    fig = px.pie(count, values='Number of Students', names='conditions')
    st.plotly_chart(fig)

    class_count = cbp[s1].value_counts()
    count1 = pd.DataFrame({'conditions':class_count.index, 'Number of Students':class_count.values})
    count1 = count1.set_index("conditions")
    for x in count1.keys():
        if x=='NA':
            count1 = count1.drop('NA')
    count1=count1.reset_index()
    st.markdown("Pie chart for class "+ s3 +" from "+s2+" based on "+s1)
    fig = px.pie(count1, values='Number of Students', names='conditions')
    st.plotly_chart(fig)


    class_count = gbp[s1].value_counts()
    count2 = pd.DataFrame({'conditions':class_count.index, 'Number of Students':class_count.values})
    count2 = count2.set_index("conditions")
    for x in count2.keys():
        if x=='NA':
            count2 = count2.drop('NA')
    count2=count2.reset_index()
    st.markdown("Pie chart for "+s4+" students in class "+ s3 +" from "+s2+" based on "+s1)
    fig = px.pie(count2, values='Number of Students', names='conditions')
    st.plotly_chart(fig)






































































































#NULL BP values        
#st.sidebar.markdown("### Replacing NULL BP values")
#st.sidebar.button('BP manipulation')
#st.dataframe(data)

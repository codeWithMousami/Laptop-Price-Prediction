import streamlit as st
import pickle
import pandas as pd
import  numpy as np

# Import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
#df = pickle.load(open('df.pkl', 'rb'))
df = pd.read_pickle('df.pkl')

st.title('Laptop Price Predictor')

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
laptop_type = st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM (inGB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['NO', 'YES'])

# IPS
ips = st.selectbox('IPS', ['NO', 'YES'])

# Screensize
screensize = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen Resolution', ['1920*1080', '1366*768', '1600*900', '3840*2160', '3200*1800', '2880*1800', '2560*1600', '2560*1440', '2304*1440'])

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# HDD
hdd = st.selectbox('HDD (IN GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD (IN GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# OS
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    #query
    ppi=None
    if  touchscreen=='Yes':
        touchscreen=1
    else:
        touchscreen=0

    if ips =='Yes':
        ips=1
    else:
        ips= 0
    X_res= int(resolution.split('*')[0])
    Y_res= int(resolution.split('*')[1])
    ppi=((X_res**2)+(Y_res**2))**0.5/screensize

    query=np.array([company,laptop_type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query=query.reshape(1,12)
    st.title("The predicted price of this configuration is : Rs. "+str(int(np.exp(pipe.predict(query)[0]))))

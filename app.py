# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:40:02 2021

@author: Aravind
"""

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from textblob import TextBlob
from wordcloud import WordCloud
import re
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as comp
import requests
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

from nltk.stem.snowball import SnowballStemmer
from pickle import load, dump

#sess = tf.Session()
#set_session(sess)

data=pd.read_csv("D:/Data Science/Project/ExcelR Project/Medicines Side Effect Analysis/new_data.csv",encoding = "utf-8")
st.markdown("<h1 style='text-align: center;'> <img src='https://placeit-assets0.s3-accelerate.amazonaws.com/custom-pages/landing-page-medical-logo-maker/Pharmacy-Logo-Maker-Red.png' alt='' width='120' height='120'</h1>", unsafe_allow_html=True)

data_review=pd.DataFrame(columns=['Reviews'],data=data)

st.title = '<p style="font-family:Imprint MT Shadow; text-align:center;background-color:#FF5721;border-radius: 0.4rem;  text-font:Bodoni MT Poster Compressed; color:Black; font-size: 60px;">Apna-MediCare</p>'
st.markdown(st.title,  unsafe_allow_html=True)

model_lr=pickle.load(open('D:\Data Science\Project\ExcelR Project\Medicines Side Effect Analysis/logisitc.pkl','rb'))
tfidf=pickle.load(open('D:\Data Science\Project\ExcelR Project\Medicines Side Effect Analysis/TfidfVectorizer.pkl','rb'))

#x=data['Reviews'].values.astype('U')
#y=data['Analysis']
#x=x.astype
#y=y.astype

#x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.20, random_state=42)

#vectorizer =TfidfVectorizer()

#model=Pipeline([('tfidf', TfidfVectorizer()),
                        #('logistic', LogisticRegression(max_iter=500)),
                        #])

# Feed the training data through the pipeline

#model.fit(x_train, y_train)

#prediction_log_test=model.predict(x_test)

#accuracy_score=accuracy_score(y_test,prediction_log_test )


activities=["Medicine Name","Condition","Clear"]
choice = st.sidebar.selectbox("Select Your Activity", activities)

def Average(lst):
           try:
              return sum(lst) / len(lst)
           except:
               pass

#if choice=="NONE":
 
if choice=="Medicine Name":
    
    #st.write("Top MostRecent Drugs")
    
    raw_text = st.text_area("Enter the Medicine Name")

      
    Analyzer_Choice = st.selectbox("Select the Activities", [" ","Show Related Drug Conditions"])
    if st.button("Analyzer"):
        if Analyzer_Choice =="Show Related Drug Conditions":
            #st.success("Fetching Top Conditions")
            data_top_condition=data[(data['Condition']=='Analysis') & (data['Drug']==str(raw_text))]
            data_top_condition=data[data['Drug']==raw_text] 
            data_top_condition=data_top_condition.groupby(['Drug','Condition']).agg('mean').reset_index()
            data_top_condition=data_top_condition.sort_values(by=['Condition'], ascending=False).head(5) 
            #data_top_condition=data_top_condition.head(5)
            data_top_condition_list=data_top_condition['Condition'].tolist()
            st.markdown("<h3><b>Condition :</b></h3>",unsafe_allow_html=True)
            for i in data_top_condition_list:
                st.markdown(i)
            
            

    Analyzer_Choice = st.selectbox("Reviews", [" ","Prediction","Side Effect","Rating Analysis"])
    if st.button("Reviews"):  
        if Analyzer_Choice =="Prediction":
            
            #st.success("Fetching Top Reviews")
            data_top_positive=data[(data['Analysis']=='Positive') & (data['Drug']==str(raw_text))]
            data_top_positive=data_top_positive
            data_top_positive_list=data_top_positive['Reviews'].tolist()
            st.markdown("<h3><b>Positive :</b></h3>",unsafe_allow_html=True)
            for i in data_top_positive_list:
               st.markdown(i)
           # st.markdown("<h3><b>Average Positive Review Rating :</b></h3>",unsafe_allow_html=True)
           # data_top_positive_list=data_top_positive['Rating'].tolist()
           # st.markdown(Average(data_top_positive_list))
            
              
            data_top_negative=data[(data['Analysis']=='Negative') & (data['Drug']==str(raw_text))]
            data_top_negative=data_top_negative
            data_top_negative_list=data_top_negative['Reviews'].tolist()
            st.markdown("<h3><b>Negative :</b></h3>",unsafe_allow_html=True)
            for i in data_top_negative_list:
                st.markdown(i)
             
           # st.markdown("<h3><b>Average Negative Review Rating :</b></h3>",unsafe_allow_html=True)
           # data_top_negative_list=data_top_negative['Rating'].tolist()
           # st.markdown(Average(data_top_negative_list))
                 
            data_top_neutral=data[(data['Analysis']=='Neutral') & (data['Drug']==str(raw_text))]
            data_top_neutral=data_top_neutral
            data_top_neutral_list=data_top_neutral['Reviews'].tolist()
            st.markdown("<h3><b>Neutral :</b></h3>",unsafe_allow_html=True)
            for i in data_top_neutral_list:
                st.markdown(i)
           # st.markdown("<h3><b>Average Neutral Review Rating :</b></h3>",unsafe_allow_html=True)
           # data_top_neutral_list=data_top_neutral['Rating'].tolist()
          #  st.markdown(Average(data_top_neutral_list))
            


                
        if Analyzer_Choice =="Side Effect":  
            
        #Side Effect
            Side_Effect=data[(data['Side_Effect']=='Analysis') & (data['Drug']==str(raw_text))]
            Side_Effect=data[data['Drug']==raw_text] 
            Side_Effect=Side_Effect.groupby(['Drug','Side_Effect']).agg('mean').reset_index()
            Side_Effect=Side_Effect.sort_values(by=['Side_Effect'], ascending=False)
            #data_top_condition=data_top_condition.head(5)
            Side_Effect_list=Side_Effect['Side_Effect'].tolist()
            st.markdown("<h3><b>Side Effect :</b></h3>",unsafe_allow_html=True)
            for i in Side_Effect_list:
                st.markdown(i)
                
        if Analyzer_Choice =="Rating Analysis":
            
        #Rating
            data_top_positive=data[(data['Analysis']=='Positive') & (data['Drug']==str(raw_text))]
            data_top_positive=data_top_positive
            data_top_positive_list=data_top_positive['Rating'].tolist()
            
            
            data_top_negative=data[(data['Analysis']=='Negative') & (data['Drug']==str(raw_text))]
            data_top_negative=data_top_negative
            data_top_negative_list=data_top_negative['Rating'].tolist()
            #st.markdown(Average(data_top_negative_list))
                 
            data_top_neutral=data[(data['Analysis']=='Neutral') & (data['Drug']==str(raw_text))]
            data_top_neutral=data_top_neutral
            data_top_neutral_list=data_top_neutral['Rating'].tolist()
            #st.markdown(Average(data_top_neutral_list))
            
            
            st.text("Below are the Observation plotted")
            
            rating={'avg_rat':[Average(data_top_positive_list),Average(data_top_negative_list),Average(data_top_neutral_list)],
                    'rat':['Positive','Negative','Neutral']}
            df_rating=pd.DataFrame(rating)
            #plt.bar(df_rating.avg_rat, df_rating.rat)
            st.bar_chart(df_rating['avg_rat'])
            
            st.text("0:Positive,1:Negative,2:Neutral")
            
            st.write("Total average rating=",df_rating['avg_rat'].mean())
            
         
                    
     
                                  
            
if choice=="Condition":
    
    #st.write("Top Most Condition")
    raw_text = st.text_area("Enter the Condition")
    
    Analyzer_Choice = st.selectbox("Select the Activities", [" ","Show Condition Related All Medicines","Show Condition Related Five Medicines"])
    
    if st.button("Analyzer"):
        if Analyzer_Choice =="Show Condition Related All Medicines":
            data_top_Drug=data[(data['Drug']=='Analysis') & (data['Condition']==str(raw_text))]
            data_top_Drug=data[data['Condition']==raw_text]
            data_top_Drug=data_top_Drug.groupby(['Condition','Drug']).agg('mean').reset_index()
            data_top_Drug=data_top_Drug.sort_values(by=['Drug'], ascending=True)
            data_top_Drug_list=data_top_Drug['Drug'].tolist()
            st.markdown("<h3><b>Medicines :</b></h3>",unsafe_allow_html=True)
            for i in data_top_Drug_list:
                st.markdown(i)
       
        if Analyzer_Choice =="Show Condition Related Five Medicines":
            data_top_Drug=data[(data['Drug']=='Analysis') & (data['Condition']==str(raw_text))]
            data_top_Drug=data[data['Condition']==raw_text]
            data_top_Drug=data_top_Drug.groupby(['Condition','Drug']).agg('mean').reset_index()
            data_top_Drug=data_top_Drug.sort_values(by=['Drug'], ascending=True).head(5)
            data_top_Drug_list=data_top_Drug['Drug'].tolist()
            st.markdown("<h3><b>Medicines :</b></h3>",unsafe_allow_html=True)
            for i in data_top_Drug_list:
                st.markdown(i)
        
        
            
  
    Analyzer_Choice = st.selectbox("Reviews", [" ","Prediction","Side Effect","Rating Analysis"])
    if st.button("Reviews"):  
        if Analyzer_Choice =="Prediction":
    
   #  if st.button("Reviews"):   
       # if Analyzer_Choice =="Show Top Reviews":
            
            #st.success("Fetching Top Reviews")
            data_top_positive=data[(data['Analysis']=='Positive') & (data['Condition']==str(raw_text))]
            data_top_positive=data_top_positive.head(5)
            data_top_positive_list=data_top_positive['Reviews'].tolist()
            st.markdown("<h3><b>Positive :</b></h3>",unsafe_allow_html=True)
            for i in data_top_positive_list:
               st.markdown(i)
           # st.markdown("<h3><b>Average Positive Review Rating :</b></h3>",unsafe_allow_html=True)
           # data_top_positive_list=data_top_positive['Rating'].tolist()
           # st.markdown(Average(data_top_positive_list))
            
              
            data_top_negative=data[(data['Analysis']=='Negative') & (data['Condition']==str(raw_text))]
            data_top_negative=data_top_negative.head(5)
            data_top_negative_list=data_top_negative['Reviews'].tolist()
            st.markdown("<h3><b>Negative :</b></h3>",unsafe_allow_html=True)
            for i in data_top_negative_list:
                st.markdown(i)
          #  st.markdown("<h3><b>Average Negative Review Rating :</b></h3>",unsafe_allow_html=True)
           # data_top_negative_list=data_top_negative['Rating'].tolist()
          #  st.markdown(Average(data_top_negative_list))
                 
            data_top_neutral=data[(data['Analysis']=='Neutral') & (data['Condition']==str(raw_text))]
            data_top_neutral=data_top_neutral.head(5)
            data_top_neutral_list=data_top_neutral['Reviews'].tolist()
            st.markdown("<h3><b>Neutral :</b></h3>",unsafe_allow_html=True)
            for i in data_top_neutral_list:
                st.markdown(i)
           # st.markdown("<h3><b>Average Neutral Review Rating :</b></h3>",unsafe_allow_html=True)
           # data_top_neutral_list=data_top_neutral['Rating'].tolist()
          #  st.markdown(Average(data_top_neutral_list))
            
        if Analyzer_Choice =="Side Effect":  
        #Side Effect
            Side_Effect=data[(data['Side_Effect']=='Analysis') & (data['Condition']==str(raw_text))]
            Side_Effect=data[data['Condition']==raw_text] 
            Side_Effect=Side_Effect.groupby(['Condition','Side_Effect']).agg('mean').reset_index()
            Side_Effect=Side_Effect.sort_values(by=['Side_Effect'], ascending=False)
            Side_Effect=Side_Effect.head(5)
            Side_Effect_list=Side_Effect['Side_Effect'].tolist()
            st.markdown("<h3><b>Side Effect :</b></h3>",unsafe_allow_html=True)
            for i in Side_Effect_list:
                st.markdown(i)
                
        if Analyzer_Choice =="Rating Analysis":
            
        #Rating
            data_top_positive=data[(data['Analysis']=='Positive') & (data['Condition']==str(raw_text))]
            data_top_positive=data_top_positive
            data_top_positive_list=data_top_positive['Rating'].tolist()
            
            
            data_top_negative=data[(data['Analysis']=='Negative') & (data['Condition']==str(raw_text))]
            data_top_negative=data_top_negative
            data_top_negative_list=data_top_negative['Rating'].tolist()
            #st.markdown(Average(data_top_negative_list))
                 
            data_top_neutral=data[(data['Analysis']=='Neutral') & (data['Condition']==str(raw_text))]
            data_top_neutral=data_top_neutral
            data_top_neutral_list=data_top_neutral['Rating'].tolist()
            #st.markdown(Average(data_top_neutral_list))
            
            
            st.text("Below are the Observation plotted")
            
            rating={'avg_rat':[Average(data_top_positive_list),Average(data_top_negative_list),Average(data_top_neutral_list)],
                    'rat':['Positive','Negative','Neutral']}
            df_rating=pd.DataFrame(rating)
            #plt.bar(df_rating.avg_rat, df_rating.rat)
            st.bar_chart(df_rating['avg_rat'])
            
            st.text("0:Positive,1:Negative,2:Neutral")
            
            st.write("Total average rating=",df_rating['avg_rat'].mean())    
            
            
#Background color
          
#page_bg_img = '''
#<style>
#body {
#background-image: url("https://wallpapercave.com/download/aesthetic-doctor-girl-cartoon-wallpapers-wp7425372?nocache=1");
#background-size: cover;
#}
#</style>
#'''

#st.markdown(page_bg_img, unsafe_allow_html=True)





    

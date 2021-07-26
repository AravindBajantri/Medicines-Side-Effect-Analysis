#!/usr/bin/env python
# coding: utf-8

# #                                    Medicines Side Effect Analysis
# 
# 
# 
# 
#    ### Business objective:
#    
# This product could be helpful for companies like 1mg to provide detailed rating of the side effects of the product over their site. It could also be helpful for the patients who are buying drugs online to check the side effects of the drugs before buying it.
#                  
#                  
#                  
#                  
#   ### Data Set Details:         
#   
#   The dataset contains patient reviews on specific medicines and with related conditions and a their satisfaction rate.
#   
#  #### Number of Rows: 362806
#  
#  #### Number of Columns: 12
#   
#   
#   
#  #### Column Names
#                  
# 1.	Drug (categorical): name of drug
# 2.	DrugId (numerical): drug id
# 3.	Condition (categorical): name of condition
# 4.	Review (text): patient review
# 5.	Side (text): side effects associated with drug (if any)
# 6.	EaseOfUse (numerical): star rating
# 7.	Effectiveness (numerical): star rating
# 8.	Satisfaction (numerical): star rating
# 9.	Date (date): date of review entry
# 10.	UsefulCount (numerical): number of users who found review useful.
# 11.	Age (numerical): age group range of user
# 12.	Sex (categorical): gender of user

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re  # For regular expressions
import string # For handling string
import math # For performing mathematical operations
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import spacy
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')


# In[2]:


mse_dataset=pd.read_csv("webmd.csv", parse_dates=["Date"])
mse_dataset.head()


# ### Rearrange column names

# In[80]:


mse_dataset.columns


# In[4]:


mse=mse_dataset[['Drug','DrugId','Condition','Reviews','Sides','EaseofUse','Effectiveness','Satisfaction','Date','UsefulCount','Age','Sex']]
mse.head()


#  # Exploratory Data Analysis

#  ## 1) Text Data Pre-Processing

#  #### Summary of EDA Process

# In[8]:


import sweetviz as sv
sweet_report=sv.analyze(mse_df)
sweet_report.show_html("MSE_EDA_Report.html")


# In[9]:


# Rename the Side and Gender Columns
mse_df=mse.rename({'Sides':'Side_Effect','Sex':'Gender'}, axis=1)

mse_df.head()


# #### Data Shape
# 

# In[10]:


mse_df.shape


# #### Data Type

# In[11]:


mse_df.info()


# #### Missing values

# In[12]:


mse_df=mse_df.replace(r'^\s*$', np.NaN, regex=True)
print(mse_df.isnull().sum())


# In[13]:


plt.figure(figsize=(20,8))
sns.heatmap(mse_df.isnull(),yticklabels=False,cmap='viridis')


# In[14]:


#import missingno as msno

#msno.matrix(mse_df,figsize=(20,10))


# #### Percentage of Missing Data

# In[15]:


percent_missing = mse_df.isnull().sum() * 100 / len(mse_df)
missing_value_df = pd.DataFrame({'column_name': mse_df.columns,
                                 'Percentage of Missing Data': percent_missing})


# In[16]:


missing_value_df


# In[17]:


missing_value_df.index


# In[18]:


missing_value_df.reset_index(inplace=True)


# In[19]:


missing_value_df=missing_value_df.drop('index', axis=1)


# In[20]:


missing_value_df1=missing_value_df.drop(labels=[0,1,5,6,7,8,9], axis=0).round(2)
missing_value_df1


# In[21]:


missing_value_df1.astype({"Percentage of Missing Data":'int'})


# In[22]:


import matplotlib.pyplot as plt
plt.figure(figsize = (20, 12))

sizes = [43, 41883, 17460, 12202, 26537,264681]
labels = 'Condition', 'Reviews', 'Side Effect', 'Age','Gender','Total Datapoint'
 
colors = ( "blue", "grey", "green","orange", 
          "red") 
fig1, ax1 = plt.subplots()
explode = (0.09,  0,  0.0, 0,  0, 0) 
ax1.pie(sizes, colors = colors, explode=explode, labels=labels,autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.title('Missing Data Distribution')
plt.show()


# #### Droping the Null Values

# In[23]:


mse_df1=mse_df.dropna()


# In[24]:


len(mse_df1)


#  #### Duplicates Datapoint

# In[25]:


mse_df1[mse_df1.duplicated()].shape


#  #### Droping Duplicates

# In[26]:


mse_df2=mse_df1.drop_duplicates()
mse_df2[mse_df2.duplicated()].shape


# In[27]:


len(mse_df2) # Final datapoint


# #### Checking Unique datapoint in text columns

# In[28]:


print(mse_df2['Drug'].unique())
print(mse_df2['Condition'].unique())
print(mse_df2['Reviews'].unique())
print(mse_df2['Side_Effect'].unique())
print(mse_df2['Age'].unique())


# In[29]:


total=279763


# In[30]:


Drug=len(mse_df2['Drug'].unique())
Drug


# In[31]:


percentage=Drug/total*100
percentage


# In[32]:


Condition=len(mse_df2['Condition'].unique())
Condition


# In[33]:


percentage=Condition/total*100
percentage


# In[34]:


Reviews=len(mse_df2['Reviews'].unique())
Reviews


# In[35]:


percentage=Reviews/total*100
percentage


# In[36]:


Side_Effect=len(mse_df2['Side_Effect'].unique())
Side_Effect


# In[37]:


percentage=Side_Effect/total*100
percentage


# In[38]:


Age=len(mse_df2['Age'].unique())
Age


# In[39]:


percentage=Age/total*100
percentage 


#  #### Dictionary 

# In[40]:


data= {'Name':['Drug','Condition','Reviews','Side_Effect','Age'],
        'Unique_values':(5014,1584,216359,1594,11),
       'Percentage':(1.79, 0.56, 77.33,0.57, 0.003)}


# In[41]:


uniq_df1=pd.DataFrame.from_dict(data)
uniq_df1


# In[42]:


uniq_df1.value_counts().plot(kind='barh', figsize=(8, 6))
#plt.xlabel("Unique Datapoint", labelpad=14)
plt.ylabel("Unique Datapoint with Percentage", labelpad=14)
plt.title("Unique Data Distribution", y=1.02);


#  ## 2) Data Cleaning

# In[43]:


dc_df=mse_df2.copy()
dc_df.head()


# In[44]:


dc_df1=dc_df.copy()


# In[45]:


# Creating function to clean the data
import re #regular expression
import string
def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("[0-9" "]+"," ",text)
    text = re.sub('[‘’“”...]', '', text)
    return text
clean = lambda x: clean_text(x)


# In[46]:


dc_df1['Condition']= dc_df1['Condition'].apply(clean)
dc_df1['Reviews']= dc_df1['Reviews'].apply(clean)
dc_df1['Side_Effect']= dc_df1['Side_Effect'].apply(clean)
dc_df1.head()


#  #### Expanding Contractions

# In[47]:


#Dictionary of English Contractions
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                    "can't": "cannot","can't've": "cannot have",
                    "'cause": "because","could've": "could have","couldn't": "could not",
                    "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                    "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                    "hasn't": "has not","haven't": "have not","he'd": "he would",
                    "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                    "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                    "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                    "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                    "it'd": "it would","it'd've": "it would have","it'll": "it will",
                    "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                    "mayn't": "may not","might've": "might have","mightn't": "might not", 
                    "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                    "mustn't've": "must not have", "needn't": "need not",
                    "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                    "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                    "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                    "she'll": "she will", "she'll've": "she will have","should've": "should have",
                    "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                    "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                    "there'd've": "there would have", "they'd": "they would",
                    "they'd've": "they would have","they'll": "they will",
                    "they'll've": "they will have", "they're": "they are","they've": "they have",
                    "to've": "to have","wasn't": "was not","we'd": "we would",
                    "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                    "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                    "what'll've": "what will have","what're": "what are", "what've": "what have",
                    "when've": "when have","where'd": "where did", "where've": "where have",
                    "who'll": "who will","who'll've": "who will have","who've": "who have",
                    "why've": "why have","will've": "will have","won't": "will not",
                    "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                    "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                    "y'all'd've": "you all would have","y'all're": "you all are",
                    "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                    "you'll": "you will","you'll've": "you will have", "you're": "you are",
                    "you've": "you have"}

# Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

# Function for expanding contractions
def expand_contractions(text,contractions_dict=contractions_dict):
 def replace(match):
   return contractions_dict[match.group(0)]
 return contractions_re.sub(replace, text)

# Expanding Contractions in the reviews, condition and Side_effect
dc_df1['Reviews']=dc_df1['Reviews'].apply(lambda x:expand_contractions(x))
dc_df1['Condition']=dc_df1['Condition'].apply(lambda x:expand_contractions(x))
dc_df1['Side_Effect']=dc_df1['Side_Effect'].apply(lambda x:expand_contractions(x))


# In[48]:


dc_df1.head()


# In[49]:


#final_dataset=dc_df1.copy()
#final_dataset.head()


# In[50]:


df=pd.read_csv("Final_Dataset.csv")
df.head()


# In[51]:


df.info()


# In[52]:


df1=df.isnull().sum()
df1


# In[53]:


df1=df.dropna()


# In[54]:


df2=df1[df1.duplicated()].shape
df2


# In[55]:


df2=df.drop_duplicates()


# In[56]:


len(df2) #Final datapoint


# In[57]:


df_dc=df2.copy()


# #### Drugs Distribution per Conditions

# In[58]:


df_dc=df.groupby(['Condition'])['Drug'].nunique().sort_values(ascending=False).reset_index()
df_dc.head()


# In[59]:


df_dc=df2.groupby(['Condition'])['Drug'].nunique().sort_values(ascending=False).reset_index().head(50)
plt.rcParams['figure.figsize']=[15, 6]
plt.bar(x=df_dc['Condition'],height =df_dc['Drug'], color = 'red', alpha=0.5)
plt.xticks(rotation=90)
plt.title('Drugs Available for top Conditions', fontsize=15)
plt.xlabel('Conditions', fontsize=15)
plt.ylabel('# of Drugs', fontsize=15)
plt.show()


#  #### Observation- This graphical presentation shows the drugs availability for top condition. There are  200 drugs available to treat condition like Birth Control, Cough, Pain and High BP etc.

# In[60]:


#df2=pd.DataFrame(final_dataset)
#final_dataset.to_csv('Final_Dataset1.csv')


#  #### Most Common Conditions based on Reviews

# In[61]:


cr_df=df2['Condition'].value_counts().reset_index()
cr_df.head()


# In[62]:


cr_df=cr_df.rename({'index':'Condition_Name','Condition':'Number of Condition'},axis=1)
cr_df.head()


# In[63]:


cr_df=df2['Condition'].value_counts().head(50).reset_index()
cr_df.columns = ['Condition','count']
plt.rcParams['figure.figsize']=[15,8]
plt.bar(x=cr_df['Condition'], height=cr_df['count'], color='green')
plt.xticks(rotation=90)
plt.title('Most Common Conditions Based on Reviews',fontsize =15)
plt.xlabel('Condition', fontsize=15)
plt.ylabel('# of Count', fontsize=15)
plt.show()


#  #### Observation- The above bar plot points out the most common condition among drug users. Pain, High BP, Depression and Birth Control are the top condition.

#  #### Extracting Year, Month from the date column

# In[64]:


date_df1=df2.copy()


# In[65]:


#Converting the date into datatime format
date_df1['Date']=pd.to_datetime(date_df1['Date'], errors ='coerce')
date_df1['Year']= date_df1['Date'].dt.year
date_df1['Month']= date_df1['Date'].dt.month
date_df1.head()


# #### Distribution of reviews in each year

# In[66]:


rev_df=date_df1.groupby(['Year'])['Reviews'].nunique().sort_values(ascending=False).reset_index()
rev_df


# In[67]:


plt.figure(figsize = (15, 8))
sns.countplot(date_df1['Year'], palette='CMRmap')
plt.title('Distribution of Reviews in each Year', fontsize =15)
plt.xlabel('Year', fontsize =15)
plt.ylabel('# of Reviews', fontsize=15)
plt.show()


# #### Distribution of Reviews in each month

# In[68]:


rev_df=date_df1.groupby(['Month'])['Reviews'].nunique().sort_values(ascending=False).reset_index()
rev_df


# In[69]:


plt.figure(figsize = (15, 8))
sns.countplot(date_df1['Month'], palette='RdYlBu')
plt.title('Distribution of Reviews in each Month', fontsize =15)
plt.xticks(np.arange(12),('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))
plt.xlabel('Month', fontsize =15)
plt.ylabel('# of Reviews', fontsize=15)
plt.show()


#  #### Observation- The above bar plot shows the distribution of reviews for each year and month. In 2009 and Jan, Oct month, we can see highest  reviews given by the user.

# #### Correlation Matrix for Rating dataset selection

# In[70]:


#Heat-Map
plt.rcParams['figure.figsize']=(10,8)
sns.set(font_scale=1.2)
cm_df=date_df1[['EaseofUse','Effectiveness','Satisfaction']]
corr=cm_df.corr()
#sns.heatmap(corr, annot= True, vmin=1, vmax=1,center=0.5, cmap='twilight', square=True);
plt.xticks(rotation=45)

g=sns.heatmap(corr, annot=True, cmap="copper")
    


#  #### Observation- The above heatmap indicates about drug effectiveness is highly correlated with satisfaction.That's reason we have selected satisfaction column for our rating analysis.

#  #### Rating Distribution 

# In[71]:


rating_df=date_df1.rename({'Satisfaction':'Rating'}, axis=1)
rating_df.head()


# In[72]:



# Frequency of each rating
rating_df1 = rating_df['Rating'].value_counts().reset_index()

# Converting float rating values to int
rating_df1.columns = ['Rating','count']
rating_df1 = rating_df1.astype({'Rating':'int'})

# Plotting user rating distribution
size = rating_df1['count']
colors = ['salmon','lavender','lightgreen','pink','wheat','azure','sienna','orange','turquoise','olive']
labels = rating_df1['Rating']

my_circle = plt.Circle((0, 0), 0.7, color = 'white')

plt.rcParams['figure.figsize'] = (7, 7)
plt.pie(size,colors = colors,labels = labels, autopct = '%.2f%%')
plt.title('User Rating Distribution', fontsize = 15)
plt.legend()
p = plt.gcf()
plt.gca().add_artist(my_circle)
plt.show()


#  #### Observation- The above donut chart shows how ratings from scale 1 to 5 are distributed in our dataset. As we can see the majority of users have given a higher rating as compared to lower ones.

#  #### Distribution of Ratings

# In[73]:


#plt.rcParams['figure.figsize']=(15,8)
#sns.boxplot(x=rating_df['Year'],y=rating_df['Rating'], palette='Greens')
#plt.title('Distribution of Ratings in each Year', fontsize= 15)
#plt.xlabel('Year', fontsize=15)
#plt.ylabel('Rating', fontsize=15)
#plt.show()


# In[74]:


#plt.figure(figsize = (15, 8))
#sns.boxplot(x=rating_df['Month'],y=rating_df['Rating'], palette='pastel')
#plt.title('Distribution of Rating in each Month', fontsize =15)
#plt.xticks(np.arange(12),('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))
#plt.xlabel('Month', fontsize =15)
#plt.ylabel('Rating', fontsize=15)
#plt.show()


#  #### Average Usefulcount vs Rating

# In[75]:


group=rating_df.groupby(['Rating'])['UsefulCount']
avg=group.mean().reset_index()
avg


# In[76]:


#plt.figure(figsize=(10,5))
#df2.groupby('Condition')['Reviews'].nunique().nlargest(10).plot(kind='bar', color='seagreen')
#plt.title("Number of Reviwes Per Condition")
#plt.grid()
#plt.show()


# In[77]:


# Scatter Plot
#avg_useful_count_list = []

#ratings = range(1, 6)

#for i in ratings:
    #avg_useful_count_list.append([i, np.sum(rating_df[rating_df['Rating'] == i].UsefulCount) / np.sum([rating_df['Rating'] == i])])
    
count_arr = np.asarray(avg)
plt.rcParams['figure.figsize'] = (15, 8)
plt.scatter(count_arr[:, 0], count_arr[:, 1], c=count_arr[:, 0], cmap = 'Dark2', s=400)
plt.title('Average Useful Count vs Rating',fontsize = 15)
plt.xlabel('Rating',fontsize = 15)
plt.ylabel('Average Useful Count',fontsize = 15)
plt.xticks(np.arange(1,6))
plt.yticks(np.arange(0,20,5))
plt.grid()


#  #### Observation- The above scatter plot indicates about users who found review useful. The plot shows a slightly U shape trend the average useful count of a review versus overall rating.

#  #### Side Effect Analysis based on Age factor

# In[78]:


age_df=pd.read_csv("Age_Dataset.csv")
age_df.head()


# In[79]:


se_df=age_df.groupby(['Age'])['Side_Effect'].nunique().sort_values(ascending=False).reset_index()
se_df


# In[80]:


ag=age_df['Age'].value_counts().reset_index()
ag.columns = ['Age','count']
plt.rcParams['figure.figsize']=[15,8]
plt.bar(x=ag['Age'], height=ag['count'], color='Blue')
#plt.xticks(rotation=90)
plt.title(' Side_Effect Analysis',fontsize =15)
plt.xlabel('Age', fontsize=15)
plt.ylabel('# of Count', fontsize=15)
plt.show()


# #### Observation - The above bar chart indicates correlation between Side-Effect and Age factor. as per visualization drug side effects impacting more on people age between 45-54.

#  #### Gender Analysis based on Drugs

# In[81]:


age_df['Gender'].value_counts().reset_index()


# In[82]:


plt.figure(figsize=(9,6))
age_df['Gender'].value_counts().plot(kind='barh')
plt.xticks(fontsize = 10) 
plt.title("Gender Analysis", fontsize = 16, fontweight = "bold") 
plt.ylabel("Gender", fontsize = 13 ) 


#  #### Observation- The above bar chart shows how drug is impacting on gender, as per chart Female is consuming more drugs.

#  ## Sentiment Analysis

# In[82]:


sent_df=df2[['Reviews','Drug','Condition','Side_Effect','Satisfaction']].copy()
sent_df


# In[83]:


sent_df=sent_df.replace(r'^\s*$', np.NaN, regex=True)
print(sent_df.isnull().sum())


# In[84]:


sent_df2=sent_df.dropna()


# In[86]:


len(sent_df2)


# In[87]:


rev_text=sent_df2.Reviews.tolist()
rev_text


# In[88]:


#Joining the Reviews
values = ','.join(str(v) for v in rev_text).lower()
values


# In[89]:


# Punctuation
no_punc_text=values.translate(str.maketrans('', '', string.punctuation))
print(no_punc_text[0:241114])


# In[86]:


#Stopwords

import nltk
from nltk.corpus import stopwords
nltk.download ('punkt')


# In[87]:


#Tokenization
from nltk.tokenize import word_tokenize
text_tokens= word_tokenize(no_punc_text)
print(text_tokens[0:241114])


# In[88]:


#Remove the stopwords
import nltk
nltk.download('stopwords')

my_stop_words =stopwords.words('english')
my_stop_words.append('the')
no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens[0:241146])


# In[89]:


#Noramalize the data
lower_words=[Reviews.lower() for Reviews in no_stop_tokens]
print(lower_words[0:241114])


# In[90]:


#Stemming
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemmed_tokens =[ps.stem(word) for word in lower_words]
print(stemmed_tokens[0:241114])


# In[91]:


#NLP english language model of spacy library
nlp=spacy.load("en_core_web_sm")


# In[92]:



text=("'im a retired physician and of all the meds i have tried for my allergies seasonal and not  this one is the most effective for me  when i first began using this drug some years ago  tiredness as a problem but is not currently',")


# In[93]:


#Check spacy is working or not
doc=nlp(text)
doc


# In[94]:


#Lemmatization (checking)

doc1=nlp("im a retired physician and of all the meds i have tried for my allergies seasonal and not  this one is the most effective for me  when i first began using this drug some years ago  tiredness as a problem but is not currently")
print(doc1[0:1000000])


# In[95]:


lemmas = [token.lemma_ for token in doc1]
print(lemmas[0:25])


#  ### Word Cloud

# In[96]:


from PIL import Image

wc_image = np.array(Image.open('wc_mask.jpg'))
wordcloud1 = WordCloud(max_font_size=50, mask = wc_image, max_words=500, width=500,contour_width=1, contour_color='red', background_color="white").generate(values)
plt.figure(figsize=(16,8))
plt.imshow(wordcloud1, interpolation="bilinear")
plt.axis("off")
plt.show()


#  ### Polarity Score

# In[87]:


sent_df1=sent_df2.copy()


# In[88]:


sent_df1.head()


# In[89]:


# Create a funcation to get the polarity
def getPloarity(text):
   return TextBlob(text).sentiment.polarity


# In[90]:


sent_df1['Polarity Score']=sent_df1['Reviews'].apply(lambda Reviews: getPloarity(str(Reviews)))
sent_df1.head()


# In[101]:


#sent_df1=sent_df1.drop('score', axis=1)
#sent_df1=sent_df1.drop('compound', axis=1)
#sent_df1=sent_df1.drop('Analysis2', axis=1)
#sent_df1=sent_df1.drop('Analysis3', axis=1)
#sent_df1=sent_df1.drop('comp_score', axis=1)
#sent_df1=sent_df1.drop('Analysis', axis=1)


# In[91]:


def get_sentiment(score):
    if score>0.02:
        return 'Positive'
    elif score==0:
        return 'Neutral'
    else:
        return 'Negative'
   
    
sent_df1['Analysis'] = sent_df1['Polarity Score'].apply(get_sentiment)


# In[92]:


sent_df1


# In[104]:


# Print All the positive reviews

j=1
sortedDF =sent_df1.sort_values(by=['Polarity Score'])
for i in range(0, sortedDF.shape[0]):
    if (sortedDF['Analysis'][i] == 'Positive'):
        
        print(str(j)+ ')' + sortedDF['Reviews'][i])
        print()
        j=j+1


# In[ ]:


# Print All the Negative reviews

j=1
sortedDF =sent_df1.sort_values(by=['Polarity Score'], ascending='False')

for i in range(0, sortedDF.shape[0]):
    if (sortedDF['Analysis'][i] == 'Negative'):
        
        print( (str(j)+ ')' + sortedDF['Reviews'][i]))
        print()
        j=j+1


# In[ ]:


# Print All the Neutral reviews

j=1
sortedDF =sent_df1.sort_values(by=['Polarity Score'])
for i in range(0, sortedDF.shape[0]):
    if (sortedDF['Analysis'][i] == 'Neutral'):
        
        print(str(j)+ ')' + sortedDF['Reviews'][i])
        print()
        j=j+1


# In[ ]:


ptexts=sent_df1[sent_df1.Analysis == 'Positive']
ptexts_p=ptexts['Reviews']

round( (ptexts_p.shape[0] / sent_df1.shape[0]) * 100, 1)


# In[ ]:


p=len(ptexts_p)
p


# In[ ]:


ptexts=sent_df1[sent_df1.Analysis == 'Negative']
ptexts_n=ptexts['Reviews']

round( (ptexts_n.shape[0] / sent_df1.shape[0]) * 100, 1)


# In[ ]:


n=len(ptexts_n)
n


# In[105]:


ptexts=sent_df1[sent_df1.Analysis == 'Neutral']
ptexts_ne=ptexts['Reviews']

round( (ptexts_ne.shape[0] / sent_df1.shape[0]) * 100, 1)


# In[106]:


ne=len(ptexts_ne)
ne


# In[107]:


#keys=('Positive','Negative','Neutral')
#values=('119388','86465','35293')


# In[108]:



fig,(ax1,ax2)=plt.subplots(1,2, figsize=(10,4))
fig.suptitle('Sentiment Analysis')
sent_df1['Analysis'].value_counts().plot.bar(ax=ax1, color='tomato', ec="black")
sent_df1['Analysis'].value_counts().plot.pie(ax=ax2, autopct='%1.1f%%', colors = ( "coral", "sienna","orangered"),explode = (0,  0,  0.0),shadow=True,startangle=90)


#  #### Observation- The above bar and pie chart shows how reviews are distributed , as per graph 49.5% positive reviews are given by the consumer.

# In[109]:


#Testing the Review

def get_sentiment(review): 
    ''' 
    Utility function to classify sentiment of passed tweet 
    using textblob's sentiment method 
    '''
    # create TextBlob object of passed tweet text 
    analysis = TextBlob(review)
    # set sentiment 
    if analysis.sentiment.polarity > 0: 
        return 'positive'
    elif analysis.sentiment.polarity == 0: 
        return 'neutral'
    else: 
        return 'negative'


# In[110]:


a = 'cleared me right up even with my throat hurting it went away after taking the medicine'
get_sentiment(a)


#  ## Model Building 

# In[93]:


model_df=sent_df1.copy()
model_df.head()


# In[94]:


model_df.to_csv("new_data.csv")


# In[113]:


model_df['Analysis'].value_counts()


# ## Split the data into train and test sets

# In[114]:


from sklearn.model_selection import train_test_split

x=model_df['Reviews'].values.astype('U')
y=model_df['Analysis']

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.20, random_state=42)


# In[110]:


#from sklearn.model_selection import train_test_split

#x=model_df.iloc[:,0:3]
#y=model_df['Analysis']


#  ### CountVectorizer
#  
#  compute word counts using CountVectorizer 

# In[111]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

x_train_counts = count_vect.fit_transform(x_train)


# In[112]:


x_train_counts


# In[113]:


x_train_counts.shape


#  ## TfidfVectorizer`
#     
#     The TfidfVectorizer will tokenize documents, learn the vocabulary and inverse document frequency weightings, and allow you to encode new documents

# In[115]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer =TfidfVectorizer()

x_train_tfidf= vectorizer.fit_transform(x_train)
x_test_tfidf= vectorizer.fit_transform(x_test)
x_train_tfidf.shape


# In[ ]:





# In[116]:


import pickle
# Creating a pickle file for the CountVectorizer
pickle.dump(vectorizer, open('TfidfVectorizer.pkl', 'wb'))


# ## Model 1: Random Forest Classification

# In[117]:


# all model library
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics


#  ### Build the Pipeline

# In[113]:


text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('rfc', RandomForestClassifier(n_estimators=100)),
])

# Feed the training data through the pipeline
text_clf.fit(x_train, y_train) 


# ### Test

# In[107]:


predictions_test = text_clf.predict(x_test)
predictions_test


# ### Train

# In[114]:


predictions_train = text_clf.predict(x_train)
predictions_train


# ## Accuracy

# ### Test

# In[115]:


from sklearn import metrics
#print(metrics.accuracy_score(y_test, predictions))

rf_test=accuracy_score(y_test, predictions_test)*100
print("Accuracy:", rf_test)


# ### Train

# In[116]:


rf_train=accuracy_score(y_train, predictions_train)*100
print("Accuracy:", rf_train)


# In[117]:


print("Accuracy on training set: {}".format(text_clf.score(x_train, y_train)*100))
print("Accuracy on test set: {}".format(text_clf.score(x_test, y_test)*100))


# ## Confusion Matrix

# In[128]:


print(metrics.confusion_matrix(y_test,predictions))


# In[129]:


print(metrics.classification_report(y_test,predictions))


# ## Model 2: Naive Bayes 

# ## Build the Pipeline

# In[99]:


from sklearn.naive_bayes import MultinomialNB

text_clf_mnb = Pipeline([('tfidf', TfidfVectorizer()),
                     ('mnb', MultinomialNB(alpha=0.01)),
])

# Feed the training data through the pipeline
text_clf_mnb.fit(x_train, y_train) 


# ### Prediction

# In[100]:


#Test
prediction_mnb_test=text_clf_mnb.predict(x_test)
prediction_mnb_test


# In[101]:


#Train
prediction_mnb_train=text_clf_mnb.predict(x_train)
prediction_mnb_train


# ## Accuracy

# In[102]:


#Test
mnb_test=accuracy_score(y_test,prediction_mnb_test)*100
print("Accuracy",mnb_test)


# In[103]:


#Train
mnb_train=accuracy_score(y_train,prediction_mnb_train)*100
print("Accuracy",mnb_train)


# In[104]:


print("Accuracy on training set: {}".format(text_clf_mnb.score(x_train, y_train)*100))
print("Accuracy on test set: {}".format(text_clf_mnb.score(x_test, y_test)*100))


# ## Confusion Matrix

# In[ ]:


# Test


# In[105]:


print(metrics.confusion_matrix(y_test,prediction_mnb_test))


# In[106]:


print(metrics.classification_report(y_test,prediction_mnb_test))


# In[107]:


#Train


# In[108]:


print(metrics.confusion_matrix(y_train,prediction_mnb_train))


# In[109]:


print(metrics.classification_report(y_train,prediction_mnb_train))


# ## Model 3: SVC

# ### Build the Pipeline

# In[110]:


from sklearn.svm import LinearSVC

svc_clf=Pipeline([('tfidf', TfidfVectorizer()),
                 ('svc',LinearSVC()),
                 ])
# Feed the training data through the pipeline
svc_clf.fit(x_train, y_train)


# ### Prediction

# In[111]:


# Test
prediction_svc_test=svc_clf.predict(x_test)
prediction_svc_test


# In[112]:


#Train
prediction_svc_train=svc_clf.predict(x_train)
prediction_svc_train


#  ## Accuracy

# In[113]:


#Test
svc_test=accuracy_score(y_test, prediction_svc_test)*100
print("Accuracy",svc_test)


# In[114]:


#Train
svc_train=accuracy_score(y_train, prediction_svc_train)*100
print("Accuracy",svc_train)


# In[115]:


print("Accuracy on training set: {}".format(svc_clf.score(x_train, y_train)*100))
print("Accuracy on test set: {}".format(svc_clf.score(x_test, y_test)*100))


# ## Confusion Matrix

# In[ ]:


#Test


# In[116]:


print(metrics.confusion_matrix(y_test, prediction_svc_test))


# In[117]:


print(metrics.classification_report(y_test, prediction_svc_test))


# In[118]:


#Train


# In[119]:


print(metrics.confusion_matrix(y_train, prediction_svc_train))


# In[120]:


print(metrics.classification_report(y_train, prediction_svc_train))


# ## Model 4: Logistic Regression

# ### Build the Pipline

# In[ ]:


from sklearn.linear_model import LogisticRegression
logistic_model=Pipeline([('tfidf', TfidfVectorizer()),
                        ('logistic', LogisticRegression()),
                        ])

# Feed the training data through the pipeline

logistic_model.fit(x_train, y_train)


# In[131]:


import pickle
# Creating a pickle file for the CountVectorizer
pickle.dump(logistic_model, open('logisitc.pkl', 'wb'))


# ### Prediction

# In[122]:


# Test
prediction_log_test=logistic_model.predict(x_test)
prediction_log_test


# In[126]:


# Train
prediction_log_train=logistic_model.predict(x_train)
prediction_log_train


#  ## Accuracy

# In[124]:


#Test
logistic_test=accuracy_score(y_test,prediction_log_test )*100
print("Accuracy", logistic_test)


# In[128]:


#Train
logistic_train=accuracy_score(y_train,prediction_log_train )*100
print("Accuracy", logistic_train)


# In[129]:


print("Accuracy on training set: {}".format(logistic_model.score(x_train, y_train)*100))
print("Accuracy on test set: {}".format(logistic_model.score(x_test, y_test)*100))


# ## Confusion Matrix

# In[ ]:


#Test


# In[130]:


print(metrics.confusion_matrix(y_test, prediction_log_test))


# In[131]:


print(metrics.classification_report(y_test, prediction_log_test))


# In[132]:


#Train


# In[133]:


print(metrics.confusion_matrix(y_train, prediction_log_train))


# In[134]:


print(metrics.classification_report(y_train, prediction_log_train))


# ## Model 5: XG-Boost

# ### Build Pipline

# In[135]:


from xgboost import XGBClassifier
xgb_clf= Pipeline([('tfidf', TfidfVectorizer()),
              ('xbg', XGBClassifier()),
              ])
#Feed the training data through the pipline
xgb_clf.fit(x_train, y_train)


# In[136]:


### Prediction


# In[137]:


#Test
prediction_xgb_test=xgb_clf.predict(x_test)
prediction_xgb_test


# In[138]:


#Train
prediction_xgb_train=xgb_clf.predict(x_train)
prediction_xgb_train


#  ## Accuracy

# In[140]:


#Test
xgb_test=accuracy_score(y_test, prediction_xgb_test)*100
print("Accuracy",xgb_test)


# In[141]:


xgb_train=accuracy_score(y_train, prediction_xgb_train)*100
print("Accuracy",xgb_train)


# In[144]:


print("Accuracy on training set: {}".format(xgb_clf.score(x_train, y_train)*100))
print("Accuracy on test set: {}".format(xgb_clf.score(x_test, y_test)*100))


# ## Confusion Matrix

# In[ ]:


# Test


# In[145]:


print(metrics.confusion_matrix(y_test, prediction_xgb_test))


# In[146]:


print(metrics.classification_report(y_test, prediction_xgb_test))


# In[ ]:


# Train


# In[147]:


print(metrics.confusion_matrix(y_train, prediction_xgb_train))


# In[148]:


print(metrics.classification_report(y_train, prediction_xgb_train))


#  ## Neural Network MLP Classifier

# In[ ]:


#from sklearn.neural_network import MLPClassifier
#mlp= Pipeline([('tfidf', TfidfVectorizer()),
              #('mlp', MLPClassifier()),
              #])
#Feed the training data through the pipline
#mlp.fit(x_train, y_train)


# In[110]:


##prediction_mlp=mlp.predict(x_test)
#prediction_mlp


#  ## Summary of Model Accuracy Score

# In[151]:


Result={"Model":pd.Series(["SVC","Logistic Regression"," XG-Boost","Naive Bayes"]),"Train_Accuracy":pd.Series([96.64,94.3,90.16,80.73]),"Test_Accuracy":pd.Series([93.06,92.19,87.96,69.63])}


# In[152]:


Results=pd.DataFrame(Result)
Results


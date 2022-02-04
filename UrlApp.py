import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import pickle
import zipfile

with zipfile.ZipFile("./Url_spam_model.zip",'r') as zipref:
    zipref.extractall('./')

st.image("./UrlSpam.jpg")

## User Input Processing
def userInput():
    in_url = st.text_input("Paste URL here !")
    data = {"url":in_url}
    dt = pd.DataFrame(data,index=[0])
    return dt


dt = userInput()

## Data piplines

dt["url_length"] = [len(u) for u in dt.url]
dt["IS_subscribe"] = [1 if 'subscribe' in u else 0 for u in dt.url ]
dt["IS_www"] = [1 if 'www' in u else 0 for u in dt.url]
dt["IS_https"] = [1 if 'https' in u else 0 for u in dt.url]
dt["word_count"] = [len(u.split("/")) for u in dt.url]


#### Domain name creation
dt["Domain"] = [u[u.rfind('.')+1:u.rfind("/",3)].split("/")[0] for u in dt.url]
doms = ["com"," ","org","co","io","mp","uk","net","fm","edu","gov"]
dt["Domain_grp"] = [u if u in doms else "other" for u in dt["Domain"]]
domain_codes = {"com":0," ":1,"org":2,"co":3,"io":4,"mp":5,"uk":6,"net":7,"fm":8,"edu":9,"gov":10,"other":11}
dt.Domain_grp = dt.Domain_grp.map(domain_codes)
dt.drop(columns=['url','Domain'],inplace=True)


model = pickle.load(open("./Url_spam_model.pkl","rb"))



if st.button("Click me to check Prediction"):
    result = model.predict(dt)
    if result[0]==1:
        st.markdown("<h5 style='text-align:center; color:Tomato'>This url is may be Spam !!!</h5>",unsafe_allow_html=True)
    else:
        st.markdown("<h5 style='text-align:center; color:green'>This url is not Spam !!!</h5>",unsafe_allow_html=True)


st.write("")
st.write("")
st.image("./Deciding_Factors.jpg")

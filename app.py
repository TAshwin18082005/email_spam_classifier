import streamlit as st
import pickle
import nltk,string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()


# transforming the text:

def Text_transform(text):
    text=text.lower()                       # transforming to lower case.
    text=nltk.word_tokenize(text)           # tokenizing words.
    y=[]                    
    for i in text:      
        if i.isalnum(): 
            y.append(i)                     # removing special characters.

    text=y[:]
    y.clear()
    for i in text:
        if i not in string.punctuation and i not in stopwords.words('english'):
            y.append(i)    
    
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfid= pickle.load(open('vectorised.pkl','rb'))
model= pickle.load(open('model.pkl','rb'))



st.title("Email spam classifier")
input_text=st.text_area("Enter the message")

if st.button('predict'):

    # pre_processing:
    transformed_text=Text_transform(input_text)
    # vectorise:
    vector_input=tfid.transform([transformed_text])
    # model training:
    result=model.predict(vector_input)[0]
    # displaying the result:
    if result == 1:
        st.header("spam")
    else:
        st.header("Not spam")


# open virtual env and then --- new_venv\Scripts\activate
# use command ----- streamlit run app.py
import streamlit as st
import pandas as pd 
import pickle
from sklearn.preprocessing import LabelEncoder
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

# Load your model
with open('SMSSpamCollection/model.pkl', 'rb') as file:
    model, tfidf_vectorizer = pickle.load(file)

# Function to predict if the SMS is spam or not
def predict_spam(text):
    print(text)
    transform_text1 = transform_text(text)
    X = tfidf_vectorizer.transform([transform_text1]).toarray()
    print(X)
    prediction = model.predict(X)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Streamlit App
st.title("SMS Classification App")
st.write("Enter an SMS message to classify it as Spam or Not Spam.")

# Input text box
user_input = st.text_area("SMS Input")

# Predict button
if st.button("Classify"):
    result = predict_spam(user_input)
    st.write(f"The SMS is: {result}")

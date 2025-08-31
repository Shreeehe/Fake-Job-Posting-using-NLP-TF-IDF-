import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack
import re
from nltk.corpus import stopwords
import nltk

# --- App Configuration ---
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🕵️‍♀️",
    layout="centered"
)

# --- NLTK Stopwords ---
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# --- Load Model and Vectorizer ---
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load('svm_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# --- Text Cleaning ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# --- UI ---
st.title("🕵️‍♀️ Fake Job Posting Detector")
st.write(
    "Enter the details of a job posting below. Our highly trained SVM detective "
    "will analyze the information and give its verdict!"
)
st.markdown("---")

# --- Inputs ---
col1, col2 = st.columns(2)
with col1:
    has_company_logo = st.radio("Does it have a company logo?", ("Yes", "No"))
    has_questions = st.radio("Does it have screening questions?", ("Yes", "No"))
    telecommuting = st.radio("Is it a remote/telecommuting job?", ("Yes", "No"))

with col2:
    title = st.text_input("Job Title", placeholder="e.g., Marketing Intern")
    industry = st.text_input("Industry", placeholder="e.g., Marketing")
    function = st.text_input("Job Function", placeholder="e.g., Marketing")

company_profile = st.text_area("Company Profile", placeholder="A short description of the company...")
description = st.text_area("Job Description", placeholder="A description of the job duties...")
requirements = st.text_area("Job Requirements", placeholder="List of requirements for the job...")
benefits = st.text_area("Benefits", placeholder="List of benefits offered...")

# --- Prediction ---
if st.button("Get Detective's Verdict", type="primary"):

    # 1. Collect inputs
    job_data = {
        'title': [title],
        'company_profile': [company_profile],
        'description': [description],
        'requirements': [requirements],
        'benefits': [benefits],
        'telecommuting': [1 if telecommuting == "Yes" else 0],
        'has_company_logo': [1 if has_company_logo == "Yes" else 0],
        'has_questions': [1 if has_questions == "Yes" else 0],
        'industry': [industry],
        'function': [function]
    }
    unseen_job_df = pd.DataFrame(job_data)

    # 2. Preprocess
    unseen_job_df['has_company_profile'] = unseen_job_df['company_profile'].apply(lambda x: 0 if pd.isna(x) or x.strip() == '' else 1)
    unseen_job_df['has_benefits'] = unseen_job_df['benefits'].apply(lambda x: 0 if pd.isna(x) or x.strip() == '' else 1)

    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits', 'industry', 'function']
    unseen_job_df['combined_text'] = unseen_job_df[text_columns].agg(' '.join, axis=1)  # FIXED HERE
    unseen_job_df['cleaned_text'] = unseen_job_df['combined_text'].apply(clean_text)

    new_text_tfidf = vectorizer.transform(unseen_job_df['cleaned_text'])
    numerical_clues = unseen_job_df[['telecommuting', 'has_company_logo', 'has_questions', 'has_company_profile', 'has_benefits']].values

    new_data_prepared = hstack([new_text_tfidf, numerical_clues])

    # 3. Predict
    verdict = model.predict(new_data_prepared)

    # 4. Display
    st.markdown("---")
    st.subheader("🕵️‍♀️ The Verdict Is In...")
    if verdict[0] == 0:
        st.success("This job posting seems to be REAL ✅")
    else:
        st.error("This job posting seems to be FAKE ❌")

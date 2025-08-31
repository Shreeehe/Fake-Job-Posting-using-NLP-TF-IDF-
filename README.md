# 🕵️‍♀️ Fake Job Posting Detector  

A machine learning project to identify fraudulent job postings, built with **Scikit-learn** and deployed with a **Streamlit web application**.  

---

## ✨ Overview  
In today’s job market, it’s easy for scammers to create fake job listings to trick applicants.  
This project aims to tackle that problem using **Natural Language Processing (NLP)** and **Machine Learning** to build a model that can distinguish between **real and fake job postings**.  

We explored the dataset, engineered features, compared multiple models, and finally deployed our best-performing model into a user-friendly web app.  

---

## 💾 Dataset  
We used the **Real or Fake Job Postings Prediction** dataset from Kaggle.  
- **18,000 job postings**  
- **18 features** including a flag for fraudulent posts  

📂 [Dataset on Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-job-postings)  

---

## 🚀 Our Journey & Methodology  

### 1. Exploratory Data Analysis (EDA) 🗺️  
We first investigated the dataset and identified key patterns:  
- Dataset is **highly imbalanced** (more real jobs than fake).  
- Fake postings often lack a **company logo**.  
- Screening questions are usually missing.  
- Descriptions are vague with no company profile/benefits.  
- Many target **entry-level candidates**.  

---

### 2. Data Preparation & Preprocessing 🧹  
Steps taken to prepare the dataset:  
- **Feature Engineering**: Combined text columns (`title`, `description`, `requirements`, etc.) into one.  
- **Text Cleaning**:  
  - Lowercased text  
  - Removed punctuation  
  - Removed stopwords  

---

### 3. Modeling & Experiments 🔬  
We tested multiple NLP techniques and classification models:  

- **NLP Techniques**:  
  - **TF-IDF**: Word frequency importance  
  - **Word2Vec**: Semantic meaning of words  
  - **DistilBERT**: Transformer model for deep contextual understanding  

- **Classification Models**:  
  - Logistic Regression  
  - Support Vector Machine (SVM)  
  - Gradient Boosting  

---

### 4. The Champion Model 🏆  
The best performing model:  
- **SVM + TF-IDF**  
- **F1-Score (Fake class): 0.77**  

✅ Best balance of **precision** & **recall**  
✅ Much faster than DistilBERT → more practical for real-world deployment  

---

## 🛠️ Technologies Used  
- **Data Analysis**: Pandas, NumPy  
- **Visualizations**: Matplotlib, Seaborn  
- **NLP & ML**: Scikit-learn, NLTK  
- **Deployment**: Streamlit  
- **Model Storage**: Joblib  

---

## 💻 Run the App Locally  

### Prerequisites  
- Python 3.7+  
- pip  

### Setup  
```bash
# Clone repo
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

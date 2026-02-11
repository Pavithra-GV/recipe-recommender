# 🍲 Recipe Recommender

## Overview
This project is an intelligent recipe recommendation system designed to help students and budget-conscious users cook meals using the ingredients they already have. Instead of suggesting complex recipes that require many new items, the system prioritizes recipes with minimal additional ingredients, low estimated cost, and simple preparation. By combining semantic understanding of ingredients with natural language processing of cooking instructions, the recommender provides practical, affordable, and waste-reducing meal suggestions in an interactive web application.

---

## Why This Project Is Useful
- Reduces food waste by maximizing the use of existing pantry ingredients  
- Helps students cook affordable meals with minimal extra spending  
- Provides simple, quick recipes suited for everyday cooking  
- Offers explainable recommendations showing what you have vs. what to buy  

---

## Novelty Points
- Ingredient-gap–aware recommendation prioritizing minimal extra ingredients  
- Budget-aware scoring to favor low-cost recipes  
- SBERT-based semantic ingredient matching instead of exact keyword matching  
- NLP extraction of hidden ingredients from cooking instructions  

---

## Tech Stack
- **Programming Language:** Python  
- **Machine Learning / NLP:** Sentence-BERT (SBERT), spaCy  
- **Data Processing:** Pandas, NumPy, scikit-learn  
- **Web App Framework:** Streamlit  
- **Version Control & Deployment:** GitHub, Streamlit Community Cloud  

---

## Dataset link 

- https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews

---

## How to Download and Run the Project

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd recipe-recommender
```
### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```
### 3. Run the Streamlit App
```bash
pip install -r requirements.txt
```
### 4. Run the Streamlit App
```bash
streamlit run app/app.py
```

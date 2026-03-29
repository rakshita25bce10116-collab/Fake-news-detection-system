# Fake News Detection System 📰🕵️‍♂️

**Author:** Rakshita Rao  
**Registration Number:** 25BCE10116  
**Program:** B.Tech CSE Core, 1st Year  

## Overview
This project is a Machine Learning application designed to classify news articles as either **Real** or **Fake**. In an era of rampant misinformation, distinguishing credible news from fabricated stories is a critical real-world problem. This project applies Natural Language Processing (NLP) techniques to analyze text and predict its authenticity.

This project was built as the Bring Your Own Project (BYOP) for the Fundamentals of AI and ML course and submitted via the VITyarthi platform.

## How It Works
The system uses a **TF-IDF Vectorizer** (Term Frequency-Inverse Document Frequency) to convert raw news text into numerical features. It then feeds those features into a **Logistic Regression** classification model. 
* **TF-IDF** helps identify the most significant words in a document relative to the entire dataset.
* **Logistic Regression** is an efficient and highly interpretable algorithm for binary classification tasks (Fake vs. Real).

## Prerequisites
To run this project, you need Python installed on your system along with the following libraries:
* `pandas` (for data manipulation)
* `scikit-learn` (for machine learning models and evaluation)

You can install the dependencies using:
`pip install pandas scikit-learn`

## Setup and Execution
1. **Clone the repository:**
   `git clone https://github.com/rakshita25bce10116-collab/Fake-news-detection-system/tree/main`
2. **Add the dataset:**
   Ensure you have the dataset saved as `news.csv` in the root directory of this project. It must contain at least two columns: `text` (the article content) and `label` (the classification).
3. **Run the script:**
   Navigate to the directory in your terminal and run:
   `python main.py`

## Expected Output
The script will output the loading status, confirm when training is complete, and then print the overall **Accuracy Score** as well as a detailed **Classification Report** (Precision, Recall, F1-Score) evaluating how well it identified both fake and real news on the testing data.

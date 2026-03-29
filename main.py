import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("Initializing Fake News Detection System...")

    # 1. Load the Dataset
    # Make sure you have a dataset named 'news.csv' in the same folder
    try:
        df = pd.read_csv('news.csv')
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: 'news.csv' not found. Please add the dataset to the directory.")
        return

    # Assuming the dataset has 'text' for the article content and 'label' for Fake/Real
    X = df['text']
    y = df['label']

    # 2. Split the data into Training and Testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Feature Extraction using TF-IDF
    # This converts the raw text into a matrix of TF-IDF features
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = vectorizer.fit_transform(X_train)
    tfidf_test = vectorizer.transform(X_test)

    # 4. Initialize and Train the Machine Learning Model
    # We are using Logistic Regression for its efficiency with text classification
    model = LogisticRegression()
    model.fit(tfidf_train, y_train)
    print("Model training complete.")

    # 5. Make Predictions and Evaluate
    y_pred = model.predict(tfidf_test)
    
    score = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {round(score*100, 2)}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Optional: Test with a custom headline
    # sample_news = ["Scientists discover a new planet made entirely of chocolate"]
    # vectorized_sample = vectorizer.transform(sample_news)
    # prediction = model.predict(vectorized_sample)
    # print(f"\nPrediction for sample news: {prediction[0]}")

if __name__ == "__main__":
    main()

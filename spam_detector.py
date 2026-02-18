import pandas as pd
import requests
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# --- Constants ---
DATASET_URL = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
LOCAL_FILE = "spam_data.csv"


# --- Data Loading and Preparation ---
def load_and_prepare_data():
    """Loads data from local file or downloads it, prepares labels."""
    df = None
    if not os.path.exists(LOCAL_FILE):
        print(f"{LOCAL_FILE} not found. Attempting to download from {DATASET_URL}...")
        try:
            response = requests.get(DATASET_URL, timeout=10)
            response.raise_for_status()
            with open(LOCAL_FILE, 'wb') as f:
                f.write(response.content)
            print(f"Dataset downloaded successfully and saved as {LOCAL_FILE}.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading dataset: {e}")
            print("Falling back to dummy data.")
            # Define dummy data if download fails
            data = {
                'text': ["Free Viagra now!!!", "Winner! You have won a free ticket", "How are you doing?", "Meeting scheduled for tomorrow", "Urgent account update required", "Can you send the report?", "Claim your prize now", "Dinner tonight?", "Special offer just for you", "See you later"],
                'label': ['spam', 'spam', 'ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
            }
            df = pd.DataFrame(data)
    else:
        print(f"Found local file: {LOCAL_FILE}.")

    # Read the CSV if df hasn't been assigned dummy data
    if df is None:
        try:
            df = pd.read_csv(LOCAL_FILE, encoding='latin-1', usecols=['v1', 'v2'])
            df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
            print("Data loaded successfully from file.")
        except Exception as e:
            print(f"Error reading {LOCAL_FILE}: {e}")
            print("Falling back to dummy data.")
            # Define dummy data again if initial file existed but was unreadable
            data = {
                'text': ["Free Viagra now!!!", "Winner! You have won a free ticket", "How are you doing?", "Meeting scheduled for tomorrow", "Urgent account update required", "Can you send the report?", "Claim your prize now", "Dinner tonight?", "Special offer just for you", "See you later"],
                'label': ['spam', 'spam', 'ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
            }
            df = pd.DataFrame(data)

    # Prepare labels
    df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})
    print("Labels converted to numerical format.")
    return df


# --- Model Training ---
def train_model(df):
    """Splits data, trains TF-IDF vectorizer and Naive Bayes model."""
    # Split Data
    X = df['text']
    y = df['label_num']
    # Use a small test set just for reporting during training, app uses the full data essentially
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples (for metrics).")

    # Vectorize Text
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test) # Used only for evaluation metrics
    print("Text data vectorized using TF-IDF.")

    # Train Model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    print("Multinomial Naive Bayes model trained.")

    # Evaluate (optional printout)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])
    print("\n--- Model Evaluation (on initial split) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # For the web app, we want to fit the vectorizer on ALL data eventually
    # But for consistency with typical ML flow, we'll fit here on train
    # and the app will use this fitted vectorizer. A better approach might
    # involve saving the fitted objects after training on full data.
    # For simplicity, we return the model and vectorizer trained on the 80% split.
    # A more robust approach would be to retrain on ALL data after finding good params.
    # Or even better: save/load the trained objects.
    # Let's refit vectorizer and model on *all* data for the app to use.
    print("\nRefitting vectorizer and model on the full dataset for the web app...")
    full_X_tfidf = tfidf_vectorizer.fit_transform(df['text'])
    model.fit(full_X_tfidf, df['label_num'])
    print("Model and vectorizer refit on full dataset.")

    return model, tfidf_vectorizer


# --- Prediction ---
def predict_spam(text_input, model, vectorizer):
    """Predicts if a single text input is spam or ham."""
    if not isinstance(text_input, list):
        text_input = [text_input] # Vectorizer expects iterable
    text_tfidf = vectorizer.transform(text_input)
    prediction = model.predict(text_tfidf)
    prediction_proba = model.predict_proba(text_tfidf)

    is_spam = prediction[0] == 1
    label = 'spam' if is_spam else 'ham'
    confidence_ham = prediction_proba[0][0]
    confidence_spam = prediction_proba[0][1]

    return label, confidence_spam # Return label and spam confidence


# --- Get Random Email ---
def get_random_email(df=None):
    """Returns a random email message from the dataset."""
    if df is None:
        df = load_and_prepare_data()
    return random.choice(df['text'].tolist())


# --- Main Execution Block (for running script directly) ---
if __name__ == "__main__":
    print("Running spam_detector script directly...")
    dataframe = load_and_prepare_data()
    trained_model, fitted_vectorizer = train_model(dataframe)

    # Example Prediction (when run directly)
    print("\n--- Example Prediction (when run directly) ---")
    # example_emails = [
    #     "Hi mom, see you tomorrow! Hope you are well.",
    #     "Congratulations! You've won a $1000 gift card. Click here!"
    # ]
    # Use example from original script for consistency
    example_emails = ["Hi mom, see you tomorrow! Hope you are well."]

    for email in example_emails:
        pred_label, pred_conf_spam = predict_spam(email, trained_model, fitted_vectorizer)
        print(f"Email: '{email}'")
        print(f"Predicted label: {pred_label}")
        print(f"Spam Confidence: {pred_conf_spam:.4f}")
        print("---")

    # Example of getting a random email
    print("\n--- Random Email Example ---")
    random_email = get_random_email(dataframe)
    print(f"Random Email: {random_email}")
    print("Script execution finished.")

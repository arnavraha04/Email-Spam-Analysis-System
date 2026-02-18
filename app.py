from flask import Flask, request, render_template, url_for
import spam_detector
import sys # Import sys to exit cleanly after printing error

app = Flask(__name__) # Initialize Flask app

# --- Load Data, Model and Vectorizer on Startup ---
data = None
model = None
vectorizer = None
try:
    print("Attempting to load data and train model...")
    data = spam_detector.load_and_prepare_data()
    model, vectorizer = spam_detector.train_model(data)
    print("Model and vectorizer loaded successfully.")
except Exception as e:
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"CRITICAL ERROR during startup: {e}")
    print("Flask app cannot start due to error in spam_detector loading/training.")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    # Optionally print traceback for more detail
    import traceback
    traceback.print_exc()
    sys.exit(1) # Exit the script so Flask doesn't try to run in a broken state


# --- Define Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None # Restore initial value
    submitted_text = ""
    spam_confidence = 0.0
    # Removed basic print

    if request.method == 'POST':
        submitted_text = request.form.get('email_text', '')
        # Restore prediction logic
        if submitted_text:
            prediction_result, spam_confidence = spam_detector.predict_spam(
                submitted_text, model, vectorizer
            )
        else:
            submitted_text = ""

    elif request.method == 'GET':
        if request.args.get('random') == 'true':
            print("Random email requested!")
            # Restore random email logic
            submitted_text = spam_detector.get_random_email(data)
            print(f"Fetched random text (first 100 chars): {submitted_text[:100]}...")
        else:
            print("Standard GET request.")
            submitted_text = ""

    print(f"Rendering template with email_text: {submitted_text[:100]}...")
    return render_template(
        'index.html',
        prediction=prediction_result,
        email_text=submitted_text,
        spam_confidence=f"{spam_confidence:.2%}"
    )

# --- Run the App ---
if __name__ == '__main__':
    if model is None or vectorizer is None:
        print("Model or vectorizer failed to load. Cannot start server.")
    else:
        print("Starting Flask server with spam detection (port 5000) (debug disabled)...")
        # For reliable local testing we disable debug/reloader and bind to localhost only
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False) 
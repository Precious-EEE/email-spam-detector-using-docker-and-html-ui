import joblib
import sys

def load_model():
    model = joblib.load('spam_detector_model.pkl') 
    vectorizer = joblib.load('spam_detector_vector.pkl') 
    return model, vectorizer


def predict(text):
    model, vectorizer = load_model()
    text_transformed = vectorizer.transform([text])
    prediction = model.predict(text_transformed)
    return "Spam" if prediction[0] == 1 else 'Not Spam'

if __name__ == "__main__":
    text = sys.argv[0]
    prediction = predict(text)
    print (f"The message is : {prediction}")
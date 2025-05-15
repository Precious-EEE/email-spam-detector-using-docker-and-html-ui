from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load model and vectorizer once
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('spam_detector_vector.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # DO NOT prefix with "templates/"

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        # Handle API (JSON)
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        transformed = vectorizer.transform([text])
        prediction = model.predict(transformed)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        return jsonify({'prediction': result})
    else:
        # Handle Web form
        text = request.form.get('text', '')
        if not text:
            return render_template('index.html', prediction="Please enter some text.")
        transformed = vectorizer.transform([text])
        prediction = model.predict(transformed)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        return render_template('index.html', prediction=result, text=text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

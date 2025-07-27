from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form['email']
    data = [email]
    vect = vectorizer.transform(data)
    prediction = model.predict(vect)
    result = ' Not Spam' if prediction[0] == 1 else 'Spam'
    return render_template('index.html', prediction=result, email=email)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template, redirect, url_for, session
import numpy as np
import pickle
import os
import time

# Language for audio
language = 'en'

# Importing model and scalers
model = pickle.load(open("model.pkl", 'rb'))
ms = pickle.load(open("scaler.pkl", 'rb'))
le = pickle.load(open("label_encoder.pkl", 'rb'))

# Creating flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Dummy user database (for demonstration purposes)
users = {
    'jeevan': 'password1',
    'srirama': 'password2',
    'preethi31': 'passWord3',
    'preethi32': 'passWord4',
}

@app.route('/')
def index():
    if 'username' in session:
        return render_template("index.html")
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route("/predict", methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Fetch user inputs
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    # Prepare input features
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Scale input features and make prediction
    scaled_features = ms.transform(single_pred)
    prediction = model.predict(scaled_features)
    predicted_crop = le.inverse_transform(prediction)[0]

    # Generate result and image file path
    result = f"The best crop to be cultivated is {predicted_crop}"
    image_file = f"static/images/{predicted_crop.lower()}.jpg"
    
    # Render the result
    return render_template('index.html', result=result, image_file=image_file)

# Run the flask app
if __name__ == "__main__":
    app.run(debug=True)

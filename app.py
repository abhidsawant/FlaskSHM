from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

app = Flask(__name__)
SECRET_KEY = os.urandom(24).hex()  # Generate a random secret key
app.secret_key = os.getenv('SECRET_KEY', SECRET_KEY)

# Connect to MongoDB
client = MongoClient('mongodb://127.0.0.1:27017/')
db = client['user_database']
users = db['users']

# route to check the mongodb connection:
@app.route('/test_db')
def test_db():
    try:
        db.command('ping')
        return "MongoDB connection successful!"
    except Exception as e:
        return f"Failed to connect to MongoDB: {e}"


import joblib
import pandas as pd

# Load the model and scalers
model = joblib.load('model/rf_bridge_model.pkl')
scaler = joblib.load('model/scaler.pkl')
label_encoders = joblib.load('model/label_encoders.pkl')

# Categorical columns used in training
categorical_columns = ["Material", "Weather_Conditions", "Material_Composition", "Bridge_Design", "Construction_Quality"]

@app.route('/')
def home():
    if 'username' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid content type. Please use 'application/json'."}), 415

        data = request.get_json()
        new_df = pd.DataFrame([data])

        for col in categorical_columns:
            if col in new_df.columns and col in label_encoders:
                new_df[col] = label_encoders[col].transform(new_df[col])

        new_data_scaled = scaler.transform(new_df)
        prediction = model.predict(new_data_scaled)
        result = "Collapsed" if prediction[0] == 1 else "Standing"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})


# signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if users.find_one({'username': username}):
            flash('Username already exists. Please choose a different one.')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password)
        users.insert_one({'username': username, 'password': hashed_password})

        # Log the user in (store in session)
        session['username'] = username
        flash('Signup successful! Welcome to the home page.')
        return redirect(url_for('home'))

    return render_template('signup.html')

# login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            flash('Login successful! Welcome to the home page.')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password. Please try again.')

    return render_template('login.html')

# logout route:
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/profile')
def profile():
    if 'username' in session:
        return render_template('profile.html')
    else:
        flash('Please log in to view your profile.')
        return redirect(url_for('login'))

@app.route('/reviews')
def reviews():
    return render_template('reviews.html')

if __name__ == '__main__':
    app.run(debug=True)
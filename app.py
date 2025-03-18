from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import os
import joblib
import pandas as pd
import numpy as np

# Load .env file
load_dotenv()

app = Flask(__name__)
SECRET_KEY = os.urandom(24).hex()  # Generate a random secret key
app.secret_key = os.getenv('SECRET_KEY', SECRET_KEY)

# Connect to MongoDB
client = MongoClient('mongodb://127.0.0.1:27017/')
db = client['user_database']
users = db['users']

# Route to check the mongodb connection
@app.route('/test_db')
def test_db():
    try:
        db.command('ping')
        return "MongoDB connection successful!"
    except Exception as e:
        return f"Failed to connect to MongoDB: {e}"

# Load the model and related files
try:
    model = joblib.load('model/rf_bridge_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    label_encoders = joblib.load('model/label_encoders.pkl')
    selected_features = joblib.load('model/selected_features.pkl')
    
    # Try to load original column names if available
    try:
        original_column_names = joblib.load('model/original_column_names.pkl')
        print(f"Loaded original column names: {original_column_names}")
    except:
        original_column_names = selected_features
        print("Could not load original column names, using selected features instead")
    
    print(f"Model successfully loaded. Selected features: {selected_features}")
except Exception as e:
    print(f"Error loading model files: {e}")

@app.route('/')
def home():
    if 'username' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if request.content_type.startswith('multipart/form-data'):
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        try:
            # Read the uploaded file
            data = pd.read_csv(file)
            
            # Drop irrelevant columns if present
            data = data.drop(columns=["Bridge_ID", "Collapse_Status", "Location"], errors='ignore')
            
            # Create a column mapping to handle potential naming differences
            column_mapping = {}
            for col in data.columns:
                for orig_col in selected_features:
                    # Compare normalized column names (lowercase, without spaces/special chars)
                    col_norm = col.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("_m", "").replace("_c", "")
                    orig_norm = orig_col.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("_m", "").replace("_c", "")
                    
                    if col_norm == orig_norm or col_norm in orig_norm or orig_norm in col_norm:
                        column_mapping[col] = orig_col
                        break
            
            # Apply column mapping
            if column_mapping:
                data = data.rename(columns=column_mapping)
                print(f"Applied column mapping: {column_mapping}")
            
            # Check for missing features and add them with default values
            missing_cols = set(selected_features) - set(data.columns)
            if missing_cols:
                print(f"Adding missing columns with default values: {missing_cols}")
                for col in missing_cols:
                    data[col] = 0  # Default value
            
            # Select and reorder columns to match the training data
            try:
                data = data[selected_features]
            except KeyError as e:
                print(f"KeyError when selecting features: {e}")
                print(f"Available columns: {data.columns.tolist()}")
                print(f"Required columns: {selected_features}")
                return jsonify({'error': f'Column mismatch. Available: {data.columns.tolist()}, Required: {selected_features}'}), 400
            
            # Encode categorical variables
            for col in label_encoders.keys():
                if col in data.columns:
                    try:
                        data[col] = label_encoders[col].transform(data[col])
                    except Exception as e:
                        print(f"Error encoding column {col}: {e}")
                        # Handle unknown categories gracefully
                        data[col] = -1
            
            # Standardize numerical features
            try:
                data_scaled = scaler.transform(data)
            except Exception as e:
                print(f"Error in scaling: {e}")
                return jsonify({'error': f'Scaling error: {str(e)}'}), 500
            
            # Predict
            predictions = model.predict(data_scaled)
            probabilities = model.predict_proba(data_scaled)[:, 1]  # Probability of collapse
            
            # Prepare results
            results = []
            for i, pred in enumerate(predictions):
                results.append({
                    'prediction': 'Collapsed' if pred == 1 else 'Standing',
                    'collapse_probability': float(probabilities[i])
                })
            
            return jsonify({
                'success': True,
                'predictions': results
            })
            
        except Exception as e:
            import traceback
            print(f"Prediction error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    elif request.content_type == 'application/json':
        # Handle JSON input for single prediction
        try:
            input_data = request.json
            
            # Create DataFrame from JSON input
            df = pd.DataFrame([input_data])
            
            # Apply column mapping if needed
            column_mapping = {}
            for col in df.columns:
                for orig_col in selected_features:
                    col_norm = col.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("_m", "").replace("_c", "")
                    orig_norm = orig_col.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("_m", "").replace("_c", "")
                    
                    if col_norm == orig_norm or col_norm in orig_norm or orig_norm in col_norm:
                        column_mapping[col] = orig_col
                        break
            
            if column_mapping:
                df = df.rename(columns=column_mapping)
            
            # Add missing columns
            for col in selected_features:
                if col not in df.columns:
                    df[col] = 0
            
            # Select only required features in the right order
            df = df[selected_features]
            
            # Encode categorical variables
            for col in label_encoders.keys():
                if col in df.columns:
                    try:
                        df[col] = label_encoders[col].transform(df[col])
                    except:
                        df[col] = -1
            
            # Scale features
            df_scaled = scaler.transform(df)
            
            # Predict
            prediction = model.predict(df_scaled)[0]
            probability = model.predict_proba(df_scaled)[0][1]
            
            return jsonify({
                'success': True,
                'prediction': 'Collapsed' if prediction == 1 else 'Standing',
                'collapse_probability': float(probability)
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    else:
        return jsonify({'error': 'Invalid content type. Please use multipart/form-data or application/json.'}), 415


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
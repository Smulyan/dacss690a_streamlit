from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import requests
import gzip
from io import BytesIO
from flasgger import Swagger
from io import StringIO
import re


app = Flask(__name__)

# Swagger config
app.config['SWAGGER'] = {
    'title': 'Retraction Time Prediction',
    'uiversion': 3
}
swagger = Swagger(app)

# SQLite DB setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///retractions.db'
db = SQLAlchemy(app)

#define the database model
class Retraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # Unique ID for each record
    DaysToRetraction = db.Column(db.Integer, nullable=False)

    # Dummy variables for subject areas
    Subject_BT = db.Column(db.Integer, nullable=False, default=0)
    Subject_BLS = db.Column(db.Integer, nullable=False, default=0)
    Subject_ENV = db.Column(db.Integer, nullable=False, default=0)
    Subject_HSC = db.Column(db.Integer, nullable=False, default=0)
    Subject_HUM = db.Column(db.Integer, nullable=False, default=0)
    Subject_PHY = db.Column(db.Integer, nullable=False, default=0)
    Subject_SOC = db.Column(db.Integer, nullable=False, default=0)

    # Dummy variables for countries
    Country_China = db.Column(db.Integer, nullable=False, default=0)
    Country_United_States = db.Column(db.Integer, nullable=False, default=0)
    Country_India = db.Column(db.Integer, nullable=False, default=0)
    Country_Russia = db.Column(db.Integer, nullable=False, default=0)
    Country_Iran = db.Column(db.Integer, nullable=False, default=0)
    Country_United_Kingdom = db.Column(db.Integer, nullable=False, default=0)
    Country_Japan = db.Column(db.Integer, nullable=False, default=0)
    Country_Saudi_Arabia = db.Column(db.Integer, nullable=False, default=0)
    Country_South_Korea = db.Column(db.Integer, nullable=False, default=0)
    Country_Germany = db.Column(db.Integer, nullable=False, default=0)

    #Paywall status
    Paywalled = db.Column(db.Integer, nullable=False, default=0)

# Create the database
with app.app_context():
    db.create_all()

def preprocess_data(df):
    #drop all columns except indicated
    retractions = df[['Subject', 'Country', 'RetractionDate','OriginalPaperDate', 'RetractionNature', 'Paywalled']]

    #Calculate length of time between publication and retraction

    # Ensure both columns are in datetime format
    retractions['OriginalPaperDate'] = pd.to_datetime(retractions['OriginalPaperDate'], format='%m/%d/%Y %H:%M', errors='coerce')
    retractions['RetractionDate'] = pd.to_datetime(retractions['RetractionDate'], format='%m/%d/%Y %H:%M', errors='coerce')

    # Add new column with the difference in days
    retractions['DaysToRetraction'] = (retractions['RetractionDate'] - retractions['OriginalPaperDate']).dt.days

    # Drop the OriginalPaperDate and RetractionDate columns
    retractions = retractions.drop(columns=['OriginalPaperDate', 'RetractionDate'])

    #drop rows where DaysToRetraction is 0
    retractions = retractions.drop(retractions[retractions['DaysToRetraction'] == 0].index)

    #drop the rows where Retraction Nature is anything other than 'Retraction'

    retractions = retractions.drop(retractions[retractions['RetractionNature'] != 'Retraction'].index)

    #then drop the RetractionNature column
    retractions = retractions.drop(columns=['RetractionNature'])

    #Drop rows where Paywalled = 'Unknown'
    retractions = retractions.drop(retractions[retractions['Paywalled'] == 'Unknown'].index)

    #Reduce the Subject values to just their high-level codes
    # B/T (Busines and technology), BLS (Biology and Life Sciences), ENV (Environmental Sciences), HSC (Health Sciences), HUM (Humanities), PHY (Physical Sciences), and SOC (Social Sciences)
    retractions['Subject'] = retractions['Subject'].apply(lambda x: '; '.join(re.findall(r'\(([^)]+)\)', x)))

    #split subject codes into individual columns; use those to create dummy varibles for each subject
    subject_codes = retractions['Subject'].str.split('; ', expand=True)
    subject_dummies = pd.get_dummies(subject_codes.stack()).groupby(level=0).sum()
    subject_dummies.columns = ['Subject_' + col for col in subject_dummies.columns]
    retractions = pd.concat([retractions, subject_dummies], axis=1)
    retractions = retractions.drop(columns=['Subject'])

    #same maneuver for Countries - convert to dummy variables, for just the 10 most frequent countries
    country_codes = retractions['Country'].str.split(';', expand=True)
    country_dummies = pd.get_dummies(country_codes.stack()).groupby(level=0).sum()
    top_10_countries = country_dummies.sum().sort_values(ascending=False).head(10).index
    country_dummies = country_dummies[top_10_countries]
    country_dummies.columns = ['Country_' + col.strip() for col in country_dummies.columns]
    retractions = pd.concat([retractions, country_dummies], axis=1)
    retractions = retractions.drop(columns=['Country'])

    #drop rows with no country in top 10
    country_columns = [col for col in retractions.columns if col.startswith('Country_')]
    retractions = retractions[retractions[country_columns].any(axis=1)]

    #Convert Paywalled values 'No' and 'Yes' to 0 and 1 respectively
    retractions['Paywalled'] = retractions['Paywalled'].map({'No': 0, 'Yes': 1})

    # Drop any rows that still have NaN values at this point (forcefully)
    retractions = retractions.dropna()

    return retractions

# Global variables for model and encoder
model = None

@app.route('/reload', methods=['POST'])
def reload_data():
    '''
    Reload data from the Retraction Watch dataset, clear the database, load new data, and return summary stats
    ---
    responses:
      200:
        description: Summary statistics of reloaded data
    '''
    global model

    # Step 1: Download and decompress data
    url = "https://gitlab.com/crossref/retraction-watch-data/-/raw/main/retraction_watch.csv"
    response = requests.get(url)

    # Step 2: Load data into pandas
    rw_file = StringIO(response.text)  # Convert text response to a file-like object
    retractions = pd.read_csv(rw_file)

    # Step 3: Clear the database
    db.session.query(Retraction).delete()

    #Step 4: Process/Clean data
    df = preprocess_data(retractions)

    # Step 5: Insert data into the database

    for _, row in df.iterrows():
        new_retraction = Retraction(
            DaysToRetraction=row['DaysToRetraction'],
            Subject_BT=row['Subject_B/T'],
            Subject_BLS=row['Subject_BLS'],
            Subject_ENV=row['Subject_ENV'],
            Subject_HSC=row['Subject_HSC'],
            Subject_HUM=row['Subject_HUM'],
            Subject_PHY=row['Subject_PHY'],
            Subject_SOC=row['Subject_SOC'],
            Country_China=row['Country_China'],
            Country_United_States=row['Country_United States'],
            Country_India=row['Country_India'],
            Country_Russia=row['Country_Russia'],
            Country_Iran=row['Country_Iran'],
            Country_United_Kingdom=row['Country_United Kingdom'],
            Country_Japan=row['Country_Japan'],
            Country_Saudi_Arabia=row['Country_Saudi Arabia'],
            Country_South_Korea=row['Country_South Korea'],
            Country_Germany=row['Country_Germany'],
            Paywalled=row['Paywalled']
        )
        db.session.add(new_retraction)

    db.session.commit()

    # Step 6: Train model
    X = df.drop(columns='DaysToRetraction')
    y = df['DaysToRetraction']
    model = LinearRegression()
    model.fit(X, y)

    # Step 6: Generate summary statistics
    summary = {
        'total_retractions': len(df),
        'average_days_to_retraction': df['DaysToRetraction'].mean(),
        'min_days_to_retraction': df['DaysToRetraction'].min(),
        'max_days_to_retraction': df['DaysToRetraction'].max(),
        'most_common_subjects': df[[col for col in retractions.columns if col.startswith('Subject_')]].sum().nlargest(5).to_dict(),
        'most_common_countries': df[[col for col in retractions.columns if col.startswith('Country_')]].sum().nlargest(5).to_dict(),
        'paywalled_distribution': df['Paywalled'].value_counts().to_dict()
    }

    return jsonify(summary)

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the retraction time for a publiction
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            subject:
              type: string
              description: "The publication's subject category -  B/T (Busines and technology), BLS (Biology and Life Sciences), ENV (Environmental Sciences), HSC (Health Sciences), HUM (Humanities), PHY (Physical Sciences), or SOC (Social Sciences)"
            country:
              type: string
              description: "The country of authors' affilated institution - China, United States, India, Russia, Iran, United Kingdom, Japan, Saudi Arabia, South Korea, or Germany "
            paywalled:
              type: string
              enum: ["Yes", "No"]
              description: "Indicates if the publication was behind a paywall ('Yes' or 'No')"
    responses:
      200:
        description: Predicted days to retraction
    '''
    global model  # Ensure that the model is available for prediction

    # Define the list of valid subjects
    valid_subjects = [
        "B/T", "BLS", "ENV", "HSC", "HUM", "PHY", "SOC"
    ]

    #And, likewise, list of valid Countries
    valid_countries = [
        "China", "United States", "India", "Russia", "Iran", "United Kingdom", "Japan", "Saudi Arabia", "South Korea", "Germany"
    ]

    # Check if the model is initialized
    if model is None:
        return jsonify({"error": "The data has not been loaded. Please refresh the data by calling the '/reload' endpoint first."}), 400

    data = request.json
    try:
        # Extract inputs
        subject = data.get('subject')
        country = data.get('country')
        paywalled = data.get('paywalled')

        # Validate inputs
        if subject not in valid_subjects:
            return jsonify({"error": f"Invalid subject. Choose from: {', '.join(valid_subjects)}"}), 400
        if country not in valid_countries:
            return jsonify({"error": f"Invalid country. Choose from: {', '.join(valid_countries)}"}), 400
        if paywalled not in ["Yes", "No"]:
            return jsonify({"error": "Paywalled must be 'Yes' or 'No'"}), 400

        # Prepare input data (match model feature structure)
        input_data = {col: 0 for col in model.feature_names_in_}

        # Set the correct subject and country variables
        subject_col = f"Subject_{subject}"
        country_col = f"Country_{country}"

        if subject_col in input_data:
            input_data[subject_col] = 1
        if country_col in input_data:
            input_data[country_col] = 1

        # Convert paywalled to binary
        input_data["Paywalled"] = 1 if str(paywalled).strip().lower() == "yes" else 0

        # Convert dictionary to NumPy array
        input_df = pd.DataFrame([input_data])
        input_array = input_df.to_numpy()

        # Predict days to retraction
        predicted_days = model.predict(input_array)[0]

        return jsonify({"predicted_days_to_retraction": predicted_days})


    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

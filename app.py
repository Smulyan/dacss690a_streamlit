from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import requests
from io import StringIO
import re
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

#global model variable
model = None

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





@st.cache_data
def load_data():

    # Step 1: Download and decompress data
    url = "https://gitlab.com/crossref/retraction-watch-data/-/raw/main/retraction_watch.csv"
    response = requests.get(url)

    # Step 2: Load data into pandas
    rw_file = StringIO(response.text)  # Convert text response to a file-like object
    raw_df = pd.read_csv(rw_file)

    #Step 3: Process/Clean data
    df = preprocess_data(raw_df)

    # Step 4: Train regression model
    X = df.drop(columns='DaysToRetraction')
    y = df['DaysToRetraction']
    model = LinearRegression()
    model.fit(X, y)

    # Step 5: Generate summary statistics
    summary = {
        'total_retractions': len(df),
        'average_days_to_retraction': df['DaysToRetraction'].mean(),
        'min_days_to_retraction': df['DaysToRetraction'].min(),
        'max_days_to_retraction': df['DaysToRetraction'].max(),
        'most_common_subjects': df[[col for col in df.columns if col.startswith('Subject_')]].sum().nlargest(5).to_dict(),
        'most_common_countries': df[[col for col in df.columns if col.startswith('Country_')]].sum().nlargest(5).to_dict(),
        'paywalled_distribution': df['Paywalled'].value_counts().to_dict()
    }

    return df, model, summary


def predict_ui():
    st.subheader("Predict Days to Retraction")

    global model
    if model is None:
        st.error("Model not trained. Please reload data.")

    # Define the list of valid subjects
    valid_subjects = [
        "B/T", "BLS", "ENV", "HSC", "HUM", "PHY", "SOC"
    ]

    #And, likewise, list of valid Countries
    valid_countries = [
        "China", "United States", "India", "Russia", "Iran", "United Kingdom", "Japan", "Saudi Arabia", "South Korea", "Germany"
    ]

    with st.form("prediction_form"):
        subject = st.selectbox("Subject Area", valid_subjects)
        country = st.selectbox("Country", valid_countries)
        paywalled = st.radio("Paywalled?", ["Yes", "No"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            # Create input template with 0s for all model features
            input_data = {col: 0 for col in model.feature_names_in_}

            subject_col = f"Subject_{subject}"
            country_col = f"Country_{country}"

            if subject_col in input_data:
                input_data[subject_col] = 1
            if country_col in input_data:
                input_data[country_col] = 1

            input_data["Paywalled"] = 1 if paywalled == "Yes" else 0

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Make prediction
            predicted_days = model.predict(input_df)[0]
            st.success(f"📅 Predicted Days to Retraction: **{int(predicted_days)} days**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


def viz_ui(df):
    st.subheader("Visualize Retraction Time")

    subject_cols = [col for col in df.columns if col.startswith("Subject_")]
    country_cols = [col for col in df.columns if col.startswith("Country_")]

    selected_subjects = st.multiselect("Subjects", subject_cols, default=subject_cols)
    selected_countries = st.multiselect("Countries", country_cols, default=country_cols)
    paywall_option = st.selectbox("Paywall Status", ["All", "Yes", "No"])

    filtered = df[
        (df[selected_subjects].sum(axis=1) > 0) &
        (df[selected_countries].sum(axis=1) > 0)
        ]
    if paywall_option != "All":
        filtered = filtered[filtered["Paywalled"] == (1 if paywall_option == "Yes" else 0)]

    st.write(f"Filtered rows: {len(filtered)}")

    plot_type = st.radio("Plot Type", ["Boxplot", "Histogram"])

    fig, ax = plt.subplots()
    if plot_type == "Boxplot":
        sns.boxplot(data=filtered, y="DaysToRetraction", ax=ax)
    else:
        sns.histplot(data=filtered, x="DaysToRetraction", bins=30, kde=True, ax=ax)

    st.pyplot(fig)


#main streamlit app
st.set_page_config(layout="wide")
st.title("Retraction Watch Data Explorer")

data = load_data()

tab1, tab2 = st.tabs(["Prediction", "Visualization"])
with tab1:
    predict_ui(data)
with tab2:
    viz_ui(data)
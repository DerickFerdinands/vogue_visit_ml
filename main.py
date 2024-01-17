import random

from fastapi import FastAPI, HTTPException, Depends
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from typing import List
from faker import Faker
import pandas as pd
import datetime
app = FastAPI()

# Load the dataset
# Replace this with your actual dataset or data loading logic
# For simplicity, let's assume a sample DataFrame df

# Initialize Faker
fake = Faker()

# Define number of records
num_records = 10000
number_of_peoples = 1000
number_of_days = 30
peoples = [fake.name() for _ in range(number_of_peoples)]
salon_services = [
  "Haircut",
  "Hair Coloring",
  "Hair Styling",
  "Manicure",
  "Pedicure",
  "Facial",
  "Waxing",
  "Massage",
  "Makeup",
  "Spa Packages"
];

genders = ["Male", "Female", "Custom"]
purchase_dates = [fake.date_between(start_date=datetime.datetime(2024,1,1), end_date='today') for _ in range(number_of_days)]

data = {
    'Name': [random.choice(peoples) for _ in range(num_records)],
    'service': [random.choice(salon_services) for _ in range(num_records)],
    'Age': [random.randint(16, 70) for _ in range(num_records)],
    'Date':[random.choice(purchase_dates) for _ in range(num_records)],
    'Gender': [random.choice(genders) for _ in range(num_records)],
}

# Update DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the DataFrame with the new columns
df.head()

# Create a Surprise dataset
reader = Reader(rating_scale=(0, 1))  # Assuming binary ratings (0: not purchased, 1: purchased)
data = Dataset.load_from_df(df[['Name', 'service', 'Age']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Create a collaborative filtering model (SVD algorithm)
model = SVD()

# Train the model on the training set
model.fit(trainset)


def get_recommendations(user_name: str, N: int = 5):
    # Check if the user exists in the dataset
    if user_name in df['Name'].values:
        user_data = df[df['Name'] == user_name][['Name', 'service', 'Purchased']]
        # Convert user data to Surprise format
        user_data_surprise = [tuple(x) for x in user_data.values]
        # Get the top N recommendations for the user
        top_n = sorted(model.test(user_data_surprise), key=lambda x: x.est, reverse=True)[:N]
    else:
        # Use other customers' data for suggestions
        all_items = df['service'].unique()
        testset = [(user_name, item, 0) for item in all_items]
        predictions = model.test(testset)
        top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:N]

    # Extract the service names from recommendations
    recommendations = [pred.iid for pred in top_n]

    return recommendations


@app.get("/recommendations/{user_name}")
def get_user_recommendations(user_name: str, N: int = 5):
    recommendations = get_recommendations(user_name, N)
    return {"user_name": user_name, "recommendations": recommendations}


@app.post("/purchase")
def purchase_service(user_name: str, service_name: str):
    # Assume service is purchased (binary, 1)
    # Add the purchase to the DataFrame or your data storage logic
    df = df.append({'Name': user_name, 'service': service_name, 'Purchased': 1}, ignore_index=True)

    # Re-train the model with the updated data
    data = Dataset.load_from_df(df[['Name', 'service', 'Purchased']], reader)
    trainset, _ = train_test_split(data, test_size=0.0)  # Use the entire dataset for training
    model.fit(trainset)

    return {"message": f"Service '{service_name}' purchased by {user_name}. Model updated."}

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess(path='data/india_housing_prices.csv'):
    df = pd.read_csv(path)
    print(f"Raw shape: {df.shape}")

    # ── 1. Drop duplicates ──────────────────────────────────────────────
    df.drop_duplicates(inplace=True)

    # ── 2. Handle missing values ────────────────────────────────────────
    df['Furnished_Status'].fillna(df['Furnished_Status'].mode()[0], inplace=True)
    df['Facing'].fillna(df['Facing'].mode()[0], inplace=True)
    df['Amenities'].fillna(df['Amenities'].mode()[0], inplace=True)
    df['Floor_No'].fillna(df['Floor_No'].median(), inplace=True)
    df['Nearby_Schools'].fillna(df['Nearby_Schools'].median(), inplace=True)

    # ── 3. Feature engineering ──────────────────────────────────────────
    df['Age_of_Property'] = 2024 - df['Year_Built']
    df['Floor_Ratio'] = df['Floor_No'] / (df['Total_Floors'] + 1)
    df['School_Density_Score'] = df['Nearby_Schools'] * 10 + df['Nearby_Hospitals'] * 8
    df['Infra_Score'] = (
        df['Public_Transport_Accessibility'] * 0.4 +
        df['Nearby_Schools'] * 0.3 +
        df['Nearby_Hospitals'] * 0.3
    )
    df['Amenity_Score'] = df['Amenities'].map({
        'All': 5, 'Gym+Pool': 4, 'Pool': 3, 'Gym': 2,
        'Clubhouse': 2, 'None': 0
    }).fillna(1)
    df['RERA_Ready'] = (df['Availability_Status'] == 'Available').astype(int)

    # ── 4. Target variables ─────────────────────────────────────────────
    # Regression target: future price after 5 years (feature-based growth)
    growth_rate = 0.08 + (df['Infra_Score'] / 100)
    df['Future_Price_5Y'] = df['Price_in_Lakhs'] * ((1 + growth_rate) ** 5)
    df['Future_Price_5Y'] = df['Future_Price_5Y'].round(2)

    # Classification target: Good Investment (multi-factor)
    city_median = df.groupby('City')['Price_per_SqFt'].transform('median')
    score = (
        (df['Price_per_SqFt'] <= city_median).astype(int) * 2 +
        (df['BHK'] >= 3).astype(int) +
        (df['RERA_Ready'] == 1).astype(int) +
        (df['Infra_Score'] >= df['Infra_Score'].median()).astype(int) +
        (df['Amenity_Score'] >= 3).astype(int)
    )
    df['Good_Investment'] = (score >= 3).astype(int)

    # ── 5. Encode categoricals ──────────────────────────────────────────
    le_cols = ['State', 'City', 'Locality', 'Property_Type', 'Furnished_Status',
               'Facing', 'Owner_Type', 'Availability_Status', 'Security', 'Amenities']
    label_encoders = {}
    for col in le_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    os.makedirs('models', exist_ok=True)
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)

    print(f"Processed shape: {df.shape}")
    print(f"Good Investment ratio: {df['Good_Investment'].mean():.2%}")
    return df, label_encoders


def get_feature_cols():
    return [
        'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt',
        'Age_of_Property', 'Floor_No', 'Total_Floors', 'Floor_Ratio',
        'Nearby_Schools', 'Nearby_Hospitals', 'Public_Transport_Accessibility',
        'Parking_Space', 'Infra_Score', 'School_Density_Score', 'Amenity_Score',
        'RERA_Ready',
        'State_enc', 'City_enc', 'Locality_enc', 'Property_Type_enc',
        'Furnished_Status_enc', 'Facing_enc', 'Owner_Type_enc',
        'Availability_Status_enc', 'Security_enc', 'Amenities_enc'
    ]


if __name__ == '__main__':
    df, _ = load_and_preprocess()
    df.to_csv('data/processed_data.csv', index=False)
    print("Saved processed_data.csv")

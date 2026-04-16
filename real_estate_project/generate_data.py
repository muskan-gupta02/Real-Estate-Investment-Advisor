import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 2000

states = ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Telangana', 'Gujarat', 'Rajasthan', 'West Bengal']
city_map = {
    'Maharashtra': ['Mumbai', 'Pune', 'Nagpur'],
    'Karnataka': ['Bangalore', 'Mysore', 'Hubli'],
    'Tamil Nadu': ['Chennai', 'Coimbatore', 'Madurai'],
    'Delhi': ['New Delhi', 'Dwarka', 'Rohini'],
    'Telangana': ['Hyderabad', 'Warangal', 'Karimnagar'],
    'Gujarat': ['Ahmedabad', 'Surat', 'Vadodara'],
    'Rajasthan': ['Jaipur', 'Jodhpur', 'Udaipur'],
    'West Bengal': ['Kolkata', 'Howrah', 'Siliguri'],
}
locality_options = ['Koramangala', 'Banjara Hills', 'Andheri', 'Whitefield', 'Salt Lake',
                    'Gomti Nagar', 'Satellite', 'Jubilee Hills', 'Powai', 'Electronic City',
                    'HSR Layout', 'Indiranagar', 'Thane', 'Gachibowli', 'Kondapur']
property_types = ['Apartment', 'Villa', 'House', 'Penthouse', 'Studio']
furnished = ['Unfurnished', 'Semi-Furnished', 'Fully Furnished']
facing_dirs = ['North', 'South', 'East', 'West', 'North-East', 'South-West']
owner_types = ['Individual', 'Builder', 'Agent']
avail_status = ['Available', 'Under Construction', 'Sold']
security_opts = ['Gated', 'CCTV', 'Guard', 'None']
amenities_opts = ['Gym', 'Pool', 'Clubhouse', 'Gym+Pool', 'All', 'None']

records = []
for i in range(n):
    state = np.random.choice(states)
    city = np.random.choice(city_map[state])
    locality = np.random.choice(locality_options)
    prop_type = np.random.choice(property_types, p=[0.5, 0.15, 0.2, 0.08, 0.07])
    bhk = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.3, 0.35, 0.18, 0.07])
    size = int(np.random.normal(loc=bhk * 450, scale=150))
    size = max(300, min(size, 6000))

    base_price = size * np.random.uniform(3.5, 12.0)
    if city in ['Mumbai', 'Bangalore', 'New Delhi', 'Hyderabad']:
        base_price *= np.random.uniform(1.3, 1.8)
    elif city in ['Pune', 'Chennai', 'Ahmedabad']:
        base_price *= np.random.uniform(1.0, 1.3)

    price_in_lakhs = round(base_price / 100, 2)
    price_per_sqft = round((price_in_lakhs * 100000) / size, 2)
    year_built = np.random.randint(1990, 2023)
    age = 2024 - year_built
    floor_no = np.random.randint(0, 25)
    total_floors = floor_no + np.random.randint(1, 10)
    nearby_schools = np.random.randint(0, 10)
    nearby_hospitals = np.random.randint(0, 8)
    transport = np.random.randint(1, 10)
    parking = np.random.randint(0, 4)
    furn = np.random.choice(furnished, p=[0.3, 0.4, 0.3])
    facing = np.random.choice(facing_dirs)
    owner = np.random.choice(owner_types, p=[0.4, 0.35, 0.25])
    avail = np.random.choice(avail_status, p=[0.55, 0.3, 0.15])
    security = np.random.choice(security_opts)
    amenities = np.random.choice(amenities_opts)

    records.append({
        'ID': i + 1,
        'State': state,
        'City': city,
        'Locality': locality,
        'Property_Type': prop_type,
        'BHK': bhk,
        'Size_in_SqFt': size,
        'Price_in_Lakhs': price_in_lakhs,
        'Price_per_SqFt': price_per_sqft,
        'Year_Built': year_built,
        'Furnished_Status': furn,
        'Floor_No': floor_no,
        'Total_Floors': total_floors,
        'Age_of_Property': age,
        'Nearby_Schools': nearby_schools,
        'Nearby_Hospitals': nearby_hospitals,
        'Public_Transport_Accessibility': transport,
        'Parking_Space': parking,
        'Security': security,
        'Amenities': amenities,
        'Facing': facing,
        'Owner_Type': owner,
        'Availability_Status': avail,
    })

df = pd.DataFrame(records)

# Introduce missing values (~5%)
for col in ['Furnished_Status', 'Floor_No', 'Nearby_Schools', 'Facing', 'Amenities']:
    mask = np.random.rand(n) < 0.05
    df.loc[mask, col] = np.nan

os.makedirs('data', exist_ok=True)
df.to_csv('data/india_housing_prices.csv', index=False)
print(f"Dataset created: {df.shape[0]} rows x {df.shape[1]} cols")
print(df.head(3))

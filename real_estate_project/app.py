import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size:2.2rem; font-weight:800; color:#1a3c5e; margin-bottom:0; }
    .sub-title  { font-size:1rem;  color:#546e7a; margin-bottom:1.5rem; }
    .metric-box { background:#f0f7ff; border-radius:12px; padding:18px 22px;
                  text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.07); }
    .metric-val { font-size:2rem; font-weight:700; color:#1a3c5e; }
    .metric-lbl { font-size:0.85rem; color:#546e7a; margin-top:4px; }
    .good-badge { background:#e8f5e9; color:#2e7d32; border-radius:20px;
                  padding:6px 18px; font-weight:700; font-size:1.1rem; }
    .bad-badge  { background:#ffebee; color:#c62828; border-radius:20px;
                  padding:6px 18px; font-weight:700; font-size:1.1rem; }
    .section-hdr{ font-size:1.15rem; font-weight:700; color:#1a3c5e;
                  border-left:4px solid #1a3c5e; padding-left:10px; margin-bottom:10px; }
    hr.divider  { border:0; border-top:1px solid #e0e0e0; margin:1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Load models & encoders ───────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('models/best_classifier.pkl', 'rb') as f:
        cls_bundle = pickle.load(f)
    with open('models/best_regressor.pkl', 'rb') as f:
        reg_bundle = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('models/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return cls_bundle, reg_bundle, scaler, le, feature_cols


@st.cache_data
def load_data():
    return pd.read_csv('data/processed_data.csv')


# ── Helper: encode a single row ──────────────────────────────────────────────
def encode_input(row_dict, le):
    cat_cols = ['State', 'City', 'Locality', 'Property_Type', 'Furnished_Status',
                'Facing', 'Owner_Type', 'Availability_Status', 'Security', 'Amenities']
    for col in cat_cols:
        enc_key = col + '_enc'
        val = str(row_dict.get(col, ''))
        encoder = le[col]
        if val in encoder.classes_:
            row_dict[enc_key] = int(encoder.transform([val])[0])
        else:
            row_dict[enc_key] = 0
    return row_dict


def build_feature_vector(row_dict, feature_cols):
    vec = []
    for col in feature_cols:
        vec.append(row_dict.get(col, 0))
    return np.array(vec).reshape(1, -1)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  – property input form
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Enter Property Details")
    st.markdown("---")

    state = st.selectbox("State", [
        'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi',
        'Telangana', 'Gujarat', 'Rajasthan', 'West Bengal'
    ])
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
    city = st.selectbox("City", city_map[state])
    locality = st.selectbox("Locality", [
        'Koramangala', 'Banjara Hills', 'Andheri', 'Whitefield', 'Salt Lake',
        'Gomti Nagar', 'Satellite', 'Jubilee Hills', 'Powai', 'Electronic City',
        'HSR Layout', 'Indiranagar', 'Thane', 'Gachibowli', 'Kondapur'
    ])
    prop_type = st.selectbox("Property Type", ['Apartment', 'Villa', 'House', 'Penthouse', 'Studio'])
    bhk = st.slider("BHK", 1, 5, 2)
    size = st.number_input("Size (SqFt)", min_value=200, max_value=10000, value=1200, step=50)
    price = st.number_input("Current Price (Lakhs)", min_value=5.0, max_value=5000.0, value=80.0, step=5.0)

    st.markdown("---")
    year_built = st.slider("Year Built", 1980, 2024, 2010)
    floor_no = st.slider("Floor Number", 0, 40, 3)
    total_floors = st.slider("Total Floors", floor_no + 1, 50, max(floor_no + 5, 10))
    furnished = st.selectbox("Furnished Status", ['Unfurnished', 'Semi-Furnished', 'Fully Furnished'])
    facing = st.selectbox("Facing", ['North', 'South', 'East', 'West', 'North-East', 'South-West'])

    st.markdown("---")
    schools = st.slider("Nearby Schools", 0, 10, 3)
    hospitals = st.slider("Nearby Hospitals", 0, 8, 2)
    transport = st.slider("Transport Accessibility (1–10)", 1, 10, 5)
    parking = st.slider("Parking Spaces", 0, 4, 1)
    security = st.selectbox("Security", ['Gated', 'CCTV', 'Guard', 'None'])
    amenities = st.selectbox("Amenities", ['All', 'Gym+Pool', 'Pool', 'Gym', 'Clubhouse', 'None'])
    owner = st.selectbox("Owner Type", ['Individual', 'Builder', 'Agent'])
    avail = st.selectbox("Availability", ['Available', 'Under Construction', 'Sold'])

    predict_btn = st.button("🔍 Analyze Property", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🏠 Real Estate Investment Advisor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">ML-powered property profitability & future value predictor</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Market Insights", "🗂️ Dataset Explorer"])

with tab1:
    if predict_btn:
        try:
            cls_bundle, reg_bundle, scaler, le, feature_cols = load_artifacts()

            # Build derived features matching preprocess.py
            age = 2024 - year_built
            floor_ratio = floor_no / (total_floors + 1)
            infra_score = transport * 0.4 + schools * 0.3 + hospitals * 0.3
            school_density = schools * 10 + hospitals * 8
            amenity_map = {'All': 5, 'Gym+Pool': 4, 'Pool': 3, 'Gym': 2, 'Clubhouse': 2, 'None': 0}
            amenity_score = amenity_map.get(amenities, 1)
            rera_ready = 1 if avail == 'Available' else 0
            price_per_sqft = (price * 100000) / size

            row = {
                'BHK': bhk, 'Size_in_SqFt': size,
                'Price_in_Lakhs': price, 'Price_per_SqFt': price_per_sqft,
                'Age_of_Property': age, 'Floor_No': floor_no,
                'Total_Floors': total_floors, 'Floor_Ratio': floor_ratio,
                'Nearby_Schools': schools, 'Nearby_Hospitals': hospitals,
                'Public_Transport_Accessibility': transport,
                'Parking_Space': parking, 'Infra_Score': infra_score,
                'School_Density_Score': school_density,
                'Amenity_Score': amenity_score, 'RERA_Ready': rera_ready,
                'State': state, 'City': city, 'Locality': locality,
                'Property_Type': prop_type, 'Furnished_Status': furnished,
                'Facing': facing, 'Owner_Type': owner,
                'Availability_Status': avail, 'Security': security,
                'Amenities': amenities,
            }
            row = encode_input(row, le)
            X_vec = build_feature_vector(row, feature_cols)

            # Classification
            cls_model = cls_bundle['model']
            cls_scaled = cls_bundle.get('scaled', False)
            X_cls = scaler.transform(X_vec) if cls_scaled else X_vec
            cls_pred = int(cls_model.predict(X_cls)[0])
            cls_proba = float(cls_model.predict_proba(X_cls)[0][cls_pred])

            # Regression
            reg_model = reg_bundle['model']
            reg_scaled = reg_bundle.get('scaled', False)
            X_reg = scaler.transform(X_vec) if reg_scaled else X_vec
            future_price = float(reg_model.predict(X_reg)[0])
            appreciation = ((future_price - price) / price) * 100

            # ── Results ──────────────────────────────────────────────
            st.markdown('<div class="section-hdr">🎯 Prediction Results</div>', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                badge = '<span class="good-badge">✅ Good Investment</span>' if cls_pred else '<span class="bad-badge">❌ Poor Investment</span>'
                st.markdown(f'<div class="metric-box"><div class="metric-val">{badge}</div>'
                            f'<div class="metric-lbl">Investment Decision</div></div>',
                            unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-box"><div class="metric-val">{cls_proba*100:.1f}%</div>'
                            f'<div class="metric-lbl">Model Confidence</div></div>',
                            unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-box"><div class="metric-val">₹{future_price:.1f}L</div>'
                            f'<div class="metric-lbl">Estimated Price (5 Yrs)</div></div>',
                            unsafe_allow_html=True)
            with col4:
                color = "#2e7d32" if appreciation > 0 else "#c62828"
                st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:{color}">'
                            f'+{appreciation:.1f}%</div>'
                            f'<div class="metric-lbl">Expected Appreciation</div></div>',
                            unsafe_allow_html=True)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            # ── Feature importance (if tree model) ───────────────────
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="section-hdr">🔑 Feature Importance (Classifier)</div>', unsafe_allow_html=True)
                if hasattr(cls_model, 'feature_importances_'):
                    imp = pd.Series(cls_model.feature_importances_, index=feature_cols)
                    top10 = imp.sort_values(ascending=True).tail(12)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    top10.plot(kind='barh', ax=ax, color='steelblue')
                    ax.set_title('Top Features', fontsize=11)
                    ax.set_xlabel('Importance')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Feature importance not available for this model type.")

            with col_b:
                st.markdown('<div class="section-hdr">📈 Price Projection over 5 Years</div>', unsafe_allow_html=True)
                years = list(range(6))
                growth = 0.08 + (infra_score / 100)
                prices_proj = [round(price * ((1 + growth) ** y), 2) for y in years]
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.plot(years, prices_proj, marker='o', linewidth=2.5, color='darkorange')
                ax2.fill_between(years, prices_proj, alpha=0.15, color='darkorange')
                ax2.set_title('Projected Property Value (Lakhs)', fontsize=11)
                ax2.set_xlabel('Years from Now')
                ax2.set_ylabel('Price (Lakhs)')
                for yr, pr in zip(years, prices_proj):
                    ax2.annotate(f'₹{pr:.0f}L', (yr, pr), textcoords="offset points",
                                 xytext=(0, 8), ha='center', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig2)

        except FileNotFoundError:
            st.error("⚠️ Models not found. Please run `train_models.py` first.")
        except Exception as ex:
            st.error(f"Prediction error: {ex}")
    else:
        st.info("👈 Fill in property details in the sidebar and click **Analyze Property**.")


with tab2:
    st.markdown('<div class="section-hdr">📈 Market Insights</div>', unsafe_allow_html=True)
    try:
        df = load_data()

        c1, c2 = st.columns(2)
        with c1:
            # City-wise avg price
            city_avg = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(12)
            fig, ax = plt.subplots(figsize=(7, 4))
            city_avg.plot(kind='bar', color='cornflowerblue', ax=ax)
            ax.set_title('Avg Property Price by City (Lakhs)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Avg Price (Lakhs)')
            plt.xticks(rotation=40, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)

        with c2:
            # Good Investment ratio by city
            gi_city = df.groupby('City')['Good_Investment'].mean().sort_values(ascending=False).head(12) * 100
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            colors = ['#4CAF50' if v >= 50 else '#EF5350' for v in gi_city.values]
            gi_city.plot(kind='bar', color=colors, ax=ax2)
            ax2.set_title('Good Investment Rate by City (%)', fontsize=11, fontweight='bold')
            ax2.set_ylabel('% Good Investments')
            plt.xticks(rotation=40, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig2)

        c3, c4 = st.columns(2)
        with c3:
            # Price per sqft by property type
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            pt_order = df.groupby('Property_Type')['Price_per_SqFt'].median().sort_values(ascending=False).index
            import seaborn as sns
            sns.boxplot(data=df, x='Property_Type', y='Price_per_SqFt', order=pt_order,
                        palette='Set2', ax=ax3)
            ax3.set_title('Price per SqFt by Property Type', fontsize=11, fontweight='bold')
            plt.xticks(rotation=20)
            plt.tight_layout()
            st.pyplot(fig3)

        with c4:
            # Price distribution
            fig4, ax4 = plt.subplots(figsize=(7, 4))
            import seaborn as sns
            sns.histplot(df['Price_in_Lakhs'], bins=50, kde=True, color='steelblue', ax=ax4)
            ax4.set_title('Overall Price Distribution (Lakhs)', fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig4)

    except FileNotFoundError:
        st.warning("Run `preprocess.py` first to generate processed_data.csv.")


with tab3:
    st.markdown('<div class="section-hdr">🗂️ Dataset Explorer</div>', unsafe_allow_html=True)
    try:
        df = load_data()

        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            filter_city = st.multiselect("Filter by City", sorted(df['City'].unique()))
        with col_f2:
            price_range = st.slider("Price Range (Lakhs)", float(df['Price_in_Lakhs'].min()),
                                    float(df['Price_in_Lakhs'].max()),
                                    (float(df['Price_in_Lakhs'].min()), float(df['Price_in_Lakhs'].max())))
        with col_f3:
            bhk_filter = st.multiselect("BHK", sorted(df['BHK'].unique()))

        filtered = df.copy()
        if filter_city:
            filtered = filtered[filtered['City'].isin(filter_city)]
        filtered = filtered[(filtered['Price_in_Lakhs'] >= price_range[0]) &
                            (filtered['Price_in_Lakhs'] <= price_range[1])]
        if bhk_filter:
            filtered = filtered[filtered['BHK'].isin(bhk_filter)]

        display_cols = ['City', 'Locality', 'Property_Type', 'BHK', 'Size_in_SqFt',
                        'Price_in_Lakhs', 'Price_per_SqFt', 'Good_Investment',
                        'Future_Price_5Y', 'Availability_Status']
        st.dataframe(filtered[display_cols].head(200), use_container_width=True)
        st.caption(f"Showing {min(200, len(filtered))} of {len(filtered)} records after filters.")

    except FileNotFoundError:
        st.warning("Run `preprocess.py` first to generate processed_data.csv.")

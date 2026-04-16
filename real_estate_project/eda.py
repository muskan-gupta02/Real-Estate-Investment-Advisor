import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='muted')
os.makedirs('eda_plots', exist_ok=True)


def run_eda(df):
    print("=" * 55)
    print("   EXPLORATORY DATA ANALYSIS — REAL ESTATE DATASET")
    print("=" * 55)

    # ── 1. Price distribution ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df['Price_in_Lakhs'], bins=50, kde=True, color='steelblue', ax=ax)
    ax.set_title('Distribution of Property Prices (Lakhs)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Price (Lakhs)')
    plt.tight_layout()
    plt.savefig('eda_plots/01_price_distribution.png', dpi=120)
    plt.close()

    # ── 2. Size distribution ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df['Size_in_SqFt'], bins=50, kde=True, color='darkorange', ax=ax)
    ax.set_title('Distribution of Property Sizes (SqFt)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda_plots/02_size_distribution.png', dpi=120)
    plt.close()

    # ── 3. Price per sqft by property type ──────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df.groupby('Property_Type')['Price_per_SqFt'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='Property_Type', y='Price_per_SqFt', order=order,
                palette='Set2', ax=ax)
    ax.set_title('Price per SqFt by Property Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda_plots/03_price_per_sqft_by_type.png', dpi=120)
    plt.close()

    # ── 4. Size vs Price scatter ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df['Size_in_SqFt'], df['Price_in_Lakhs'], alpha=0.3, color='teal', s=18)
    ax.set_title('Property Size vs Price', fontsize=14, fontweight='bold')
    ax.set_xlabel('Size (SqFt)')
    ax.set_ylabel('Price (Lakhs)')
    plt.tight_layout()
    plt.savefig('eda_plots/04_size_vs_price.png', dpi=120)
    plt.close()

    # ── 5. Outliers — price per sqft ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=df['Price_per_SqFt'], color='salmon', ax=axes[0])
    axes[0].set_title('Outliers: Price per SqFt')
    sns.boxplot(y=df['Size_in_SqFt'], color='lightblue', ax=axes[1])
    axes[1].set_title('Outliers: Property Size')
    plt.suptitle('Outlier Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda_plots/05_outliers.png', dpi=120)
    plt.close()

    # ── 6. Avg price per sqft by state ──────────────────────────
    state_avg = df.groupby('State')['Price_per_SqFt'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    state_avg.plot(kind='bar', color='mediumseagreen', ax=ax)
    ax.set_title('Avg Price per SqFt by State', fontsize=14, fontweight='bold')
    ax.set_ylabel('Avg Price/SqFt')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('eda_plots/06_avg_price_by_state.png', dpi=120)
    plt.close()

    # ── 7. Avg property price by city (top 15) ──────────────────
    city_avg = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(12, 5))
    city_avg.plot(kind='bar', color='cornflowerblue', ax=ax)
    ax.set_title('Avg Property Price by City (Top 15)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Avg Price (Lakhs)')
    plt.xticks(rotation=40, ha='right')
    plt.tight_layout()
    plt.savefig('eda_plots/07_avg_price_by_city.png', dpi=120)
    plt.close()

    # ── 8. Median age by locality (top 10) ──────────────────────
    loc_age = df.groupby('Locality')['Age_of_Property'].median().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(11, 5))
    loc_age.plot(kind='barh', color='orchid', ax=ax)
    ax.set_title('Median Property Age by Locality (Top 10)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda_plots/08_age_by_locality.png', dpi=120)
    plt.close()

    # ── 9. BHK distribution by city (top 6) ─────────────────────
    top_cities = df['City'].value_counts().head(6).index
    sub = df[df['City'].isin(top_cities)]
    fig, ax = plt.subplots(figsize=(12, 5))
    bhk_city = sub.groupby(['City', 'BHK']).size().unstack(fill_value=0)
    bhk_city.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
    ax.set_title('BHK Distribution Across Top Cities', fontsize=14, fontweight='bold')
    ax.legend(title='BHK', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('eda_plots/09_bhk_by_city.png', dpi=120)
    plt.close()

    # ── 10. Price trends — top 5 expensive localities ────────────
    top_loc = df.groupby('Locality')['Price_per_SqFt'].mean().sort_values(ascending=False).head(5).index
    fig, ax = plt.subplots(figsize=(10, 5))
    for loc in top_loc:
        sub = df[df['Locality'] == loc].sort_values('Year_Built')
        ax.plot(sub['Year_Built'], sub['Price_in_Lakhs'], alpha=0.6, label=loc)
    ax.set_title('Price Trends — Top 5 Expensive Localities', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year Built')
    ax.legend()
    plt.tight_layout()
    plt.savefig('eda_plots/10_price_trends_top_localities.png', dpi=120)
    plt.close()

    # ── 11. Correlation heatmap ──────────────────────────────────
    num_cols = ['Price_in_Lakhs', 'Price_per_SqFt', 'Size_in_SqFt', 'BHK',
                'Age_of_Property', 'Nearby_Schools', 'Nearby_Hospitals',
                'Public_Transport_Accessibility', 'Parking_Space', 'Infra_Score']
    fig, ax = plt.subplots(figsize=(11, 8))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix — Numeric Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda_plots/11_correlation_heatmap.png', dpi=120)
    plt.close()

    # ── 12. Nearby schools vs price per sqft ─────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    school_price = df.groupby('Nearby_Schools')['Price_per_SqFt'].mean()
    school_price.plot(kind='bar', color='gold', ax=ax)
    ax.set_title('Nearby Schools vs Avg Price per SqFt', fontsize=14, fontweight='bold')
    ax.set_xlabel('Nearby Schools Count')
    ax.set_ylabel('Avg Price/SqFt')
    plt.tight_layout()
    plt.savefig('eda_plots/12_schools_vs_price.png', dpi=120)
    plt.close()

    # ── 13. Nearby hospitals vs price per sqft ───────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    hosp_price = df.groupby('Nearby_Hospitals')['Price_per_SqFt'].mean()
    hosp_price.plot(kind='bar', color='tomato', ax=ax)
    ax.set_title('Nearby Hospitals vs Avg Price per SqFt', fontsize=14, fontweight='bold')
    ax.set_xlabel('Nearby Hospitals Count')
    ax.set_ylabel('Avg Price/SqFt')
    plt.tight_layout()
    plt.savefig('eda_plots/13_hospitals_vs_price.png', dpi=120)
    plt.close()

    # ── 14. Price by furnished status ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x='Furnished_Status', y='Price_in_Lakhs', palette='pastel', ax=ax)
    ax.set_title('Price by Furnished Status', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda_plots/14_price_by_furnished.png', dpi=120)
    plt.close()

    # ── 15. Price per sqft by facing direction ───────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    order_f = df.groupby('Facing')['Price_per_SqFt'].median().sort_values(ascending=False).index
    sns.barplot(data=df, x='Facing', y='Price_per_SqFt', order=order_f,
                palette='viridis', ax=ax)
    ax.set_title('Price per SqFt by Facing Direction', fontsize=14, fontweight='bold')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig('eda_plots/15_price_by_facing.png', dpi=120)
    plt.close()

    # ── 16. Owner type distribution ──────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    df['Owner_Type'].value_counts().plot(kind='pie', autopct='%1.1f%%',
                                         colors=['#66b3ff', '#ff9999', '#99ff99'],
                                         startangle=140, ax=ax)
    ax.set_ylabel('')
    ax.set_title('Owner Type Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda_plots/16_owner_type.png', dpi=120)
    plt.close()

    # ── 17. Availability status distribution ─────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    df['Availability_Status'].value_counts().plot(kind='bar', color=['#4CAF50', '#2196F3', '#FF5722'], ax=ax)
    ax.set_title('Properties by Availability Status', fontsize=14, fontweight='bold')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('eda_plots/17_availability_status.png', dpi=120)
    plt.close()

    # ── 18. Parking space vs price ───────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    park_price = df.groupby('Parking_Space')['Price_in_Lakhs'].mean()
    park_price.plot(kind='bar', color='mediumpurple', ax=ax)
    ax.set_title('Effect of Parking Space on Property Price', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Parking Spaces')
    ax.set_ylabel('Avg Price (Lakhs)')
    plt.tight_layout()
    plt.savefig('eda_plots/18_parking_vs_price.png', dpi=120)
    plt.close()

    # ── 19. Amenities vs price per sqft ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    order_a = df.groupby('Amenities')['Price_per_SqFt'].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x='Amenities', y='Price_per_SqFt', order=order_a,
                palette='cubehelix', ax=ax)
    ax.set_title('Amenities vs Avg Price per SqFt', fontsize=14, fontweight='bold')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig('eda_plots/19_amenities_vs_price.png', dpi=120)
    plt.close()

    # ── 20. Public transport vs investment potential ──────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    transport_invest = df.groupby('Public_Transport_Accessibility')['Good_Investment'].mean() * 100
    transport_invest.plot(kind='bar', color='darkcyan', ax=ax)
    ax.set_title('Public Transport Score vs Good Investment (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Transport Accessibility Score')
    ax.set_ylabel('% Good Investments')
    plt.tight_layout()
    plt.savefig('eda_plots/20_transport_vs_investment.png', dpi=120)
    plt.close()

    print("All 20 EDA plots saved to eda_plots/")


if __name__ == '__main__':
    from preprocess import load_and_preprocess
    df, _ = load_and_preprocess()
    run_eda(df)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pickle, json, os
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Crime Analysis & Prediction",
    page_icon="🔍",
    layout="wide"
)

# ─── Paths ───────────────────────────────────────────────────────
BASE = '/Users/prernamalhotra/CRIME/crime-analysis-prediction'
DATA_PATH    = f'{BASE}/data/crime_data_cleaned.csv'
MODEL_PATH   = f'{BASE}/models/best_model.pkl'
SCALER_PATH  = f'{BASE}/models/scaler.pkl'
LE_PATH      = f'{BASE}/models/label_encoder.pkl'
FEATURES_PATH= f'{BASE}/models/feature_cols.json'

# ─── Load Model ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = pickle.load(open(MODEL_PATH, 'rb'))
    scaler   = pickle.load(open(SCALER_PATH, 'rb'))
    le       = pickle.load(open(LE_PATH, 'rb'))
    features = json.load(open(FEATURES_PATH, 'r'))
    return model, scaler, le, features

# ─── Load Data ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

model, scaler, le, feature_cols = load_model()
df = load_data()

# ─── Sidebar Navigation ──────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/9/99/UPES_Dehradun_logo.svg/220px-UPES_Dehradun_logo.svg.png", width=160)
st.sidebar.title("🔍 Crime Analysis")
st.sidebar.markdown("**Chicago Dataset | 2012–2017**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Dashboard",
    "📊 EDA Charts",
    "🗺️ Crime Hotspot Map",
    "🤖 Predict Crime"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Info**")
st.sidebar.markdown("🎓 UPES Dehradun")
st.sidebar.markdown("📚 B.Tech CSE — Major Project")
st.sidebar.markdown("👩‍💻 Prerna Malhotra  , Aniket Agrawal , Aryan Karnwal , Varun Gupta")

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("🔍 Crime Analysis & Prediction Platform")
    st.markdown("##### Chicago Crime Dataset | 2012–2017 | Machine Learning Project")
    st.markdown("---")

    # Metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📋 Total Records",    f"{len(df):,}")
    col2.metric("🗂️ Crime Categories", df['Crime Category'].nunique())
    col3.metric("📅 Years Covered",    f"{int(df['Year'].min())}–{int(df['Year'].max())}")
    col4.metric("🚔 Arrest Rate",      f"{df['Arrest'].mean()*100:.1f}%")
    col5.metric("🎯 Model Accuracy",   "59.96%")

    st.markdown("---")

    # Two column layout
    left, right = st.columns(2)

    with left:
        st.subheader("Crime Category Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        counts = df['Crime Category'].value_counts()
        colors = ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6']
        ax.bar(counts.index, counts.values, color=colors)
        ax.set_ylabel("Number of Crimes")
        ax.set_xlabel("Category")
        ax.tick_params(axis='x', rotation=20)
        for bar, val in zip(ax.patches, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
                    f'{val:,}', ha='center', fontsize=8, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    with right:
        st.subheader("Crime Trend Over the Years")
        fig, ax = plt.subplots(figsize=(7, 4))
        yearly = df.groupby('Year').size().reset_index(name='Count')
        ax.plot(yearly['Year'], yearly['Count'], marker='o', linewidth=2.5,
                color='#e74c3c', markersize=8)
        ax.fill_between(yearly['Year'], yearly['Count'], alpha=0.15, color='#e74c3c')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Crimes")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("📌 Key Findings")
    k1, k2, k3 = st.columns(3)
    k1.info("🏠 **Top Location**: Street and Residence account for the most crimes")
    k2.warning("🕛 **Peak Hour**: Crime is highest at 12 PM (noon)")
    k3.error("💊 **Highest Arrest Rate**: Drug crimes — 99.3% arrest rate")

    k4, k5, k6 = st.columns(3)
    k4.info("📅 **Riskiest Day**: Friday has the highest crime count")
    k5.warning("☀️ **Riskiest Season**: Summer months (June–August) see most crimes")
    k6.success("📉 **Trend**: Crime has been decreasing from 2012 to 2017")


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — EDA CHARTS
# ══════════════════════════════════════════════════════════════════
elif page == "📊 EDA Charts":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Visualizing patterns in 1.4 million Chicago crime records.")
    st.markdown("---")

    chart = st.selectbox("Select a Chart to View", [
        "1. Crime by Hour of Day",
        "2. Crime by Day of Week",
        "3. Crime by Month",
        "4. Heatmap: Day vs Hour",
        "5. Top 15 Crime Locations",
        "6. Arrest Rate by Category",
        "7. Crime Trend by Year"
    ])

    fig, ax = plt.subplots(figsize=(13, 5))

    if chart == "1. Crime by Hour of Day":
        hourly = df.groupby('Hour').size().reset_index(name='Count')
        colors = ['#e74c3c' if h in [0,1,2,22,23,12] else '#3498db' for h in hourly['Hour']]
        ax.bar(hourly['Hour'], hourly['Count'], color=colors)
        ax.set_title("Crime Distribution by Hour of Day", fontsize=14, fontweight='bold')
        ax.set_xlabel("Hour (0 = Midnight, 12 = Noon)")
        ax.set_ylabel("Number of Crimes")
        ax.set_xticks(range(0, 24))
        st.pyplot(fig)
        st.info("🔍 **Insight**: Crime peaks at noon (12 PM) and is lowest at 5 AM. Late night hours (10 PM–2 AM) are also high-risk.")

    elif chart == "2. Crime by Day of Week":
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        daily = df.groupby('DayName').size().reindex(day_order)
        colors = ['#e74c3c' if d in ['Friday','Saturday','Sunday'] else '#3498db' for d in day_order]
        ax.bar(day_order, daily.values, color=colors)
        ax.set_title("Crime Distribution by Day of Week", fontsize=14, fontweight='bold')
        ax.set_xlabel("Day")
        ax.set_ylabel("Number of Crimes")
        ax.tick_params(axis='x', rotation=20)
        st.pyplot(fig)
        st.info("🔍 **Insight**: Friday is the most crime-prone day. Weekends (shown in red) consistently have higher crime rates.")

    elif chart == "3. Crime by Month":
        monthly = df.groupby('Month').size().reset_index(name='Count')
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        month_colors = ['#3498db','#3498db','#2ecc71','#2ecc71','#2ecc71',
                        '#e74c3c','#e74c3c','#e74c3c','#f39c12','#f39c12','#f39c12','#3498db']
        ax.bar(month_names, monthly['Count'], color=month_colors)
        ax.set_title("Crime Distribution by Month", fontsize=14, fontweight='bold')
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Crimes")
        st.pyplot(fig)
        st.info("🔍 **Insight**: Summer months (June–August, shown in red) have the highest crime. Winter months are lowest.")

    elif chart == "4. Heatmap: Day vs Hour":
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        pivot = df.groupby(['DayName','Hour']).size().unstack(fill_value=0).reindex(day_order)
        fig2, ax2 = plt.subplots(figsize=(16, 6))
        sns.heatmap(pivot, cmap='YlOrRd', linewidths=0.2,
                    cbar_kws={'label': 'Crime Count'}, ax=ax2)
        ax2.set_title("Crime Heatmap: Day of Week vs Hour", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("Day of Week")
        plt.tight_layout()
        st.pyplot(fig2)
        st.info("🔍 **Insight**: Dark red cells at Friday noon and Saturday midnight reveal the highest-risk windows.")

    elif chart == "5. Top 15 Crime Locations":
        top_loc = df['Location Description'].value_counts().head(15)
        top_loc[::-1].plot(kind='barh', ax=ax, color=sns.color_palette("RdYlBu", 15))
        ax.set_title("Top 15 Crime Locations", fontsize=14, fontweight='bold')
        ax.set_xlabel("Number of Crimes")
        plt.tight_layout()
        st.pyplot(fig)
        st.info("🔍 **Insight**: Streets and Residences are by far the most common crime locations.")

    elif chart == "6. Arrest Rate by Category":
        arrest = df.groupby('Crime Category')['Arrest'].mean().sort_values(ascending=False) * 100
        bar_colors = ['#27ae60' if v > 20 else '#e74c3c' for v in arrest.values]
        bars = ax.bar(arrest.index, arrest.values, color=bar_colors)
        ax.set_title("Arrest Rate by Crime Category (%)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Arrest Rate (%)")
        for bar, val in zip(bars, arrest.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        st.info("🔍 **Insight**: Drug crimes have a 99.3% arrest rate. Property crimes have only 8.9% — most go unarrested.")

    elif chart == "7. Crime Trend by Year":
        yearly = df.groupby('Year').size().reset_index(name='Count')
        ax.plot(yearly['Year'], yearly['Count'], marker='o', linewidth=2.5,
                color='#e74c3c', markersize=10)
        ax.fill_between(yearly['Year'], yearly['Count'], alpha=0.15, color='#e74c3c')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.set_title("Yearly Crime Trend (2012–2017)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Crimes")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        st.info("🔍 **Insight**: Crime has been steadily declining from 2012 to 2017 — showing the impact of policing improvements.")


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — CRIME HOTSPOT MAP
# ══════════════════════════════════════════════════════════════════
elif page == "🗺️ Crime Hotspot Map":
    st.title("🗺️ Crime Hotspot Map")
    st.markdown("Interactive heatmap showing crime density across Chicago.")
    st.markdown("---")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### Filters")
        category_filter = st.selectbox(
            "Crime Category",
            ['All'] + sorted(df['Crime Category'].unique().tolist())
        )
        year_filter = st.selectbox(
            "Year",
            ['All'] + sorted([int(y) for y in df['Year'].dropna().unique()])
        )
        sample_size = st.slider("Sample Size (points on map)", 5000, 50000, 20000, step=5000)
        st.markdown("---")
        st.markdown("**Color Scale**")
        st.markdown("🔵 Low density")
        st.markdown("🟡 Medium density")
        st.markdown("🔴 High density")

    with col2:
        filtered = df.copy()
        if category_filter != 'All':
            filtered = filtered[filtered['Crime Category'] == category_filter]
        if year_filter != 'All':
            filtered = filtered[filtered['Year'] == year_filter]

        filtered = filtered.dropna(subset=['Latitude','Longitude'])
        filtered = filtered[
            (filtered['Latitude']  >= 41.6) & (filtered['Latitude']  <= 42.1) &
            (filtered['Longitude'] >= -87.9) & (filtered['Longitude'] <= -87.5)
        ]

        n = min(sample_size, len(filtered))
        sample = filtered.sample(n=n, random_state=42)

        st.markdown(f"**Showing {n:,} crime points** — Category: `{category_filter}` | Year: `{year_filter}`")

        m = folium.Map(location=[41.8781, -87.6298], zoom_start=11,
                       tiles='CartoDB dark_matter')

        HeatMap(
            sample[['Latitude','Longitude']].values.tolist(),
            radius=10, blur=12, max_zoom=14, min_opacity=0.4,
            gradient={0.2:'blue', 0.4:'cyan', 0.65:'lime', 0.8:'yellow', 1.0:'red'}
        ).add_to(m)

        st_folium(m, width=750, height=520)


# ══════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT CRIME
# ══════════════════════════════════════════════════════════════════
elif page == "🤖 Predict Crime":
    st.title("🤖 Crime Type Predictor")
    st.markdown("Enter the details below to predict the most likely crime category.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🕐 Time Details")
        hour      = st.slider("Hour of Day", 0, 23, 12,
                               help="0 = Midnight, 12 = Noon, 23 = 11 PM")
        day       = st.slider("Day of Month", 1, 31, 15)
        month     = st.selectbox("Month", list(range(1,13)),
                                  format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun',
                                                          'Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
        dow       = st.selectbox("Day of Week",
                                  ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        dow_num   = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'].index(dow)
        year      = st.selectbox("Year", [2018, 2019, 2020, 2021, 2022, 2023, 2024])

    with col2:
        st.markdown("### 📍 Location Details")
        latitude  = st.number_input("Latitude",  min_value=41.6, max_value=42.1,
                                     value=41.88, step=0.01,
                                     help="Chicago range: 41.6 to 42.1")
        longitude = st.number_input("Longitude", min_value=-87.9, max_value=-87.5,
                                     value=-87.63, step=0.01,
                                     help="Chicago range: -87.9 to -87.5")
        district  = st.number_input("Police District", min_value=1, max_value=25, value=8)

    with col3:
        st.markdown("### 🚔 Incident Details")
        arrest    = st.radio("Was an Arrest Made?", ["No", "Yes"])
        domestic  = st.radio("Was it a Domestic Incident?", ["No", "Yes"])

        st.markdown("---")
        st.markdown("#### Quick Location Presets")
        preset = st.selectbox("Use a Preset Location", [
            "Custom (use values above)",
            "Downtown Chicago",
            "North Side",
            "South Side",
            "West Side"
        ])
        if preset == "Downtown Chicago":
            latitude, longitude, district = 41.8827, -87.6233, 1
        elif preset == "North Side":
            latitude, longitude, district = 41.9742, -87.6688, 19
        elif preset == "South Side":
            latitude, longitude, district = 41.7486, -87.6109, 5
        elif preset == "West Side":
            latitude, longitude, district = 41.8827, -87.7200, 11

    st.markdown("---")

    predict_btn = st.button("🔍 Predict Crime Category", use_container_width=True)

    if predict_btn:
        input_dict = {
            'Hour':      hour,
            'Day':       day,
            'Month':     month,
            'DayOfWeek': dow_num,
            'Year':      year,
            'Latitude':  latitude,
            'Longitude': longitude,
            'Arrest':    1 if arrest == "Yes" else 0,
            'Domestic':  1 if domestic == "Yes" else 0,
            'District':  district
        }

        input_df     = pd.DataFrame([[input_dict.get(f, 0) for f in feature_cols]],
                                      columns=feature_cols)
        input_scaled = scaler.transform(input_df)
        pred         = model.predict(input_scaled)[0]
        pred_proba   = model.predict_proba(input_scaled)[0]
        pred_label   = le.inverse_transform([pred])[0]

        CATEGORY_INFO = {
            'Property':     ('🏠', '#3498db', 'Includes theft, burglary, motor vehicle theft, and arson.'),
            'Violent':      ('⚠️', '#e74c3c', 'Includes assault, battery, robbery, and homicide.'),
            'Drug':         ('💊', '#27ae60', 'Narcotics-related offences with very high arrest rate.'),
            'Public Order': ('📢', '#f39c12', 'Includes weapons violations, disorderly conduct, gambling.'),
            'Other':        ('📋', '#9b59b6', 'Miscellaneous offences not in main categories.'),
            'Fraud':        ('💳', '#1abc9c', 'Includes forgery, counterfeiting, and identity theft.')
        }

        emoji, color, desc = CATEGORY_INFO.get(pred_label, ('🔍','#555',''))

        st.markdown(f"""
        <div style='background:{color}22; border-left: 6px solid {color};
             padding: 20px; border-radius: 8px; margin-top: 10px;'>
            <h2 style='color:{color}; margin:0;'>{emoji} Predicted: {pred_label} Crime</h2>
            <p style='margin-top:8px; font-size:15px;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Crime Category': le.classes_,
            'Probability (%)': (pred_proba * 100).round(2)
        }).sort_values('Probability (%)', ascending=False).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        bar_colors = [color if c == pred_label else '#bdc3c7' for c in prob_df['Crime Category']]
        bars = ax.barh(prob_df['Crime Category'], prob_df['Probability (%)'], color=bar_colors)
        for bar, val in zip(bars, prob_df['Probability (%)']):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')
        ax.set_xlabel("Probability (%)")
        ax.set_title("Crime Category Probabilities", fontsize=13, fontweight='bold')
        ax.set_xlim(0, 110)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("#### Input Summary")
        summary_df = pd.DataFrame([{
            'Hour': hour, 'Day': day, 'Month': month, 'Day of Week': dow,
            'Year': year, 'Latitude': latitude, 'Longitude': longitude,
            'Arrest': arrest, 'Domestic': domestic, 'District': district
        }])
        st.dataframe(summary_df, use_container_width=True)
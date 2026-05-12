# Crime Analysis and Prediction Platform

An intelligent machine learning-based platform designed to analyze historical crime data, identify crime patterns, detect hotspots, and predict crime categories using advanced data analytics and visualization techniques.

---

## 📌 Project Overview

This project focuses on analyzing historical crime data and predicting crime categories using Machine Learning algorithms. The platform provides interactive visualizations, hotspot analysis, and a Streamlit dashboard to help understand crime trends and support proactive decision-making.

The system uses the Chicago Crime Dataset (2012–2017) containing over 1.4 million crime records.

---

## 🚀 Features

- Crime Data Analysis
- Machine Learning-Based Crime Prediction
- Exploratory Data Analysis (EDA)
- Interactive Crime Hotspot Maps
- Streamlit Dashboard
- Real-Time Crime Prediction
- Visualization Charts & Heatmaps
- Multiple ML Model Comparison

---

## 📂 Dataset Information

| Feature | Details |
|---|---|
| Dataset Source | Chicago Crime Dataset (Kaggle) |
| Time Period | 2012 – 2017 |
| Original Records | 1,456,714 |
| Cleaned Records | 1,419,631 |
| Categories | Property, Violent, Drug, Public Order, Other |

---

## 🛠️ Technology Stack

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Folium
- Matplotlib
- SMOTE

---

## 🤖 Machine Learning Models Used

| Model | Accuracy |
|---|---|
| Logistic Regression | 58.00% |
| Decision Tree | 59.51% |
| Random Forest | 59.96% |
| XGBoost + SMOTE | 58.49% |

### Best Performing Model
**Random Forest** achieved the highest accuracy of **59.96%**.

---

## 📊 Key Insights

- Latitude & Longitude are the most important predictive features.
- Peak crime hour observed: 12 PM.
- Friday has the highest crime rate among weekdays.
- Street, Residence, and Apartment are top crime locations.
- Crime rates gradually decreased from 2012 to 2017.

---

## 🧩 Methodology

1. Data Collection
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Training
6. Hotspot Mapping
7. Dashboard Deployment

---

## 🌐 Streamlit Dashboard

The dashboard contains:
- Home Page
- EDA Charts
- Crime Hotspot Maps
- AI Crime Prediction Page

---

## 📁 Project Structure

```bash
Crime-Analysis-and-Prediction/
│
├── data/
├── notebooks/
├── models/
├── app/
├── requirements.txt
└── README.md
```

---

## ▶️ Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/your-username/crime-analysis-prediction.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## 📌 Future Scope

- Real-time crime prediction
- LSTM-based forecasting
- Cloud deployment using AWS/GCP
- Multi-city crime analysis
- Advanced deep learning integration

---

## 👨‍💻 Contributors

- Varun Gupta
- Aniket Agrawal
- Prerna Malhotra
- Aryan Karnwal

Guided by:
**Dr. Sanoj Kumar**
UPES Dehradun

---

## 📜 License

This project is developed for educational and research purposes.

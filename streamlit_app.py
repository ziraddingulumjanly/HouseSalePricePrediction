import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load trained model
model = joblib.load("house_price_model.pkl")

# Load a sample template row
@st.cache_data
def load_template():
    df = pd.read_csv("train.csv")
    df['Alley'] = df['Alley'].fillna('NoAlley')
    df['MasVnrType'] = df['MasVnrType'].fillna('NoMasonry')
    df['LotFrontage'] = df['LotFrontage'].fillna(0)
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(f'No{col}')
            else:
                df[col] = df[col].fillna(0)

    # Feature engineering
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath'] = (
        df['FullBath'] + 0.5 * df['HalfBath'] +
        df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    )
    df['TotalPorchSF'] = (
        df['OpenPorchSF'] + df['EnclosedPorch'] +
        df['3SsnPorch'] + df['ScreenPorch']
    )
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)

    df.drop(columns=['Id', 'SalePrice', 'LogSalePrice'], errors='ignore', inplace=True)
    return df.iloc[0].copy()

template_row = load_template()

# Streamlit UI
st.title("🏠 House Sale Price Prediction")
st.markdown("Edit the key property details below:")

# Editable fields
template_row["OverallQual"] = st.slider("Overall Quality", 1, 10, int(template_row["OverallQual"]))
template_row["GrLivArea"] = st.number_input("Above Ground Living Area (sq ft)", value=int(template_row["GrLivArea"]))
template_row["GarageCars"] = st.slider("Garage Cars", 0, 4, int(template_row["GarageCars"]))
template_row["TotalBsmtSF"] = st.number_input("Basement Area (sq ft)", value=int(template_row["TotalBsmtSF"]))
template_row["FullBath"] = st.slider("Full Bathrooms", 0, 4, int(template_row["FullBath"]))
template_row["TotRmsAbvGrd"] = st.slider("Total Rooms Above Ground", 2, 12, int(template_row["TotRmsAbvGrd"]))
template_row["YearBuilt"] = st.number_input("Year Built", value=int(template_row["YearBuilt"]))
template_row["YearRemodAdd"] = st.number_input("Year Remodeled", value=int(template_row["YearRemodAdd"]))

# Update engineered features
template_row["HouseAge"] = template_row["YrSold"] - template_row["YearBuilt"]
template_row["RemodAge"] = template_row["YrSold"] - template_row["YearRemodAdd"]
template_row["TotalSF"] = template_row["TotalBsmtSF"] + template_row["1stFlrSF"] + template_row["2ndFlrSF"]
template_row["TotalBath"] = (
    template_row["FullBath"] + 0.5 * template_row["HalfBath"] +
    template_row["BsmtFullBath"] + 0.5 * template_row["BsmtHalfBath"]
)
template_row["TotalPorchSF"] = (
    template_row["OpenPorchSF"] + template_row["EnclosedPorch"] +
    template_row["3SsnPorch"] + template_row["ScreenPorch"]
)
template_row["HasPool"] = int(template_row["PoolArea"] > 0)
template_row["HasFireplace"] = int(template_row["Fireplaces"] > 0)
template_row["HasGarage"] = int(template_row["GarageArea"] > 0)

# Convert to DataFrame
input_df = pd.DataFrame([template_row])

# 🛠️ Ensure input_df matches model’s expected columns
expected_columns = model.named_steps["preprocessor"].feature_names_in_
input_df = input_df.reindex(columns=expected_columns, fill_value=0)

# Predict
if st.button("Predict Sale Price"):
    log_price = model.predict(input_df)[0]
    sale_price = np.expm1(log_price)
    st.success(f"💰 Estimated Sale Price: ${sale_price:,.0f}")

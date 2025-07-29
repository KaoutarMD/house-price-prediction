import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(page_title="House Price Predictor in Melbourne", layout="wide")

# Chargement des modèles et scalers
loaded_lr_model= joblib.load('linear_regression_model.pkl')
loaded_rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load("scaler_columns.pkl")

# Encoders
encoders = {
    'Type': joblib.load('le_Type.pkl'),
    'Method': joblib.load('le_Method.pkl'),
    'SellerG': joblib.load('le_SellerG.pkl'),
    'Suburb': joblib.load('le_Suburb.pkl'),
    'CouncilArea': joblib.load('le_CouncilArea.pkl'),
    'Regionname': joblib.load('le_Regionname.pkl')
}

# Titre principal
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>🏠 House Price Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: grey;'>Predict the price of a house based on its features</h4>",
    unsafe_allow_html=True
)

# Formulaire avec mise en page responsive
with st.form(key='prediction_form'):
    features = {}
    columns = st.columns(3)  # 3 colonnes pour éviter le scroll

    for i, col in enumerate(expected_columns):
        with columns[i % 3]:  # Répartition dans les 3 colonnes
            if col in encoders:
                le = encoders[col]
                features[col] = st.selectbox(f"{col}", options=le.classes_)
            elif col in ['Rooms', 'BuildingArea', 'YearBuilt', 'Car']:
                features[col] = st.number_input(f"{col}", min_value=0, value=0)
            elif col in ['Distance', 'Lattitude', 'Longtitude']:
                features[col] = st.number_input(f"{col}", value=0.0, format="%.4f")

    submit_button = st.form_submit_button(label='🎯 Predict Price')

# Traitement
if submit_button:
    try:
        input_df = pd.DataFrame([features])
        input_df = input_df.reindex(columns=expected_columns)

        for col in encoders:
            le = encoders[col]
            input_df[col] = le.transform(input_df[col])

        
        if input_df.isnull().any().any():
            st.error("❌ Des valeurs manquent. Veuillez remplir tous les champs.")
        else:
            input_scaled = scaler.transform(input_df)
            log_prediction = loaded_lr_model.predict(input_scaled)
            rf_prediction = loaded_rf_model.predict(input_scaled)


            prix_lr = np.expm1(log_prediction[0])
            prix_rf = np.expm1(rf_prediction[0])

            col1, col2 = st.columns(2)
            with col1:
               st.metric("🔹 Linear Regression", f"${prix_lr:,.2f}")
            with col2:
               st.metric("🌲 Random Forest", f"${prix_rf:,.2f}")

    except Exception as e:
        st.error(f"❌ Une erreur est survenue : {e}")

# Pied de page
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "Made with ❤️ by <b>Mahda Kaoutar</b><br>"
    "<a href='https://github.com/KaoutarMD'>🔗 GitHub Repository</a> | "
    "<a href='https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot'>📊 Dataset</a>"
    "</div>",
    unsafe_allow_html=True
)

# 🏠 House Price Prediction App

This project is an interactive web application that predicts house prices in Melbourne based on their features. It uses Linear Regression and Random Forest models trained on the [Melbourne Housing Snapshot](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot) dataset.

## Features

- Intuitive user interface built with [Streamlit](https://streamlit.io/)
- Price prediction using two models: Linear Regression and Random Forest
- Automatic encoding of categorical variables
- Data scaling for improved accuracy
- Real-time display of prediction results

## Project Structure

```
app.py
house-price-prediction.ipynb
le_CouncilArea.pkl
le_Method.pkl
le_Regionname.pkl
le_SellerG.pkl
le_Suburb.pkl
le_Type.pkl
linear_regression_model.pkl
random_forest_model.pkl
Readme.md
scaler_columns.pkl
scaler.pkl
data/
    melb_data.csv
images/
    Cap1.png
    Cap2.png
```

## How to Run the App

```sh
streamlit run app.py
```

## Usage

- Fill in the form fields with the house features.
- Click "🎯 Predict Price" to get predictions from both models.
- Results are displayed instantly.

## Key Files

- [`app.py`](app.py): Main Streamlit application code
- [`house-price-prediction.ipynb`](house-price-prediction.ipynb): Model training notebook
- `data/` and `images/` folders: Dataset and screenshots

## Author

Made with ❤️ by [Mahda Kaoutar](https://github.com/KaoutarMD)


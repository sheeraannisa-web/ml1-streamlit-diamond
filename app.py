import streamlit as st
import pickle
import pandas as pd

# Load model dan kolom training
model = pickle.load(open("model_xgboost.pkl","rb"))
model_columns = pickle.load(open("model_columns.pkl","rb"))

st.title("Prediksi Harga Diamond (XGBoost)")

st.write("Masukkan karakteristik diamond untuk memprediksi harga")

# Input user
carat = st.number_input("Carat", min_value=0.0)
depth = st.number_input("Depth", min_value=0.0)
table = st.number_input("Table", min_value=0.0)
x = st.number_input("Length (x)", min_value=0.0)
y = st.number_input("Width (y)", min_value=0.0)
z = st.number_input("Height (z)", min_value=0.0)

cut = st.selectbox("Cut", ["Fair","Good","Very Good","Premium","Ideal"])
color = st.selectbox("Color", ["D","E","F","G","H","I","J"])
clarity = st.selectbox("Clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])

# Tombol prediksi
if st.button("Prediksi Harga"):

    # Data input
    input_data = {
        "carat": carat,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z,
        "cut": cut,
        "color": color,
        "clarity": clarity
    }

    df_input = pd.DataFrame([input_data])

    # Encoding seperti training
    df_input = pd.get_dummies(df_input)

    # Samakan kolom dengan training
    df_input = df_input.reindex(columns=model_columns, fill_value=0)

    # Prediksi
    prediction = model.predict(df_input)

    st.success(f"Perkiraan Harga Diamond: ${prediction[0]:,.2f}")
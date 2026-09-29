import os
import pandas as pd
import streamlit as st
import pickle
st.set_page_config(page_title="Стоимость недвижимости")
st.title("Прогноз стоимости")

model_path = "priceM.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

feature_columns = list(model.feature_names_in_)
values = {}
for column in feature_columns:
    values[column] = st.sidebar.number_input(column, value=0.0)

inputDF = pd.DataFrame(values, index=[0])
if st.button("Рассчитать стоимость"):
    prediction = model.predict(inputDF)[0]
    st.write(f"Прогноз: {prediction:,.2f}")

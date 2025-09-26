import numpy as np
import pickle
import streamlit as st

# Load model
try:
    with open("iris.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Model file not found: 'iris.pkl'")
    st.stop()

# Set app page settings
st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸")

# Custom CSS for colored labels
st.markdown("""
    <style>
    /* Label styling */
    label[for="sepal_length"] { color: #228B22; font-weight: bold; }      /* Green */
    label[for="sepal_width"]  { color: #006400; font-weight: bold; }      /* Dark green */
    label[for="petal_length"] { color: #8A2BE2; font-weight: bold; }      /* Blue violet */
    label[for="petal_width"]  { color: #FF1493; font-weight: bold; }      /* Deep pink */

    /* Optional: improve button appearance */
    button[kind="primary"] {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🌸 Iris Flower Prediction App")
st.markdown("This app uses a machine learning model to predict the **species** of an Iris flower based on its measurements.")

st.divider()

# Sidebar for info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app predicts the Iris species based on 4 key flower features.")
    st.markdown("""
    **Features:**
    - 🌿 Sepal Length & Width
    - 🌸 Petal Length & Width
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", caption="Iris Example")

# Main form inputs
st.header("📏 Enter Flower Measurements (in cm)")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("🌿 Sepal Length", key="sepal_length", min_value=0.0, value=5.0, step=0.1)
    petal_length = st.number_input("🌸 Petal Length", key="petal_length", min_value=0.0, value=1.5, step=0.1)

with col2:
    sepal_width = st.number_input("🌿 Sepal Width", key="sepal_width", min_value=0.0, value=3.5, step=0.1)
    petal_width = st.number_input("🌸 Petal Width", key="petal_width", min_value=0.0, value=0.2, step=0.1)

# Predict button
if st.button("🔍 Predict Species", use_container_width=True):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    iris_species_map = {
        0: "Setosa 🌼",
        1: "Versicolor 🌺",
        2: "Virginica 🌹"
    }

    predicted_species = iris_species_map[prediction[0]]

    st.success(f"✅ **The predicted species is: {predicted_species}**")

    # Show image
    if prediction[0] == 0:
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg", caption="Iris Setosa")
    elif prediction[0] == 1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/1024px-Iris_versicolor_3.jpg", caption="Iris Versicolor")
    else:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/1024px-Iris_virginica.jpg", caption="Iris Virginica")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model


@st.cache_resource
def load_scaler():
    return joblib.load("./models/scaler.pkl")


@st.cache_resource
def load_pca():
    return joblib.load("./models/pca.pkl")


@st.cache_resource
def load_random_forest():
    return joblib.load("./models/random_forest.pkl")


@st.cache_resource
def load_sgd():
    return joblib.load("./models/sgd.pkl")


@st.cache_resource
def load_lstm():
    return load_model("./models/lstm_model.keras")


@st.cache_resource
def load_lstm_scaler():
    return joblib.load("./models/scaler_lstm.pkl")


CLASS_NAMES = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}

# Class names for SGD/RF (1-indexed)
CLASS_NAMES_1_INDEXED = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

# Sidebar - Model Selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.radio("Choose a model:", ("Random Forest", "SGD", "LSTM"))

# Show input format info based on model
st.sidebar.markdown("---")
st.sidebar.markdown("### Input Format")
if model_choice == "LSTM":
    st.sidebar.info("LSTM expects a CSV with **128 rows × 9 columns** (raw inertial signals)")
else:
    st.sidebar.info("RF/SGD expect a CSV with **561 features** per row")

# Main UI
st.title("Human Activity Recognition")
st.write(f"**Current Model:** {model_choice}")

if model_choice == "LSTM":
    st.write("Upload a CSV file with **128 rows × 9 columns** (raw inertial signals)")
    st.caption("Columns: body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z, total_acc_x, total_acc_y, total_acc_z")
else:
    st.write("Upload a CSV file with **561 features** per row")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        if model_choice == "LSTM":
            # LSTM processing
            df = pd.read_csv(uploaded_file)
            
            st.write("### Preview of Data")
            st.dataframe(df.head())
            
            # Validate shape
            if df.shape != (128, 9):
                st.error(f"Expected shape (128, 9), got {df.shape}. Please upload a CSV with 128 rows and 9 columns.")
            else:
                # Load LSTM model and scaler
                lstm_model = load_lstm()
                lstm_scaler = load_lstm_scaler()
                
                # Preprocess: scale and reshape to 3D
                x_scaled_2d = lstm_scaler.transform(df.values)
                x_scaled_3d = x_scaled_2d.reshape(1, 128, 9)  # (1 sample, 128 timesteps, 9 channels)
                
                # Predict
                y_pred_proba = lstm_model.predict(x_scaled_3d, verbose=0)
                y_pred = np.argmax(y_pred_proba, axis=1)[0]
                confidence = np.max(y_pred_proba) * 100
                
                # Display result
                st.write("### Prediction Result")
                st.success(f"**Predicted Activity:** {CLASS_NAMES[y_pred]}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                
                # Show all class probabilities
                st.write("### Class Probabilities")
                proba_df = pd.DataFrame({
                    "Activity": [CLASS_NAMES[i] for i in range(6)],
                    "Probability (%)": [f"{p * 100:.2f}%" for p in y_pred_proba[0]]
                })
                st.dataframe(proba_df)
        else:
            # RF/SGD processing
            df = pd.read_csv(uploaded_file, header=None)
            
            st.write("### Preview of Data")
            st.dataframe(df.head())
            
            # Load shared resources
            scaler = load_scaler()
            pca = load_pca()
            
            # Load selected model
            if model_choice == "Random Forest":
                model = load_random_forest()
            else:
                model = load_sgd()
            
            # Preprocess
            x_scaled = scaler.transform(df)
            x_pca = pca.transform(x_scaled)

            # Predict
            y_pred = model.predict(x_pca)
            y_pred_proba = model.predict_proba(x_pca)

            # Build results dataframe
            results = pd.DataFrame(
                {
                    "Predicted Activity": [CLASS_NAMES_1_INDEXED[i] for i in y_pred],
                    "Confidence (%)": [
                        f"{np.max(proba) * 100:.2f}%" for proba in y_pred_proba
                    ],
                }
            )

            st.write("### Prediction Results")
            st.dataframe(results)

    except Exception as e:
        st.error(f"Error processing file: {e}")

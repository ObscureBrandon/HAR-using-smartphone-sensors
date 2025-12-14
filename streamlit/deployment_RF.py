import streamlit as st
import pandas as  pd
import numpy as  np
import joblib

scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
model = joblib.load('random_forest_model.pkl')
class_names = {
    0: 'WALKING',
1 :'WALKING_UPSTAIRS',
2 :'WALKING_DOWNSTAIRS',
3 :'SITTING',
4 :'STANDING',
5 :'LAYING',
}

st.title("human activity recognition")

st.write("upload CSV file for prediction")

uploaded_file = st.file_uploader("choose a CSV file",type = 'csv')

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, header=None)  # header in CSV
        # df = df.loc[:,~df.columns.duplicated()]   # keep only unique columns

        # df.columns = features
        st.write('preview of data')
        st.dataframe(df.head())
        
        x_scaled = scaler.transform(df)
        x_scaled = pca.transform(x_scaled)
        
        y_pred = model.predict(x_scaled)
        y_pred_proba = model.predict_proba(x_scaled)
        
        results = pd.DataFrame({
            'predicted label': [class_names[i] for i in y_pred],
            'predictions': [f"{np.max(x)*100}" for x in y_pred_proba]
        })
        
        st.write("prediction results")
        st.dataframe(results)
        
    except Exception as e:
        st.error(f"error processing file: {e}")
        
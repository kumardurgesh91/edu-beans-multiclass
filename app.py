import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page config
st.set_page_config(page_title="Bean Classifier", page_icon="🌱", layout="wide")

# 1. Load the Pipeline
@st.cache_resource
def load_pipeline():
    with open('bean_complete_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

pipeline = load_pipeline()

# 2. UI Header
st.title("🌱 Dry Bean Variety Classifier")

# Education Warning
st.warning(
    "⚠️ **Educational Application Only** - This is a demonstration project created for learning purposes. "
    "It should NOT be used for real-world bean classification or any production environment. "
    "Predictions may be inaccurate and are not guaranteed.",
    icon="📚"
)

st.markdown("""
Enter the physical characteristics of the bean below. 
The system uses an **SVC Model** with **Yeo-Johnson transformation** to predict the variety.
""")

# 3. Sidebar Inputs
st.sidebar.header("Bean Measurements")

def get_inputs():
    # Group 1: Features requiring Yeo-Johnson
    st.sidebar.subheader("Size & Shape (YJ Transformed)")
    area = st.sidebar.number_input("area", value=28395.0)
    perimeter = st.sidebar.number_input("perimeter", value=610.29)
    majoraxis = st.sidebar.number_input("majoraxislength", value=208.17)
    minoraxis = st.sidebar.number_input("minoraxislength", value=173.88)
    eccentricity = st.sidebar.number_input("eccentricity", value=0.54)
    solidity = st.sidebar.number_input("solidity", value=0.99)
    sf4 = st.sidebar.number_input("shapefactor4", value=0.99)
    
    # Group 2: Other features
    st.sidebar.subheader("Ratios & Factors")
    aspect = st.sidebar.number_input("aspectration", value=1.19)
    extent = st.sidebar.number_input("extent", value=0.76)
    roundness = st.sidebar.number_input("roundness", value=0.95)
    sf1 = st.sidebar.number_input("shapefactor1", value=0.0073)
    sf2 = st.sidebar.number_input("shapefactor2", value=0.001)
    sf3 = st.sidebar.number_input("shapefactor3", value=0.83)

    # Creating the dataframe in the exact order the pipeline expects
    data = {
        'area': area, 'perimeter': perimeter, 'majoraxislength': majoraxis,
        'minoraxislength': minoraxis, 'eccentricity': eccentricity,
        'solidity': solidity, 'shapefactor4': sf4,
        'aspectration': aspect, 'extent': extent, 'roundness': roundness,
        'shapefactor1': sf1, 'shapefactor2': sf2, 'shapefactor3': sf3
    }
    return pd.DataFrame([data])

input_df = get_inputs()

# 4. Prediction Logic
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Parameters")
    st.write(input_df.T.rename(columns={0: "Value"}))

with col2:
    if st.button("Predict Variety", use_container_width=True):
        # The pipeline automatically scales and transforms the data!
        prediction = pipeline.predict(input_df)[0]
        probs = pipeline.predict_proba(input_df)
        confidence = np.max(probs) * 100
        
        st.success(f"### Predicted Variety: **{prediction}**")
        st.metric("Confidence", f"{confidence:.2f}%")
        
        # Display Probability Chart
        st.write("#### Probability Breakdown")
        prob_df = pd.DataFrame(probs, columns=pipeline.classes_).T
        prob_df.columns = ["Probability"]
        st.bar_chart(prob_df)

st.divider()
st.caption("Note: Features 'convexarea', 'equivdiameter', and 'compactness' were removed to reduce multicollinearity.")
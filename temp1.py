import numpy as np
import pickle
import pandas as pd
import streamlit as st
pickle_in = open("admission_lr_model.pickle","rb")
classifier = pickle.load(pickle_in)
def college_prediction(GRE_Score,TOEFL_Score,University_Rating,SOP,LOR,CGPA,Research):
    prediction=classifier.predict([[GRE_Score,TOEFL_Score,University_Rating,SOP,LOR,CGPA,Research]])
    print(prediction)
    return prediction

def main():
    st.title("College-Admission-Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style ="color:white;text-align:center;">Streamlit college admission predictor</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    GRE_Score = st.number_input("GRE Score", value=0, min_value=0, max_value=340, step=1)
    TOEFL_Score = st.number_input("TOEFL Score", value=0, min_value=0, max_value=120, step=1)
    University_Rating = st.number_input("University rating", value=0, min_value=0, max_value=5, step=1)
    SOP = st.number_input("SOP", value=0.0, min_value=0.0, max_value=5.0, step=0.1)
    LOR = st.number_input("LOR", value=0.0, min_value=0.0, max_value=5.0, step=0.1)
    CGPA = st.number_input("CGPA", value=0.0, min_value=0.0, max_value=10.0, step=0.1)
    Research = st.number_input("Research", value=0, min_value=0, max_value=1, step=1)
    result=""
    if st.button("Predict"):
        result = college_prediction(GRE_Score,TOEFL_Score,University_Rating,SOP,LOR,CGPA,Research)
    st.success("The output is {}".format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Build with Streamlit")
if __name__=='__main__':
    main()

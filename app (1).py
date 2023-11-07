import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('model_forest_2.pkl','rb'))


def predict_forest(oxygen,humidity,temperature):
    input=np.array([[oxygen,humidity,temperature]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

def main():
    st.title("Machine Learning Tutorial Day-6")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Forest Fire Prediction Using Logistic Regression Model </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    oxygen = st.text_input("Oxygen","Please enter Oxygen Value")
    humidity = st.text_input("Humidity","Please enter Humidity Value")
    temperature = st.text_input("Temperature","Type Here")
    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> Your forest is safe! Enjoy!!!!!</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Your forest is in danger</h2>
       </div>
    """

    if st.button("Predict_Forest_Fire"):
        output=predict_forest(oxygen,humidity,temperature)
        st.success('The probability of fire taking place is {}'.format(output))

        if output > 0.5:
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()
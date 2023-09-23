import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

def ordinal_encoder(input_val, feats): 
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    value = feat_dict[input_val]
    return value


def get_prediction(data,model):
    return model.predict(data)


model = joblib.load(r'Model\best_randomforest_model.joblib')

st.set_page_config(page_title="Accident Severity Prediction App",page_icon="🚧", layout="wide")

st.title('Road Accident Severity Classification')


features = ['hour', 'day_of_week', 'cause_of_accident', 'number_of_casualties', 'number_of_vehicles_involved', 
                            'type_of_vehicle', 'area_accident_occured', 'types_of_junction', 'age_band_of_driver', 
                            'driving_experience', 'lanes_or_medians', 'light_conditions']

def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input to know the Severity of Accident")

        hour = st.number_input("Hour", 0, 23)

        day_of_week = st.selectbox("Day", ['Monday', 'Sunday', 'Friday', 'Wednesday', 'Saturday', 'Thursday',
       'Tuesday'])
        
        cause_of_accident = st.selectbox("Cause of Accident",['Moving Backward', 'Overtaking', 'Changing lane to the left',
       'Changing lane to the right', 'Overloading', 'No distancing',
       'No priority to vehicle', 'No priority to pedestrian',
       'Getting off the vehicle improperly', 'Improper parking',
       'Overspeed', 'Driving carelessly', 'Driving at high speed',
       'Driving to the left', 'Overturning', 'Turnover',
       'Driving under the influence of drugs', 'Drunk driving'])
        
        number_of_casualties = st.number_input("Number of casualities", 1,8)

        number_of_vehicles_involved = st.selectbox("Number of Vehicles involved",[2, 1, 3, 6, 4, 7])

        type_of_vehicle = st.selectbox("Type of Vehicle Involved",['Automobile', 'Public (> 45 seats)', 'Lorry (41?100Q)', 'Unknown',
       'Public (13?45 seats)', 'Lorry (11?40Q)', 'Long lorry',
       'Public (12 seats)', 'Taxi', 'Pick up upto 10Q', 'Stationwagen',
       'Ridden horse', 'Bajaj', 'Turbo', 'Motorcycle', 'Special vehicle',
       'Bicycle'])
        
        area_accident_occured = st.selectbox("Area Accident occured",['Residential areas', 'Office areas', '  Recreational areas',
       ' Industrial areas', 'Unknown', ' Church areas', '  Market areas',
       'Rural village areas', ' Outside rural areas', ' Hospital areas',
       'School areas', 'Rural village areasOffice areas',
       'Recreational areas'])
        
        types_of_junction = st.selectbox("Type of Junction",['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Unknown',
       'T Shape', 'X Shape'])
        
        age_band_of_driver = st.selectbox("Age of Driver",['18-30', '31-50', 'Under 18', 'Over 51', 'Unknown'])

        driving_experience = st.selectbox("Driver's Experience",['1-2yr', 'Above 10yr', '5-10yr', '2-5yr', 'No Licence',
       'Below 1yr'])
        
        lanes_or_medians = st.selectbox("Road Lanes",['Two-way (divided with broken lines road marking)',
       'Undivided Two way', 'other', 'Double carriageway (median)',
       'One way', 'Two-way (divided with solid lines road marking)'])
        
        light_conditions = st.selectbox("Light Conditions",['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
       'Darkness - lights unlit'])
        

        submit = st.form_submit_button("Predict")

        if submit:

            day_of_week= ordinal_encoder(day_of_week, ['Monday', 'Sunday', 'Friday', 'Wednesday', 'Saturday', 'Thursday',
                                                        'Tuesday'])
            cause_of_accident = ordinal_encoder(cause_of_accident, ['Moving Backward', 'Overtaking', 
                                                                    'Changing lane to the left',
                                                                    'Changing lane to the right', 'Overloading', 'No distancing',
                                                                    'No priority to vehicle', 'No priority to pedestrian',
                                                                    'Getting off the vehicle improperly', 'Improper parking',
                                                                    'Overspeed', 'Driving carelessly', 'Driving at high speed',
                                                                    'Driving to the left', 'Overturning', 'Turnover',
                                                                    'Driving under the influence of drugs', 'Drunk driving'])
            type_of_vehicle = ordinal_encoder(type_of_vehicle, ['Automobile', 'Public (> 45 seats)', 'Lorry (41?100Q)',
                                                                 'Unknown','Public (13?45 seats)', 'Lorry (11?40Q)', 
                                                                 'Long lorry','Public (12 seats)', 'Taxi', 'Pick up upto 10Q',
                                                                   'Stationwagen','Ridden horse', 'Bajaj', 'Turbo', 
                                                                   'Motorcycle', 'Special vehicle','Bicycle'])
            area_accident_occured = ordinal_encoder(area_accident_occured, ['Residential areas', 'Office areas', 
                                                                            '  Recreational areas',' Industrial areas', 
                                                                            'Unknown', ' Church areas', '  Market areas',
                                                                            'Rural village areas', ' Outside rural areas',
                                                                            ' Hospital areas','School areas', 
                                                                            'Rural village areasOffice areas','Recreational areas'])
            types_of_junction = ordinal_encoder(types_of_junction, ['No junction', 'Y Shape', 'Crossing', 'O Shape', 
                                                                    'Unknown','T Shape', 'X Shape'])
            age_band_of_driver = ordinal_encoder(age_band_of_driver, ['18-30', '31-50', 'Under 18', 'Over 51', 'Unknown'])
            driving_experience = ordinal_encoder(driving_experience, ['1-2yr', 'Above 10yr', '5-10yr', '2-5yr', 
                                                                      'No Licence','Below 1yr'])
            lanes_or_medians = ordinal_encoder(lanes_or_medians, ['Two-way (divided with broken lines road marking)',
                                                                'Undivided Two way', 'other', 'Double carriageway (median)',
                                                                'One way', 'Two-way (divided with solid lines road marking)'])
            light_conditions = ordinal_encoder(light_conditions, ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
       'Darkness - lights unlit'])




        data = np.array([hour, day_of_week, cause_of_accident, number_of_casualties, number_of_vehicles_involved, 
                            type_of_vehicle,area_accident_occured,types_of_junction, age_band_of_driver, 
                            driving_experience, lanes_or_medians, light_conditions]).reshape(1,-1)
        
        pred = get_prediction(data,model)

        st.write(f"The predicted severity is:  {pred[0]}")

if __name__ == '__main__':
    main()



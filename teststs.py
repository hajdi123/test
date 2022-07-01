# %%
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
# %%
import pandas as pd
import numpy as np
import pickle
import altair as alt
import altair_viewer
import seaborn as sns
import matplotlib.pyplot as plt

df = (pd.read_csv(r'C:\VistraBox\Box\EquipMonInputData\ML1_EquipMonInputData-3Yr Data.csv',na_values=["[-11059] No Good Data For Calculation"]))

df = df.rename(
    columns={
        'DCS_ML1_GROSS_GENERATION': 'gross_power', 
        'Martin Lake 1_Boiler_Site Ambient Air Temp': 'ambient_temp',
        'Martin Lake 1_Air Heater A_Flue Gas Inlet Temperature 1': 'gi_temp1_a',
        'Martin Lake 1_Air Heater A_Flue Gas Inlet Temperature 2':'gi_temp2_a',
        'Martin Lake 1_Air Heater A_Flue Gas Outlet Temperature 1':'go_temp1_a',
        'Martin Lake 1_Air Heater A_Flue Gas Outlet Temperature 2':'go_temp2_a',
        'Martin Lake 1_Air Heater B_Flue Gas Inlet Temperature 1':'gi_temp1_b',
        'Martin Lake 1_Air Heater B_Flue Gas Inlet Temperature 2':'gi_temp2_b',
        'Martin Lake 1_Air Heater B_Flue Gas Outlet Temperature 1':'go_temp1_b',
        'Martin Lake 1_Air Heater B_Flue Gas Outlet Temperature 2':'go_temp2_b',
        'DCS_ML1_DE_1A-ID_FAN_AMPS':'fan_a_amps',
        'DCS_ML1_DE_1B-ID_FAN_AMPS':'fan_b_amps',
        'DCS_ML1_DE_1C-ID_FAN_AMPS':'fan_c_amps',
        'DCS_ML1_DE_1D-ID_FAN_AMPS':'fan_d_amps',
        'Martin Lake 1_Induced Draft Fan A_Inlet Damper Position':'fan_a_damper',
        'Martin Lake 1_Induced Draft Fan B_Inlet Damper Position':'fan_b_damper',
        'Martin Lake 1_Induced Draft Fan C_Inlet Damper Position':'fan_c_damper',
        'Martin Lake 1_Induced Draft Fan D_Inlet Damper Position':'fan_d_damper',
        'DCS_ML1_BA_FURNACE_PRESSURE':'furnace_pressure',
        'Martin Lake 1_Air Heater_Flue Gas Inlet Pressure':'gas_inlet_pressure',
        'Martin Lake 1_Air Heater A_Flue Gas Side Prress Drop':'gas_pressure_drop_a',
        'Martin Lake 1_Air Heater B_Flue Gas Side Prress Drop':'gas_pressure_drop_b',
        'Martin Lake 1_Air Heater A_Flue Gas Outlet Pressure':'gas_out_pressure_a',
        'Martin Lake 1_Air Heater B_Flue Gas Outlet Pressure':'gas_out_pressure_b',
        'DCS_ML1_CD_COND-1_INLET_WTR_T': 'circ_wtr_inlet_1_a',                                 
        'DCS_ML1_CD_COND-2_INLET_WTR_T': 'circ_wtr_inlet_2_a',
        'DCS_ML1_CD_COND-3_INLET_WTR_T':'circ_wtr_inlet_3_b',
        'DCS_ML1_CD_COND-4_INLET_WTR_T':'circ_wtr_inlet_4_b',
        'DCS_ML1_CO_COND-A_VACUUM':'condenser_vacuum_a',
        'DCS_ML1_CO_COND-B_VACUUM':'condenser_vacumm_b',
        'DCS_ML1_BAROMETRIC_PRESS':'barometric_pressure',
        'DCS_ML1_CW_COND-1_OUT-TEMP':'north_circ_water_2',
        'DCS_ML1_CW_COND-2_OUT-TEMP':'north_center_water_2',
        'DCS_ML1_CW_COND-3_OUT-TEMP':'south_center_water_2',
        'DCS_ML1_CW_COND-4_OUT-TEMP':'south_circ_water_2',
        'DCS_ML1_CW_1A-CIRC_PMP_MTR_AMP':'circ_water_pump_a',
        'DCS_ML1_CW_1B-CIRC_PMP_MTR_AMP':'circ_water_pump_b',
        'DCS_ML1_CW_1C-CIRC_PMP_MTR_AMP':'circ_water_pump_c',                               
        'DCS_ML1_CO_COND_AIR_INLEAKAGE':'condenser_air_leakage'  
        })


# %%
df['Date'] = df['Date'].astype('datetime64[ns]')
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year

# Here I will be creating a function to show the limits for a season variable
def season(month):
    if (month == 12 or 1 <= month <= 4):
        return "winter"   
    elif (4 <= month <= 5):
        return "spring" 
    elif (6 <= month <= 9):
        return "summer"
    else:
        return "fall"
# Now I create the variable season while passing month into the function
df['Season']= df['month'].apply(season)

# %%

df['north_circ_wtr_temp_rise_a'] = df['north_circ_water_2'] - df['circ_wtr_inlet_1_a']
df['north_center_wtr_temp_rise_a']=df['north_center_water_2'] - df['circ_wtr_inlet_2_a']

df['south_center_wtr_temp_rise_b'] = df['south_center_water_2'] - df['circ_wtr_inlet_3_b']
df['south_circ_wtr_temp_rise_b'] = df['south_circ_water_2'] - df['circ_wtr_inlet_4_b']


# %%
df =(df.dropna(subset =['gross_power','ambient_temp','condenser_vacuum_a',
    'condenser_vacumm_b','barometric_pressure','north_center_wtr_temp_rise_a',
    'north_circ_water_2','north_center_water_2','south_circ_water_2',
    'south_center_water_2','circ_water_pump_a','circ_water_pump_c',
    'north_circ_wtr_temp_rise_a','condenser_air_leakage',
    'south_center_wtr_temp_rise_b','south_circ_wtr_temp_rise_b',
    'circ_water_pump_a','circ_water_pump_b','circ_water_pump_c',
    'month','day','Season']))

# %%
df = (df
                 .query('gross_power > 200')
                 .query('30 <= ambient_temp')
                 )


# %%
alt.data_transformers.disable_max_rows()

# %%
def switch(x):
    if x > 50:
        return 'On'
    else:
        return 'Off'

df['Water Pump A Switch'] = df['circ_water_pump_a'].apply(switch)
df['Water Pump B Switch'] = df['circ_water_pump_b'].apply(switch)
df['Water Pump C Switch'] = df['circ_water_pump_c'].apply(switch)

# %%
one_hot = pd.get_dummies(df[['Season','Water Pump A Switch','Water Pump B Switch','Water Pump C Switch']])
df = df.drop(['Season','Water Pump A Switch','Water Pump B Switch','Water Pump C Switch'],axis = 1)
df = df.join(one_hot)

# %%
df['sum_of_amps'] = df['circ_water_pump_a']+df['circ_water_pump_b']+df['circ_water_pump_c']

# %%
X1 = (df[['gross_power','condenser_air_leakage','sum_of_amps',
        'Season_fall','Season_spring','Season_summer','Season_winter',
        'Water Pump B Switch_On','Water Pump B Switch_Off',
        'Water Pump A Switch_On','Water Pump A Switch_Off',
        'Water Pump C Switch_On','Water Pump C Switch_Off'
        ]])
y = df[['north_center_wtr_temp_rise_a']]


# %%
from sklearn.preprocessing import RobustScaler,MinMaxScaler ,StandardScaler
scaler =RobustScaler()
X=scaler.fit_transform(X1)

# %%
from sklearn.linear_model import BayesianRidge,	LinearRegression, Ridge,Lasso
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import log

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
regressor = XGBRegressor()
regressor.fit(X_train, y_train)



# %%
pickle.load(open('ML1 Condenser Temp Rise North Center A.pkl', 'rb'))

# %%
df['Predicted_north_center_wtr_temp_rise_a'] = regressor.predict(X)

# %%
df['absolute_north_center_wtr_temp_rise_a'] = df['north_center_wtr_temp_rise_a']-df['Predicted_north_center_wtr_temp_rise_a']

# %%
chart12 = alt.Chart(df, title = 'Martin Lake:1A Condenser Actual-Expected Temp Rise ').mark_point(
    size=10,
    opacity = 1.0,
    filled=True,
    color='black').encode(

    x=alt.X('Date:T',
     scale=alt.Scale(zero=False)),
    y=alt.Y('absolute_north_center_wtr_temp_rise_a',title = 'Act-Expected Temp Rise',scale=alt.Scale(domain=[-5, 5])),
    ).properties(width=1000,height=700).interactive()
    
line1 = alt.Chart(pd.DataFrame({'y': [0.5]})).mark_rule(color = 'red', size=2).encode(y=alt.Y('y', title = ''))
line2 = alt.Chart(pd.DataFrame({'y': [-0.5]})).mark_rule(color = 'red',size=2).encode(y=alt.Y('y', title = ''))

chart23 = line1 + line2 + chart12
chart23.configure_axis(
    labelFontSize=24,
    titleFontSize=24
).configure_title(fontSize=35)
altair_viewer.display(chart23)



st.altair_chart(chart23)
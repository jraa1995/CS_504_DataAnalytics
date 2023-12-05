# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:11:35 2023

@author: hoeri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('C:/Users/hoeri/Documents/CS504/Project/Crash_Reporting_Drivers_Data_20231104_without_Duplicated_Report_Number.csv')

#Examine the dataframe
df.head()
df.values
df.columns
df.describe()
df.dtypes
df.shape
df.size

#Analysizing the time of accidents
# Define a specific timeframe (e.g., yearly, monthly, and hourly)
df.shape
df['Crash Date/Time'].dtypes
df['Crash Date/Time'] = pd.to_datetime(df['Crash Date/Time'])
df['Crash Date/Time'].dtypes
# Extract year from the Timestamp column to create Year column
df['Year'] = df['Crash Date/Time'].dt.to_period('Y')
# Extract month from the Timestamp column Month column
df['Month'] = df['Crash Date/Time'].dt.to_period('M')
# Extract day from the Timestamp column to create Hour column
df['Day'] = df['Crash Date/Time'].dt.to_period('D')
# Extract hour from the Timestamp column to create Hour column
df['Hour'] = df['Crash Date/Time'].dt.hour

#Accidents in earch year included in the dataset
plt.plot = df['Year'].value_counts().sort_index(ascending=True).plot.bar()

#Find the most 24 common Month and 
#filter the dataframe (df) to include only rows that have the most 24 common Month
top_month = df['Month'].value_counts().head(24).index
df_filtered = df[df['Month'].isin(top_month)]
df = df_filtered
df.shape
plt.plot = df['Month'].value_counts().sort_index(ascending=True).plot.bar()

#Find the most 5 common Day and 
#filter the dataframe (df) to include only rows that have the most 5 common Day
#top_day = df['Day'].value_counts().head(30).index
#df_filtered = df[df['Day'].isin(top_day)]
#df = df_filtered
#df.shape
#plt.plot = df['Day'].value_counts().sort_index(ascending=True).plot.bar()

#Find the most 12 common Hour and 
#filter the dataframe (df) to include only rows that have the most 12 common Hour
top_hour = df['Hour'].value_counts().head(12).index
df_filtered = df[df['Hour'].isin(top_hour)]
df = df_filtered
df.shape
plt.plot = df['Hour'].value_counts().sort_index(ascending=True).plot.bar()

#Find the most 5 common Municipals and 
#filter the dataframe (df) to include only rows that have the most 5 common Municipals
top_municipal = df['Municipality'].value_counts().head(5).index
df_filtered = df[df['Municipality'].isin(top_municipal)]
df = df_filtered
df.shape
plt.plot = df['Municipality'].value_counts().sort_index(ascending=True).plot.bar()

#Cross reference Municipality and Year columns
MunicipalityYear = pd.crosstab(index=df['Municipality'], columns=df['Year'])
MunicipalityYear.head()
plt.plot = MunicipalityYear.plot.bar()

#Cross reference Municipality and Month columns
MunicipalityMonth = pd.crosstab(index=df['Municipality'], columns=df['Month'])
MunicipalityMonth.head()
plt.plot = MunicipalityMonth.plot.bar()

#Cross reference Municipality and Hour columns
MunicipalityHour = pd.crosstab(index=df['Municipality'], columns=df['Hour'])
MunicipalityHour.head()
plt.plot = MunicipalityHour.plot.bar()

#Keep a copy of this df
df_copy = df
df.shape

#Find the most 5 common Collision Type and 
#filter the dataframe (df) to include only rows that have the most common 5 Collision Type
top_collision = df['Collision Type'].value_counts().head(5).index
df_filtered = df[df['Collision Type'].isin(top_collision)]
df = df_filtered
df.shape
plt.plot = df['Collision Type'].value_counts().plot.bar()

#Cross reference Municipality and Collision Type columns
MunicipalityCollision = pd.crosstab(index=df['Municipality'], columns=df['Collision Type'])
MunicipalityCollision.head()
plt.plot = MunicipalityCollision.plot.bar()

#Reset df
df = df_copy
df.shape

#Find the most 5 common Vehicle Movement and 
#filter the dataframe (df) to include only rows that have the most 5 common Vehicle Movement
top_movement = df['Vehicle Movement'].value_counts().head(5).index
df_filtered = df[df['Vehicle Movement'].isin(top_movement)]
df = df_filtered
df.shape
plt.plot = df['Vehicle Movement'].value_counts().plot.bar()

#Cross reference Municipality and Speed Limit columns
MunicipalitySpeed = pd.crosstab(index=df['Municipality'], columns=df['Vehicle Movement'])
MunicipalitySpeed.head()
plt.plot = MunicipalitySpeed.plot.bar()

#Reset df
df = df_copy
df.shape

#Find the most 5 common Weather and 
#filter the dataframe (df) to include only rows that have the most common 5 Weather
top_weather = df['Weather'].value_counts().head(5).index
df_filtered = df[df['Weather'].isin(top_weather)]
df = df_filtered
df.shape
plt.plot = df['Weather'].value_counts().plot.bar()

#Cross reference Municipality and Weather columns
MunicipalityWeather = pd.crosstab(index=df['Municipality'], columns=df['Weather'])
MunicipalityWeather.head()
plt.plot = MunicipalityWeather.plot.bar()

#Reset df
df = df_copy
df.shape

#Find the most 5 common Surface Condition and 
#filter the dataframe (df) to include only rows that have the most common 5 Surface Condition
top_surface = df['Surface Condition'].value_counts().head(5).index
df_filtered = df[df['Surface Condition'].isin(top_surface)]
df = df_filtered
df.shape
plt.plot = df['Surface Condition'].value_counts().plot.bar()

#Cross reference Municipality and Surface Condition columns
MunicipalitySurface = pd.crosstab(index=df['Municipality'], columns=df['Surface Condition'])
MunicipalitySurface.head()
plt.plot = MunicipalitySurface.plot.bar()

#Reset df
df = df_copy
df.shape

#Find the most 5 common Light and 
#filter the dataframe (df) to include only rows that have the most common 5 Light
top_light = df['Light'].value_counts().head(5).index
df_filtered = df[df['Light'].isin(top_light)]
df = df_filtered
df.shape
plt.plot = df['Light'].value_counts().plot.bar()

#Cross reference Municipality and Light columns
MunicipalityLight = pd.crosstab(index=df['Municipality'], columns=df['Light'])
MunicipalityLight.head()
plt.plot = MunicipalityLight.plot.bar()

#Reset df
df = df_copy
df.shape

#Find the most 5 common Traffic Control and 
#filter the dataframe (df) to include only rows that have the most common 5 Traffic Control
top_traffic = df['Traffic Control'].value_counts().head(5).index
df_filtered = df[df['Traffic Control'].isin(top_traffic)]
df = df_filtered
df.shape
plt.plot = df['Traffic Control'].value_counts().plot.bar()

#Cross reference Municipality and Traffic Control columns
MunicipalityTraffic = pd.crosstab(index=df['Municipality'], columns=df['Traffic Control'])
MunicipalityTraffic.head()
plt.plot = MunicipalityTraffic.plot.bar()

#Reset df
df = df_copy
df.shape

#Find the most 5 common Driver Substance Abuse and 
#filter the dataframe (df) to include only rows that have the most common 5 Substance Abuse
top_substance = df['Driver Substance Abuse'].value_counts().head(5).index
df_filtered = df[df['Driver Substance Abuse'].isin(top_substance)]
df = df_filtered
df.shape
plt.plot = df['Driver Substance Abuse'].value_counts().plot.bar()

#Cross reference Municipality and Driver Substance Abuse columns
MunicipalityDriverSubstanceAbuse = pd.crosstab(index=df['Municipality'], columns=df['Driver Substance Abuse'])
MunicipalityDriverSubstanceAbuse.head()
plt.plot = MunicipalityDriverSubstanceAbuse.plot.bar()

#Reset df
df = df_copy
df.shape

#Find the most 5 common Circumstacnce and 
#filter the dataframe (df) to include only rows that have the most 5 common Circumstance
top_circumstance = df['Circumstance'].value_counts().head(5).index
df_filtered = df[df['Circumstance'].isin(top_circumstance)]
df = df_filtered
df.shape
plt.plot = df['Circumstance'].value_counts().plot.bar()

#Cross reference Municipality and top categories of Circumstance column
MunicipalityCircumstance = pd.crosstab(index=df['Municipality'], columns=df['Circumstance'])
MunicipalityCircumstance.head()
plt.plot = MunicipalityCircumstance.plot.bar()

#Reset df
df = df_copy
df.shape

#Find the most 5 common Driver Distracted By and 
#filter the dataframe (df) to include only rows that have the most 5 common circumstances
top_distraction = df['Driver Distracted By'].value_counts().head(5).index
df_filtered = df[df['Driver Distracted By'].isin(top_distraction)]
df = df_filtered
df.shape
plt.plot = df['Driver Distracted By'].value_counts().plot.bar()

#Cross reference Municipality and top categories of Circumstance column
MunicipalityDistraction = pd.crosstab(index=df['Municipality'], columns=df['Driver Distracted By'])
MunicipalityDistraction.head()
plt.plot = MunicipalityDistraction.plot.bar()

#Reset df
df = df_copy
df.shape

#Find the most 5 common Drivers License State and 
#filter the dataframe (df) to include only rows that have the most 5 common Drivers License State
top_license = df['Drivers License State'].value_counts().head(5).index
df_filtered = df[df['Drivers License State'].isin(top_license)]
df = df_filtered
df.shape
plt.plot = df['Drivers License State'].value_counts().plot.bar()

#Cross reference Municipality and top categories of Drivers License State column
MunicipalityLicense = pd.crosstab(index=df['Municipality'], columns=df['Drivers License State'])
MunicipalityLicense.head()
plt.plot = MunicipalityLicense.plot.bar()

#Reset df
df = df_copy
df.shape

#Find the most 5 common Speed Limit and 
#filter the dataframe (df) to include only rows that have the most 5 common Speed Limit
top_license = df['Speed Limit'].value_counts().head(5).index
df_filtered = df[df['Speed Limit'].isin(top_license)]
df = df_filtered
df.shape
plt.plot = df['Speed Limit'].value_counts().plot.bar()

#Cross reference Municipality and Speed Limit columns
MunicipalitySpeed = pd.crosstab(index=df['Municipality'], columns=df['Speed Limit'])
MunicipalitySpeed.head()
plt.plot = MunicipalitySpeed.plot.bar()

#Reset df
df = df_copy
df.shape

#Find the most 5 common Vehicle Damage Extent and 
#filter the dataframe (df) to include only rows that have the most 5 common Vehicle Damage Extent
top_damage = df['Vehicle Damage Extent'].value_counts().head(5).index
df_filtered = df[df['Vehicle Damage Extent'].isin(top_damage)]
df = df_filtered
df.shape
plt.plot = df['Vehicle Damage Extent'].value_counts().plot.bar()

#Cross reference Municipality and Speed Limit columns
MunicipalitySpeed = pd.crosstab(index=df['Municipality'], columns=df['Vehicle Damage Extent'])
MunicipalitySpeed.head()
plt.plot = MunicipalitySpeed.plot.bar()

#Reset df
df = df_copy
df.shape

#Find the most 5 common Vehicle Body Type and 
#filter the dataframe (df) to include only rows that have the most 5 common Vehicle Damage Vehicle Body Type
top_damage = df['Vehicle Body Type'].value_counts().head(5).index
df_filtered = df[df['Vehicle Body Type'].isin(top_damage)]
df = df_filtered
df.shape
plt.plot = df['Vehicle Body Type'].value_counts().plot.bar()

#Cross reference Municipality and Speed Limit columns
MunicipalitySpeed = pd.crosstab(index=df['Municipality'], columns=df['Vehicle Body Type'])
MunicipalitySpeed.head()
plt.plot = MunicipalitySpeed.plot.bar()

#Reset df
df = df_copy
df.shape

#Find the most 5 common Route Type and 
#filter the dataframe (df) to include only rows that have the most 5 common Route Type
top_damage = df['Route Type'].value_counts().head(5).index
df_filtered = df[df['Route Type'].isin(top_damage)]
df = df_filtered
df.shape
plt.plot = df['Route Type'].value_counts().plot.bar()

#Cross reference Municipality and Speed Limit columns
MunicipalitySpeed = pd.crosstab(index=df['Municipality'], columns=df['Route Type'])
MunicipalitySpeed.head()
plt.plot = MunicipalitySpeed.plot.bar()

#Reset df
df = df_copy
df.shape

#Find the most 5 common Road Name and 
#filter the dataframe (df) to include only rows that have the most 5 common Road Name
top_damage = df['Road Name'].value_counts().head(5).index
df_filtered = df[df['Road Name'].isin(top_damage)]
df = df_filtered
df.shape
plt.plot = df['Road Name'].value_counts().plot.bar()

#Cross reference Municipality and Speed Limit columns
MunicipalitySpeed = pd.crosstab(index=df['Municipality'], columns=df['Road Name'])
MunicipalitySpeed.head()
plt.plot = MunicipalitySpeed.plot.bar()

#Reset df
df = df_copy
df.shape

#Find the most 5 common Cross-Street Type and 
#filter the dataframe (df) to include only rows that have the most 5 common Cross-Street Type
top_damage = df['Cross-Street Type'].value_counts().head(5).index
df_filtered = df[df['Cross-Street Type'].isin(top_damage)]
df = df_filtered
df.shape
plt.plot = df['Cross-Street Type'].value_counts().plot.bar()

#Cross reference Municipality and Cross-Street Type columns
MunicipalitySpeed = pd.crosstab(index=df['Municipality'], columns=df['Cross-Street Type'])
MunicipalitySpeed.head()
plt.plot = MunicipalitySpeed.plot.bar()

#Reset df
df = df_copy
df.shape

#Find the most 5 common Cross-Street Name and 
#filter the dataframe (df) to include only rows that have the most 5 common Cross-Street Name
top_damage = df['Cross-Street Name'].value_counts().head(5).index
df_filtered = df[df['Cross-Street Name'].isin(top_damage)]
df = df_filtered
df.shape
plt.plot = df['Cross-Street Name'].value_counts().plot.bar()

#Cross reference Municipality and Speed Limit columns
MunicipalitySpeed = pd.crosstab(index=df['Municipality'], columns=df['Cross-Street Name'])
MunicipalitySpeed.head()
plt.plot = MunicipalitySpeed.plot.bar()

#Reset df
df = df_copy
df.shape

#Find the most 5 common ACRS Report Type and 
#filter the dataframe (df) to include only rows that have the most 5 common ACRS Report Type
top_damage = df['ACRS Report Type'].value_counts().head(5).index
df_filtered = df[df['ACRS Report Type'].isin(top_damage)]
df = df_filtered
df.shape
plt.plot = df['ACRS Report Type'].value_counts().plot.bar()

#Cross reference Municipality and Speed Limit columns
MunicipalitySpeed = pd.crosstab(index=df['Municipality'], columns=df['ACRS Report Type'])
MunicipalitySpeed.head()
plt.plot = MunicipalitySpeed.plot.bar()


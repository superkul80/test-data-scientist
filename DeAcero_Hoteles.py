# -*- coding: utf-8 -*-
"""
Autor: Dan-El Neil Vila Rosado
Actividad: Analisis de datos de Hoteles en Portugal
"""


 # Carga de datos

import urllib3
from urllib3 import request
import certifi
import json

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

# Coneccion
http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
url = 'https://analytics.deacero.com/api/teenus/get-data/c744a2a4-ab89-5432-b5e6-9f320162e160?format=json'
outGet = http.request('GET', url)
outGet.status
print(outGet.status)

# Guardamos en formao json
data = json.loads(outGet.data.decode('utf-8'))

# Creamos Diccionario de datos
data = json.loads(outGet.data.decode('utf-8'))

# Guardamos datos en archivo
print(data[0])
with open('json_data.json', 'w') as outfile:
    json.dump(data, outfile)
    
# Normalizamos a partir de datos json
df_original = pd.json_normalize(data)

df = df_original.copy()

# Exploracion inicial
NumRegistros = df.shape[0]
NumVariables = df.shape[1]
Variables = df.columns
df.info() # Tipo de datos

# Identificamos valores perdidos
import missingno
missingno.bar(df)

# Valores distintos que tiene cada variable
for var in Variables:
  datos_unicos = df[var].unique()
  print(var, '(', len(datos_unicos), ') : ', datos_unicos)

# Variables con tipos de valores a considerar 
plt_children = df.children.value_counts().sort_values().plot(kind = 'barh')
plt_children.bar_label(plt_children.containers[0])
df['children'] = df['children'].replace('NA',0)
"""
Children
¿Que significa los valores NA? 
Existe 0,1,2,3,10, NA (4 registros)
Reemplazado con 0
Debilidad del sistema de registro
"""

plt_meal = df.meal.value_counts().sort_values().plot(kind = 'barh')
plt_meal.bar_label(plt_meal.containers[0])
meal_filter = df['meal']=='Undefined'
df_meal_undefined = df.where(meal_filter)
""" 
Meal
¿Que significa la categoria Indefinido?
Son 1169 registros
Oportunidad para el negocio. Investigacion sobre la variable
"""

plt_country = df.country.value_counts()[:25].sort_values().plot(kind = 'barh')
plt_country.bar_label(plt_country.containers[0])
# remplazar Null en columna 'country'
df['country'] = df['country'].replace('NULL','UNKNOWN')
"""
Country
Elementos NULL (488)
Reemplazado por 'UNKNOW'
Debilidad del sistema de registro, Oportunidad de mejora. 
Investigacion sobre la variable
"""

plt_market_segment = df.market_segment.value_counts().sort_values().plot(kind = 'barh')
plt_market_segment.bar_label(plt_market_segment.containers[0])
""" 
Market segment
¿Que significa la categoria Indefinido?
Indefinidos (2)
Debilidad del sistema de registro. Oportunidad de mejora
Investigacion sobre la variable
""" 

plt_distribution_channel = df.distribution_channel.value_counts().sort_values().plot(kind = 'barh')
plt_distribution_channel.bar_label(plt_distribution_channel.containers[0])
""" 
Distribution channel
¿Que significa la categoria Indefinido?
Indefinidos (5)
Debilidad del sistema de registro, Oportunidad de mejora. 
Investigacion sobre la variable
"""

plt_agent = df.agent.value_counts()[:10].sort_values().plot(kind = 'barh')
plt_agent.bar_label(plt_agent.containers[0]) 
df['agent'] = df['agent'].replace('NULL',0)
""" 
Agent
Los Null son una cantidad considerable 
Segundo grupo mas numeroso (16340)
Reemplazado por 0
Oportunidad de mejora. Investigación sobre la variable
"""

plt_company = df.company.value_counts()[:10].sort_values().plot(kind = 'barh')
plt_company.bar_label(plt_company.containers[0]) 
df['company'] = df['company'].replace('NULL',0)
"""
Company
Los Null son una cantidad considerable
Grupo mas numeroso (112593)
Reemplazado por 0
Oportunidad de mejora. Analisis de las compañias que mas ganancia dejan.
Determinar mas informacon de las compañias
"""

plt_reservation_status = df.reservation_status.value_counts().sort_values().plot(kind = 'barh')
plt_reservation_status.bar_label(plt_reservation_status.containers[0])
"""
Definición de huesped:= reservation_status = Check-Out
Posible Amenaza al negocio:= Numero de Cancelaciones (42993)
"""

# Convertimos a valor numerico
cols = ['is_canceled', 'lead_time', 'arrival_date_year', 
        'arrival_date_week_number', 'arrival_date_day_of_month', 
        'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 
        'children','babies', 'is_repeated_guest',
        'previous_cancellations', 'previous_bookings_not_canceled', 
        'booking_changes','agent','company',
        'days_in_waiting_list', 'adr','required_car_parking_spaces', 
        'total_of_special_requests']

df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
df.info()

Descripcion_Variables_Numericas = df.describe()

# Visualicemos boxplots
out_cols = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 
            'adults', 'babies', 'previous_cancellations', 
            'previous_bookings_not_canceled',
            'booking_changes', 'days_in_waiting_list', 
            'adr', 'required_car_parking_spaces','total_of_special_requests']

plt.figure(figsize=(20,30))

# graficar boxplots en grupos
for i, col in enumerate(out_cols[0:3]):
  plt.subplot(1,3,i+1)
  sns.boxplot(df[col])
  plt.xlabel('')
  plt.ylabel(col)
  
for i, col in enumerate(out_cols[3:6]):
  plt.subplot(1,3,i+1)
  sns.boxplot(df[col])
  plt.xlabel('')
  plt.ylabel(col)
  
for i, col in enumerate(out_cols[6:9]):
  plt.subplot(1,3,i+1)
  sns.boxplot(df[col])
  plt.xlabel('')
  plt.ylabel(col)
  
for i, col in enumerate(out_cols[9:12]):
  plt.subplot(1,3,i+1)
  sns.boxplot(df[col])
  plt.xlabel('')
  plt.ylabel(col)
  
"""
Variables a analizar por sus minimos o maximos. Busqueda de beneficios
o perjuicios para el negocio.

'stays_in_weekend_nights' (valor máximo)
'stays_in_week_nights' (valor máximo)
'adults' (maximo)
'babies' (maximo)
'previous_cancellations' (maximo)
'booking_changes' (maximo)
'days_in_waiting_list' (maximo)
'adr' (minimo)
"""

# Validacion de datos
df = df[df.adr >= 0] #Del analisis de minimos y outliers y dado el concepto
Reservacion_0 = df[((df.adults + df.children + df.babies == 0))].index # Reservacion No valida
df = df.drop(Reservacion_0)
# Adr con tipo de habitacion
# Cancelacion contra penalizacion (Adr)
# Verificar ocupacion

# Periodo de datos de llegadas
print('Periodo de datos de llegadas')
df_periodo = df.groupby(['arrival_date_year','arrival_date_month']) 
print(df_periodo.size())
"""
Julio de 2015 a Agosto 2017
"""
############################################################

# Preguntas a responder

"""
1. ¿De dónde vienen los huéspedes?
"""

# Obtenemos los registros con Check-out
df_Huespedes = df[df.reservation_status == 'Check-Out']
Huespedes_X_Pais = df_Huespedes.country.value_counts().sort_values(ascending=False)
print(Huespedes_X_Pais)
Huespedes_X_Pais_Porc = Huespedes_X_Pais / Huespedes_X_Pais.sum() *100
print(Huespedes_X_Pais_Porc)
plt_Huespedes = Huespedes_X_Pais_Porc[:10].plot.bar()
plt_Huespedes.bar_label(plt_Huespedes.containers[0])
plt_Huespedes.legend(title = "Porcentaje de Huespedes X Pais")

"""
2. ¿Cuánto pagan los huéspedes por una habitación por noche en promedio?
"""
# promedio de costo de la noche (City Hotel)
print('Promedio del costo por noche (City Hotel): ', round(df[df['is_canceled'] == 0]['adr'][df['hotel']=='City Hotel'].mean(),4))
# promedio de costo de la noche (Resort Hotel)
print('Costo promedio por noche (Resort Hotel): ', round(df[df['is_canceled'] == 0]['adr'][df['hotel']=='Resort Hotel'].mean(),4))
# Promedio y desv estandar por tipo de habitacion
df_City_Hotel = (df[df['hotel']=='City Hotel'].groupby('reserved_room_type')['adr'].mean()).to_frame()
df_City_Hotel['std'] = df[df['hotel']=='City Hotel'].groupby('reserved_room_type')['adr'].std()
df_Resort_Hotel = (df[df['hotel']=='Resort Hotel'].groupby('reserved_room_type')['adr'].mean()).to_frame()
df_Resort_Hotel['std'] = df[df['hotel']=='Resort Hotel'].groupby('reserved_room_type')['adr'].std()
# Plot error bar
plt.figure(figsize=(20,30))
plt.errorbar(df_City_Hotel.index, df_City_Hotel['adr'], yerr=df_City_Hotel['std'], fmt='o', marker='o',
             markersize=10,markeredgewidth=0.5,markeredgecolor='black',
             elinewidth=9,capsize=10,color='red',label='City Hotel')		
plt.errorbar(df_Resort_Hotel.index, df_Resort_Hotel['adr'], yerr=df_Resort_Hotel['std'], fmt='o', marker='o',
             markersize=10,markeredgewidth=0.5,markeredgecolor='black',
             elinewidth=3,capsize=10,color='black',label='Hotel Resort')

plt.xlabel('Tipo de habitación', fontsize=16)
plt.ylabel('Costo ',  fontsize=16)
plt.legend(fontsize=16)

"""
3. ¿Cómo varía el precio por noche durante el año?
"""
Orden_meses = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
df_City_Hotel_monthly = (df[df['is_canceled'] == 0][df['hotel'] == 'City Hotel'].groupby(['arrival_date_month','arrival_date_year'])['adr'].mean()).unstack()
#df_City_Hotel_monthly.columns = ['Costo X noche (Promedio Mensual)']
df_City_Hotel_monthly = df_City_Hotel_monthly.reindex(Orden_meses, axis=0).plot()
df_City_Hotel_monthly.plot.line(linewidth=3.0, color= 'red', figsize=(20,30))
#from scipy.stats import linregress
#slope, intercept, r_value, p_value, std_err = linregress(df_City_Hotel_monthly.index,df_City_Hotel_monthly.to_numpy())
#print("slope: %f, intercept: %f" % (slope, intercept))
#print("R-squared: %f" % r_value**2)

df_Resort_Hotel_monthly = (df[df['is_canceled'] == 0][df['hotel'] == 'Resort Hotel'].groupby(['arrival_date_month','arrival_date_year'])['adr'].mean()).unstack()
#df_Resort_Hotel_monthly.columns = ['Costo X noche (Promedio Mensual)']
df_Resort_Hotel_monthly = df_Resort_Hotel_monthly.reindex(Orden_meses, axis=0).plot()
#from scipy.stats import linregress
#slope, intercept, r_value, p_value, std_err = linregress(df_City_Hotel_monthly.index,df_City_Hotel_monthly.to_numpy())
#print("slope: %f, intercept: %f" % (slope, intercept))
#print("R-squared: %f" % r_value**2)

#df_Hotel_monthly = df_City_Hotel_monthly.merge(df_Resort_Hotel_monthly, on='Mes')
#df_Hotel_monthly.columns = ['Precio/noche (City Hotel)', 'Precio/noche (Resort Hotel)']
#print(df_Hotel_monthly)
#plt.figure(figsize=(20,30))
#hotel_monthly.plot.line(linewidth=3.0, color=['green', 'red'], figsize=(16,9))

"""
4. ¿Cuáles son los meses más ocupados?
"""

Meses_ocupados = df.arrival_date_month.value_counts()
Meses_ocupados = Meses_ocupados.reindex(Orden_meses, axis=0)
Meses_ocupados_porcen = np.round(Meses_ocupados / Meses_ocupados.sum() * 100,2)
plt_Meses_ocupados = Meses_ocupados_porcen.plot.bar()
plt_Meses_ocupados.bar_label(plt_Meses_ocupados.containers[0])
plt_Meses_ocupados.legend(title = "Meses mas ocupados vs Porcentaje")

"""
5. ¿Cuánto tiempo se queda la gente en los hoteles (noches)?
"""
df_Estadia = df.stays_in_weekend_nights + df.stays_in_week_nights
df_Estadia = df_Estadia.value_counts()
df_Estadia_porc = df_Estadia/df_Estadia.sum()*100
plt_Estadia = df_Estadia_porc.plot.bar()
plt_Estadia.bar_label(plt_Estadia.containers[0])
plt_Estadia.legend(title = "Noches de estadia vs Porcentaje")

"""
6. Reservas por segmento de mercado
"""
df_Market = df.market_segment.value_counts()
df_Market_porc = np.round(df_Market / df_Market.sum() * 100,2)
plt_Market = df_Market_porc.plot.bar()
plt_Market.bar_label(plt_Market.containers[0])
plt_Market.legend(title = "Reservas X Segmento de Mercado vs Porcentaje")

"""
7. ¿Cuántas reservas se cancelaron?
"""
df_canceled = df.is_canceled.value_counts()
df_canceled_porc = df_canceled / df_canceled.sum() *100
#plt_Canceled = df_canceled_porc.plot.bar()

plt.pie(df_canceled_porc, labels = ['Not Canceled', 'Canceled'],  autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.legend(title = "Reservas:")

"""
8. ¿Qué mes tiene el mayor número de cancelaciones?
"""
canceled_month =  df.groupby(['arrival_date_month'])['is_canceled'].sum()
canceled_month = canceled_month.reindex(Orden_meses, axis=0)
canceled_month_p = np.round(canceled_month / canceled_month.sum() * 100,2)
plt_Canceled_monthly = canceled_month_p.plot.bar()
plt_Canceled_monthly.bar_label(plt_Canceled_monthly.containers[0])
plt_Canceled_monthly.legend(title = "Cancelaciones X Mes vs Porcentaje")

#####################################################################

# Reservation_status-date A timestamp
df['reservation_status_date'] = pd.to_datetime(data['reservation_status_date'])

# separamos el año, mes y dia
df['reservation_status_date_year'] = df['reservation_status_date'].dt.year
df['reservation_status_date_month'] = df['reservation_status_date'].dt.month
df['reservation_status_date_day'] = df['reservation_status_date'].dt.day

# eliminamos la columna original
df.drop('reservation_status_date', axis=1, inplace=True)

# columnas con variables categoricas
categorical_cols = list(df.columns[df.dtypes == 'O'])
print(categorical_cols)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import keras
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras import backend as K

df2 = df.copy()
lbl_enc = LabelEncoder()
hot_enc = OneHotEncoder(drop='first', sparse=False)

row_size = df2.shape[0]

# codificacion de 'hotel' (one-hot)
col_name = 'hotel'
col_1 = np.reshape(np.asarray(df2[col_name]), (row_size, 1))
col_2 = np.reshape(hot_enc.fit_transform(col_1), (row_size))
col_df = pd.DataFrame(col_2, columns = [col_name+'_type'])
df2 = pd.concat([df2, col_df], axis=1)
df2.drop(col_name, axis=1, inplace=True)

# codificación de 'arrival_date_month' (label)
col_name = 'arrival_date_month'
df2[col_name] = df2[col_name].map({'January': 0, 'February': 1,
                                       'March': 2, 'April': 3, 'May': 4, 
                                       'June': 5, 'July': 6, 'August': 7, 
                                       'September': 8, 'October': 9, 
                                       'November': 10, 'December': 11})

# codificacion de 'meal' (label)
col_name = 'meal'
col_1 = np.reshape(np.asarray(df2[col_name]), (row_size, 1))
col_2 = np.reshape(lbl_enc.fit_transform(col_1), (row_size))
df2[col_name] = col_2

# codificacion de 'country' (label)
col_name = 'country'
col_1 = np.reshape(np.asarray(df2[col_name]), (row_size, 1))
col_2 = np.reshape(lbl_enc.fit_transform(col_1), (row_size))
df2[col_name] = col_2

# codificacion de 'market_segment' (label)
col_name = 'market_segment'
col_1 = np.reshape(np.asarray(df2[col_name]), (row_size, 1))
col_2 = np.reshape(lbl_enc.fit_transform(col_1), (row_size))
df2[col_name] = col_2

# codificacion de 'distribution_channel' (label)
col_name = 'distribution_channel'
col_1 = np.reshape(np.asarray(df2[col_name]), (row_size, 1))
col_2 = np.reshape(lbl_enc.fit_transform(col_1), (row_size))
df2[col_name] = col_2

# codificacion de 'reserved_room_type' (label)
col_name = 'reserved_room_type'
col_1 = np.reshape(np.asarray(df2[col_name]), (row_size, 1))
col_2 = np.reshape(lbl_enc.fit_transform(col_1), (row_size))
df2[col_name] = col_2


# codificacion de 'assigned_room_type' (label)
col_name = 'assigned_room_type'
col_1 = np.reshape(np.asarray(df2[col_name]), (row_size, 1))
col_2 = np.reshape(lbl_enc.fit_transform(col_1), (row_size))
df2[col_name] = col_2

# codificacion de 'deposit_type' (label)
col_name = 'deposit_type'
col_1 = np.reshape(np.asarray(df2[col_name]), (row_size, 1))
col_2 = np.reshape(lbl_enc.fit_transform(col_1), (row_size))
df2[col_name] = col_2

# codificacion de 'customer_type' (label)
col_name = 'customer_type'
col_1 = np.reshape(np.asarray(df2[col_name]), (row_size, 1))
col_2 = np.reshape(lbl_enc.fit_transform(col_1), (row_size))
df2[col_name] = col_2

# codificacion de 'arrival_date_year' (label)
col_name = 'arrival_date_year'
df2[col_name] = df2[col_name].map({2015: 0, 2016: 1, 2017: 2})

# codificacion de 'reservation_status_date_year' (label)
col_name = 'reservation_status_date_year'
df2[col_name] = df2[col_name].map({2015: 0, 2016: 1, 2017: 2})

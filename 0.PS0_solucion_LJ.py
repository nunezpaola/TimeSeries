# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 16:40:54 2025

@author: Ignacio
"""

# Series de Tiempo - Práctica 0

# Ejercicio 6 (Extensión) ---

# Librerías →

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg

# ---

# Importamos el Dataset →

file_path = r"World GDP - Precios Constantes (2015).xlsx"
df = pd.read_excel(file_path) # Importamos el Excel y lo limpiamos
df = df.drop([0, 1, 2], axis=0) # Dropeamos las filas que no tienen importancia

df.columns = df.iloc[0] 
df = df.drop(3, axis=0)

pd.set_option('display.float_format', lambda x: '%.6f' % x)  # Establecer formato sin notación científica

df = df.drop(columns=["Country Code", "Indicator Name", "Indicator Code"])

df_USA = df.iloc[251, :]

df_USA = pd.DataFrame(df_USA)
df_USA = df_USA.reset_index()
df_USA = df_USA.drop(0, axis=0)
df_USA.columns = ['Year', 'GDP']

df_USA = df_USA.dropna(subset=['GDP'])
df_USA['GDP'] = pd.to_numeric(df_USA['GDP'], errors='coerce')

# ---

# Ejercicio 1 - Especificación del modelo (Tendencia Lineal) →

df_USA['log_GDP'] = np.log(df_USA['GDP']) # Tomar el logaritmo del PIB (log(PIB))

# Tenemos la ecuación del modelo y = c + at + zt + vt
# - c + at → Tendencia Lineal
# - zt → Parte estacionaria
# - vt → Residuos

# 1 - Crear la variable de tendencia lineal: La tendencia es una serie de valores calculados como
# c + at donde t es el número de períodos (en este caso, podemos usar los años) 
# Para estimar la tendencia, vamos a ajustar una regresión lineal sobre log_GDP y la variable de tiempo 
# t.

# 2 - Ajustar el modelo de regresión: Usamos statsmodels para ajustar un modelo de regresión lineal. 
# El modelo será y = c + at donde y es el log(PBI) y t es la variable del tiempo

# 3 - Calcular los residuos: Los residuos et se calculan restando la tendencia estimada de los valores 
# de log_GDP. Notemos que aca et = zt + vt porque a nivel teorico todo lo que no es tendencia en este 
# modelo es residuo

df_USA['t'] = np.arange(1, len(df_USA) + 1) # Crear la variable de tiempo t
 
# Ajustar el modelo de regresión para estimar la tendencia (c + αt)
# Estamos estimando un modelo donde c es la constante ("Beta 0") y t es el regresor, con a (Beta 1)

X = sm.add_constant(df_USA['t'])  # Agregar constante (c) para el modelo
y = df_USA['log_GDP']

model = sm.OLS(y, X).fit() # Estimación del modelo de regresión
print(model.summary())

# Vemos que Beta 1 = -0,0102, lo que nos dice que por cada unidad de tiempo que incrementamos
# la tendencia en el logaritmo del PBI disminuye en 0,0102 unidades
# De todas maneras es no significativo, por lo que no hay suficiente evidencia estadística para 
# determinar que esta relación se cumpla o sea relevante

df_USA['trend'] = model.predict(X) # Predicciones de la tendencia (c + αt) (ajuste a los valores del dataset)

# Graficar los resultados
plt.figure(figsize=(10,6))

# Logaritmo del PIB y la tendencia ajustada
plt.plot(df_USA['t'], df_USA['log_GDP'], label='Log(PIB)', color='blue')
plt.plot(df_USA['t'], df_USA['trend'], label='Tendencia (c + αt)', color='red', linestyle='--')

plt.title('Logaritmo del PIB y Tendencia Lineal')
plt.xlabel('Período (t)')
plt.ylabel('Log(PIB)')
plt.legend()
plt.ylim(0, 40)
plt.show()

# ---

# 2 - Ejercicio 2 (Residuos) ---

# Le quitamos la tendencia al modelo (y = c + at + et - (c + at) → y = et)

df_USA['residuals'] = df_USA['log_GDP'] - df_USA['trend'] # Calculamos los residuos (et)

# Analizamos la signficiancia estadistica de los residuos
# Usamos el test de Dickey-Fuller que propone que la serie es no estacionaria
# (tiene una raíz unitaria)

result = adfuller(df_USA['residuals'].dropna()) 
# Imprimimos el resultado
print('Estadístico ADF:', result[0])
print('Valor p:', result[1])
print('Valor crítico (1%):', result[4]['1%'])

# En este caso con un p-value de 0,3662 no rechazamos la hipotesis nula de no estacionariedad
# con lo que la no podemos afirmar que la serie sea estacionaria

# Graficar los residuos
plt.figure(figsize=(10,6))
plt.plot(df_USA['t'], df_USA['residuals'], label='Residuos (zt)', color='green')
plt.title('Residuos de la Tendencia Lineal')
plt.xlabel('Período (t)')
plt.ylabel('Residuos (zt)')
plt.legend()
plt.ylim(-5, 5)
plt.show()

# ---

# 3 - Queremos estimar el proceso (AR, MA, ARMA) →

residuals = df_USA['residuals']

# Autocorrelación
plt.subplot(1, 2, 1)
plot_acf(residuals, lags=20, ax=plt.gca(), title="Correlograma de Autocorrelación", color='blue')

# Autocorrelación parcial
plt.subplot(1, 2, 2)
plot_pacf(residuals, lags=20, ax=plt.gca(), title="Correlograma de Autocorrelación Parcial", color='red')

# Observamos que la ACF decae geométricamente a 0 pero no se anula 
# Mientras que la PACF parece anularse entre el lag 15-20 aunque los pimeros lags son 
# los más altos (con lo cual probablemente estemos hablando de un modelo con pocos lags)
# De esta manera, parecería seguir un AR

# Empezamos probando con un AR(1)

model_ar1 = sm.tsa.AutoReg(df_USA['log_GDP'], lags=1)
result = model_ar1.fit()
residuals_ar1 = result.resid # Obtener los residuos (errores de predicción)

# Graficar ACF y PACF de los residuos
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
plot_acf(residuals_ar1, lags=20, ax=ax[0], title='Autocorrelación de los residuos') # ACF de los residuos
plot_pacf(residuals_ar1, lags=20, ax=ax[1], title='Correlación Parcial de los residuos') # PACF de los residuos
plt.tight_layout()
plt.show()

# Estimación del modelo AR(2)
model_ar2 = sm.tsa.AutoReg(df_USA['log_GDP'], lags=2)  # Usamos los residuos para el modelo AR
result = model_ar2.fit()
residuals_ar2 = result.resid # Obtener los residuos (errores de predicción)

# Graficar ACF y PACF de los residuos
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
plot_acf(residuals_ar2, lags=20, ax=ax[0], title='Autocorrelación de los residuos') # ACF de los residuos
plot_pacf(residuals_ar2, lags=20, ax=ax[1], title='Correlación Parcial de los residuos') # PACF de los residuos
plt.tight_layout()
plt.show()

# Estimemos un AR(3) pero que no incluya al lag 2 

# Creamos un DataFrame con la serie de tiempo de los residuos
df_lags = pd.DataFrame()
df_lags['residuals'] = df_USA['residuals']
df_lags['lag1'] = df_USA['residuals'].shift(1)  # Lag 1
df_lags['lag3'] = df_USA['residuals'].shift(3)  # Lag 3
df_lags = df_lags.dropna(subset=['lag1', 'lag3']) # removemos los NaNs si es que hay

# Estimar el modelo AR con lags 1 y 3
X = df_lags[['lag1', 'lag3']]  # Exogeno (lag 1 y lag 3)
y = df_lags['residuals']  # Dependiente

model_ar1_ar3 = AutoReg(y, lags=1, exog=X)  # Modelo AR (sin lag 2, solo los lags 1 y 3)
result = model_ar1_ar3.fit()

residuals_ar1_ar3 = result.resid # Obtener los residuos

# Graficar los ACF y PACF de los residuos del modelo
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
plot_acf(residuals_ar1_ar3, lags=20, ax=ax[0], title='Autocorrelación de los residuos')
plot_pacf(residuals_ar1_ar3, lags=20, ax=ax[1], title='Correlación Parcial de los residuos')
plt.tight_layout()
plt.show()

# Estimamos un modelo AR(3) 
model_ar3 = sm.tsa.AutoReg(df_USA['log_GDP'], lags=2)  # Usamos los residuos para el modelo AR
result = model_ar3.fit()
residuals_ar3 = result.resid # Obtener los residuos (errores de predicción)

# Graficar ACF y PACF de los residuos
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
plot_acf(residuals_ar3, lags=20, ax=ax[0], title='Autocorrelación de los residuos') # ACF de los residuos
plot_pacf(residuals_ar3, lags=20, ax=ax[1], title='Correlación Parcial de los residuos') # PACF de los residuos
plt.tight_layout()
plt.show()

# ---

# 4 - Realizamos un forecast dinámico a partir del 2008 en el modelo sin tendencia → 

df_pred = pd.DataFrame()
df_pred["Year"] = df_USA["Year"]
df_pred["log_GDP"] = df_USA["log_GDP"]
df_pred['log_GDP_detrended'] = df_USA['log_GDP'] - df_USA['trend']
df_pred = df_pred.iloc[0:48,:]

# Ajustamos el modelo AR(3) para los residuos de la serie sin tendencia
model_ar = AutoReg(df_pred['log_GDP_detrended'], lags=3)
result_ar = model_ar.fit()

# Predicción dinámica
forecast_steps = 10  # Número de pasos a predecir (2008 hacia adelante)
forecast_dynamic = result_ar.predict(start=len(df_pred), end=len(df_pred) + forecast_steps - 1)

# Agregamos la predicción a la serie original
forecast_years = np.arange(2008, 2008 + forecast_steps)
df_forecast = pd.DataFrame({
    'Year': forecast_years,
    'Forecast': forecast_dynamic
})


# Ahora graficamos la serie verdadera vs la serie forcasteada
# Concatenamos los valores originales con los 10 valores forcasteados y lo comparamos
# con la serie original hasta 2017

# ---

# 5 - Realizamos un forcast dinámico a partir del 2008 para todo el modelo → 

df_pred_original = pd.DataFrame()
df_pred_original["Year"] = df_USA["Year"]
df_pred_original["log_GDP"] = df_USA["log_GDP"]
df_pred_original = df_pred_original.iloc[:48, :]

# Ajustar el modelo AR(3) a la serie original (log_GDP)
model_ar_original = AutoReg(df_pred_original['log_GDP'], lags=3)
result_ar_original = model_ar_original.fit()

# Predicción dinámica para los próximos 10 períodos (2008 en adelante)
forecast_steps = 10  # Número de pasos a predecir (2008 hacia adelante)
forecast_dynamic_original = result_ar_original.predict(start=len(df_pred_original), end=len(df_pred_original) + forecast_steps - 1)

# Agregar la predicción a la serie original
forecast_years = np.arange(2008, 2008 + forecast_steps)
df_forecast_original = pd.DataFrame({
    'Year': forecast_years,
    'Forecast': forecast_dynamic_original
})

# Ahora graficamos la serie verdadera vs la serie forcasteada
# Concatenamos los valores originales con los 10 valores forcasteados y lo comparamos
# con la serie original hasta 2017

























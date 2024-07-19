# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import prophet

# Load the dataset
data = pd.read_csv('train.csv', parse_dates=['date'], index_col='date')

# Data Aggregation
monthly_data = data.resample('M').sum()

# Time Series Decomposition
decomposition = seasonal_decompose(monthly_data['sales'])
decomposition.plot()
plt.show()

# Train-Test Split
train = monthly_data.iloc[:-12]
test = monthly_data.iloc[-12:]

# SARIMA Model
model = SARIMAX(train['sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()
forecast = results.predict(start=len(train), end=len(train) + len(test) - 1, typ='levels')

# Model Evaluation
rmse = sqrt(mean_squared_error(test['sales'], forecast))
print('Test RMSE: %.3f' % rmse)

# Plotting the Results
plt.figure(figsize=(10,6))
plt.plot(train['sales'], label='Train')
plt.plot(test['sales'], label='Test')
plt.plot(forecast, label='Forecast')
plt.legend()
plt.show()

# Prophet Model
prophet_data = monthly_data.reset_index().rename(columns={'date': 'ds', 'sales': 'y'})
prophet_model = prophet.Prophet()
prophet_model.fit(prophet_data)
future = prophet_model.make_future_dataframe(periods=12, freq='M')
forecast = prophet_model.predict(future)

# Plot Prophet Forecast
prophet_model.plot(forecast)
plt.show()

# Model Evaluation for Prophet
forecast = forecast.set_index('ds')
rmse = sqrt(mean_squared_error(test['sales'], forecast['yhat'].iloc[-12:]))
print('Test RMSE (Prophet): %.3f' % rmse)
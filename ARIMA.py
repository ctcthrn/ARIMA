import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

# Загрузка и подготовка данных
train = pd.read_csv('train.csv', parse_dates=['Date'])
test = pd.read_csv('test.csv', parse_dates=['Date'])

train_ts = train.set_index('Date')['number_sold'].resample('D').sum()
test_ts = test.set_index('Date')['number_sold'].resample('D').sum()

# Полный график продаж за 2010-2019
plt.figure(figsize=(16, 8))
plt.plot(train_ts, label='Train (2010-2018)')
plt.plot(test_ts, label='Test (2019)', color = 'gray')
plt.title('Ежедневные продажи за 2010-2019 годы')
plt.xlabel('Дата')
plt.ylabel('Количество продаж')
plt.legend()
plt.grid(True)
plt.show()

# Декомпозиция временного ряда
try:
    decomposition = seasonal_decompose(train_ts, model='additive', period=365)
except:
    temp_period = min(365, len(train_ts)//2)
    decomposition = seasonal_decompose(train_ts, model='additive', period=temp_period)

plt.figure(figsize=(12, 8))
decomposition.plot()
plt.suptitle('Декомпозиция временного ряда')
plt.tight_layout()
plt.show()

# Построение ARIMA модели на детрендированных данных
print("\nОбучение ARIMA модели...")
deseasonalized = train_ts - decomposition.seasonal
model = ARIMA(deseasonalized, order=(2,1,2))
arima_model = model.fit()
print(arima_model.summary())

# Прогнозирование с восстановлением сезонности
forecast_steps = len(test_ts)
forecast = arima_model.forecast(steps=forecast_steps)

# Сезонная компонента (циклически повторяем последний год)
seasonal_pattern = decomposition.seasonal[-365:].values
repeats = int(np.ceil(forecast_steps/365))
seasonal_component = np.tile(seasonal_pattern, repeats)[:forecast_steps]

forecast = pd.Series(forecast.values + seasonal_component, index=test_ts.index)

# Детальный график с прогнозом
min_len = min(len(test_ts), len(forecast))
test_ts_aligned = test_ts.iloc[:min_len]
forecast_aligned = forecast.iloc[:min_len]

plt.figure(figsize=(16, 8))
train_ts[-365:].plot(label='Последний год обучения (2018)')
test_ts_aligned.plot(label='Реальные значения (2019)', color = 'gray')
forecast_aligned.plot(label='Прогноз ARIMA+Сезонность', style='--', color = 'red')
plt.title('Детальный прогноз продаж на 2019 год')
plt.xlabel('Дата')
plt.ylabel('Количество продаж')
plt.legend()
plt.grid(True)
plt.show()

# Расчет метрик качества
metrics = {
    'MAPE': mean_absolute_percentage_error(test_ts_aligned, forecast_aligned) * 100,
    'RMSE': sqrt(mean_squared_error(test_ts_aligned, forecast_aligned)),
    'R2': r2_score(test_ts_aligned, forecast_aligned)
}

print("\nМетрики качества прогноза:")
print(f"MAPE: {metrics['MAPE']:.2f}%")
print(f"RMSE: {metrics['RMSE']:.2f}")
print(f"R2: {metrics['R2']:.2f}")

# Сохранение результатов
forecast.to_csv('arima_seasonal_forecast.csv', header=['number_sold'])

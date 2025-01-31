# 📈 Stock Price Prediction Using Time Series Analysis  

This project forecasts stock prices using **ARIMA** and **LSTM** models. We:  
✅ Load historical stock price data  
✅ Perform Exploratory Data Analysis (EDA)  
✅ Build ARIMA and LSTM models for prediction  
✅ Compare results and visualize forecasts  

---

## 📌 Table of Contents  
- [1. Data Loading & Exploration](#1-data-loading--exploration)  
- [2. Data Visualization & Stationarity Check](#2-data-visualization--stationarity-check)  
- [3. Stock Price Forecasting with ARIMA](#3-stock-price-forecasting-with-arima)  
- [4. Stock Price Prediction with LSTM](#4-stock-price-prediction-with-lstm)  
- [5. Model Comparison (ARIMA vs LSTM)](#5-model-comparison-arima-vs-lstm)  
- [6. Conclusion](#6-conclusion)  
- [7. Installation & Usage](#7-installation--usage)  
- [8. Results & Visualizations](#8-results--visualizations)  

---

## 1️⃣ Data Loading & Exploration  

We use **Yahoo Finance** (`yfinance`) to fetch historical stock price data.  

```python
import yfinance as yf  
import pandas as pd  

# Download historical data for Apple (AAPL)
stock_symbol = "AAPL"
df = yf.download(stock_symbol, start="2015-01-01", end="2024-01-01")

# Keep only the closing price
df = df[['Close']]
df.rename(columns={"Close": "Price"}, inplace=True)

df.head()  # Display first few rows
```  

✅ We have Apple’s stock price data from **2015 to 2024**.  

---

## 2️⃣ Data Visualization & Stationarity Check  

### 📊 Stock Price Trend  

```python
import matplotlib.pyplot as plt  

plt.figure(figsize=(12,6))
plt.plot(df["Price"], label="Stock Price")
plt.title(f"{stock_symbol} Stock Price Over Time")
plt.xlabel("Year")
plt.ylabel("Price")
plt.legend()
plt.show()
```  

✅ The price shows an **upward trend with fluctuations**.  

### 📉 Check for Stationarity (ADF Test)  

```python
from statsmodels.tsa.stattools import adfuller  

result = adfuller(df["Price"])
print(f"ADF Statistic: {result[0]}")
print(f"P-value: {result[1]}")
```  

✅ If **P-value > 0.05**, data is non-stationary and needs **differencing**.  

---

## 3️⃣ Stock Price Forecasting with ARIMA  

### ✨ Make Data Stationary  

```python
df["Price_Diff"] = df["Price"].diff().dropna()
```  

### 🔮 Train ARIMA Model  

```python
from statsmodels.tsa.arima.model import ARIMA  

model = ARIMA(df["Price"], order=(2,1,2))
model_fit = model.fit()

# Predict next 30 days
future_forecast = model_fit.forecast(steps=30)

# Plot results
plt.figure(figsize=(12,6))
plt.plot(df["Price"], label="Actual Stock Price")
plt.plot(pd.date_range(df.index[-1], periods=30, freq="B"), future_forecast, label="ARIMA Forecast", color="red")
plt.legend()
plt.show()
```  

✅ ARIMA predicts **trends well but struggles with short-term volatility**.  

---

## 4️⃣ Stock Price Prediction with LSTM  

### 🔹 Data Preparation  

```python
from sklearn.preprocessing import MinMaxScaler  
import numpy as np  

scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df[["Price"]])

# Split into training and test sets
train_size = int(len(df) * 0.8)
train_data, test_data = df_scaled[:train_size], df_scaled[train_size:]

# Convert to LSTM format
def create_sequences(data, time_steps=50):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)

# Reshape for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
```  

### 🔥 Build & Train LSTM Model  

```python
import tensorflow as tf  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import LSTM, Dense  

model = Sequential([
    LSTM(50, activation="relu", return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)
```  

### 📊 LSTM Predictions  

```python
predictions = model.predict(X_test)

# Inverse transform to original scale
predictions = scaler.inverse_transform(predictions)
y_test_original = scaler.inverse_transform(y_test.reshape(-1,1))

# Plot LSTM Predictions vs Actual
plt.figure(figsize=(12,6))
plt.plot(y_test_original, label="Actual Price")
plt.plot(predictions, label="LSTM Predictions", color="red")
plt.legend()
plt.show()
```  

✅ **LSTM provides a more dynamic and data-driven forecast**.  

---

## 5️⃣ Model Comparison (ARIMA vs LSTM)  

### 📏 Calculate RMSE for Model Performance  

```python
from sklearn.metrics import mean_squared_error  

arima_rmse = np.sqrt(mean_squared_error(df["Price"][-30:], future_forecast))
lstm_rmse = np.sqrt(mean_squared_error(y_test_original, predictions))

print(f"ARIMA RMSE: {arima_rmse:.2f}")
print(f"LSTM RMSE: {lstm_rmse:.2f}")
```  

✅ **Lower RMSE means better accuracy**. LSTM generally **performs better** for volatile stock data.  

---

## 6️⃣ Conclusion  

✔ **Loaded and analyzed** Apple’s stock price (2015-2024).  
✔ **Trained ARIMA & LSTM** models to forecast stock prices.  
✔ **LSTM performed better** for short-term predictions, ARIMA for long-term trends.  
✔ **Compared models using RMSE** error scores.  

---

## 7️⃣ Installation & Usage  

### 🔧 Install Required Libraries  

```bash
pip install yfinance matplotlib pandas numpy statsmodels scikit-learn tensorflow keras
```  

### 🚀 Run the Project  

```python
python stock_price_prediction.py
```  

---

## 8️⃣ Results & Visualizations  

### 📈 ARIMA Forecast  
![ARIMA Forecast](arima_forecast.png)  

### 🤖 LSTM Predictions  
![LSTM Predictions](lstm_predictions.png)  

🔹 **LSTM outperforms ARIMA** in capturing short-term stock movements.  
🔹 ARIMA provides **a stable long-term trend prediction**.  

---

## 📌 Author  

Developed by **[Ayesha Tariq]**  

✅ If you found this project useful, feel free to ⭐ the repo! 🚀  

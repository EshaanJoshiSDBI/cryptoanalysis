import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
popular_crypto_list = {'Bitcoin':'BTC','Ethereum':'ETH','Ethereum Classic':'ETC','Doge coin':'DOGE','Tether':'USDT','XRP':'Ripple','Bitcoin Cash':'BCH','Cardano':'ADA'}
print('example: ',popular_crypto_list)
crypto_currency = input('Enter the ticker symbol ')
print('example: INR, USD, AUD, CAD, EUR')
currency = input('Currency ')
start = dt.datetime(2015,1,1)
end = dt.datetime.now()
df = web.DataReader(f'{crypto_currency}-{currency}','yahoo',start,end)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_df = scaler.fit_transform(df['Close'].values.reshape(-1,1))
days = 90
predict_for = int(input('Enter the number of days to predict the price for '))
x_train,y_train = [],[]
for i in range(days,len(scaled_df) - predict_for):
    x_train.append(scaled_df[i-days:i,0])
    y_train.append(scaled_df[i + predict_for,0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
model = Sequential()
model.add(LSTM(units = 80, return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 80,return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 80))
model.add(Dropout(0.2))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam',loss = 'mean_squared_error')
model.fit(x_train,y_train,epochs = 30, batch_size = 45)
start_test = dt.datetime(2020,1,1)
end_test = dt.datetime.now()
test_df = web.DataReader(f'{crypto_currency}-{currency}','yahoo',start_test,end_test)
correct_prices = test_df['Close'].values
data = pd.concat((df['Close'], test_df['Close']), axis = 0)
input = data[len(data) - len(test_df) - days:].values
input = input.reshape(-1,1)
input = scaler.fit_transform(input)
x_test = []
for i in range(days,len(input)):
    x_test.append(input[i - days:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))
predicted_price = model.predict(x_test)
predicted_price = scaler.inverse_transform(predicted_price)
plt.plot(correct_prices, color = 'green', label = 'Real prices')
plt.plot(predicted_price, color = 'red', label = 'Predicted prices')
plt.title(f"{crypto_currency}'s prediction")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc = 'upper right')
plt.show()

dataset = [input[len(input) + 1 - days:len(input) + 1, 0]]
dataset = np.array(dataset)
dataset = np.reshape(dataset, (dataset.shape[0],dataset.shape[1], 1))
prediction = model.predict(dataset)
prediction = scaler.inverse_transform(predicted_price)
print(prediction)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from datetime import date
import datetime
from nsepy import get_history

# a method that converts a lsit to a dataframe with dates as index
def list_to_dt(x,start):                        
    x=pd.DataFrame(x)
    datelist = pd.date_range(start, periods=len(x)).tolist()
    y=pd.DataFrame({'Date':datelist})
    y=y.join(x)
    y.set_index("Date", inplace=True)
    return y
warnings.filterwarnings("ignore")

#sym should contain the NSE symbol of the company. E.g.- for Bodal Chemicals, sym='BODALCHEM'
data = get_history(symbol=sym, start=date.today() - datetime.timedelta(days=1600), end=date.today()- datetime.timedelta(days=100))
forc_data = get_history(symbol=sym, start=date.today() - datetime.timedelta(days=300), end=date.today())
data=data['Close']
forc_data = forc_data['Close']
data=data[len(data)-900:]
forc_data=forc_data[len(forc_data)-100:]
list_=data.values
listp=forc_data.values
print(data)
print(forc_data)
# Recurrent Neural Network

# Part 1 - Data Preprocessing


# Importing the training set
dataset_train = data
training_set = data.iloc[:].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set.reshape(-1, 1))

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(50, 900):
    X_train.append(training_set_scaled[i-50:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN
# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price
dataset_test = forc_data
real_stock_price = forc_data[:].values

# Getting the predicted stock price
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 50:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(50, 150):
    X_test.append(inputs[i-50:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

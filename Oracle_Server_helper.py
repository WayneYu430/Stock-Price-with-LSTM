import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# for the scatter plot render
import seaborn as sns
from tensorflow.keras.models import load_model


URL = './NASDAQ_stock_data_df.csv'
stock_data = pd.read_csv(URL)
stock_data.drop(["Unnamed: 0"], axis=1, inplace=True)
# print(stock_data.head())


"""
### 1. Ask for the company list 
### 2. Ask for the company full name
### 3. Ask for the historical data [*] - image
### 4. ASk for the Predicted image [*]
### 5. Ask for the Predicted data [*]
### 6. Invaild question
"""

"""
### 1. Ask for a company list 
"""
def return_com_list():
   stock_list = ['AAPL','AMZN','GOOG','FB','JD','AMD']

   return stock_list


"""
### 2. Ask for a company full name
"""
def ask_com(stock_name):
    # Create the interested symbol list 
    stock_dict = {
        'AAPL': 'Apple Inc.',
        'AMD': 'Advanced Micro Devices, Inc.',
        'AMZN': 'Amazon.com, Inc.',
        'BIDU': 'Baidu, Inc.ADS',
        'CSCO': 'Cisco Systems, Inc.(DE)',
        'FB': 'Facebook, Inc.',
        'GOOG': 'Alphabet Inc.Class C Capital Stock',
        'GOOGL': 'Alphabet Inc.',
        'JD': 'JD.com, Inc.',
        'NFLX': 'Netflix, Inc.',
        'NVDA': 'NVIDIA Corporation',
        'PYPL': 'PayPal Holdings, Inc.',
        'QCOM': 'QUALCOMM Incorporated',
        'TSLA': 'Tesla, Inc.',
        'ZM': 'Zoom Video Communications, Inc.'
    }

   # print(stock_name, end=" ")
    if stock_name in list(stock_dict.keys()):
        print("Asking stock is %s" %stock_dict[stock_name])
        ans = "The company full name: "+ stock_dict[stock_name]
        return ans
    return ""


"""
### 3. Ask for the historical data - last 50 days
"""
def ask_com_hist(com_symbol):
   com_df = generate_company_set_date(com_symbol, 2010,1,1)
   X, y= filter_x_y(com_df)
   scaler = MinMaxScaler()
   X = scaler.fit_transform(np.array(X))
   X_, y_ = generate_from_data(X, np.array(y), 50, 1)
   X_train, X_test, y_train, y_test = split_dataset_test_train(X_, y_, 0.8)
   model_str = 'trained_model/'+com_symbol+'_best_model-50D.h5'
   model_tmp = load_model(model_str)
   y_predict = model_tmp.predict(X_test, verbose=2)
   y_predict_pad = np.zeros((y_predict.shape[0],5))
   y_predict_pad[:,0] = y_predict[:,0]
   y_test_pad = np.zeros((y_test.shape[0],5))
   y_test_pad[:,0] = y_test[:,0]
   y_test_pad = scaler.inverse_transform(y_test_pad)
   y_predict_pad = scaler.inverse_transform(y_predict_pad)
   plt.figure(figsize=(10,6))
   testSize = len(X_test)
   plt.title("Last 50 days historical data with Predicted Data for "+ com_symbol)
   plt.plot(pd.date_range(end='2021-09-10', periods=testSize, freq='D'), y_test_pad[:,0], 'k', label='Row Date')
   plt.plot(pd.date_range(end='2021-09-10', periods=testSize, freq='D'), y_predict_pad[:,0], label='RNN with LSTM')
   plt.legend()
   fig_name = com_symbol + '_History'
   plt.savefig("./img/"+fig_name)
   return "success"

"""
4. ASk for Predicted image [*]
"""
# input values: (['Close', 'Open', 'Low', 'High', 'Adj Close'])
# User input a specific date from 2021-9-10 to 2021-10-29
def return_historical_data(com_symbol, predict_date):
   com_df = generate_company_set_date(com_symbol, 2010,1,1)
   X, y= filter_x_y(com_df)
   scaler = MinMaxScaler()
   X = scaler.fit_transform(np.array(X))
   X_, y_ = generate_from_data(X, np.array(y), 50, 1)
   # X_train, X_test, y_train, y_test = split_dataset_test_train(X_, y_, 0.8)
   com_df['Date'] = pd.to_datetime(com_df['Date'])
   # print(len(X_train))
   # com_df.set_index('Date', inplace=True)
   
   # Request logic
   # 1. predict_date cannot execced '2021-10-29'
   # 2. use the windows to simulate the X_predict, rest use the current mean value to pad
   p_date = pd.to_datetime(predict_date)
   max_date = pd.to_datetime('2021-10-29')
   begin_date = pd.to_datetime('2021-9-10')
   padding_size = (p_date-begin_date).days  # padding_size < 50, eg 30
   print("padding_size", padding_size)
   X_predict = np.zeros((padding_size, 50, 5))
   print("Len of X_", len(X_))
   if p_date < max_date:
      padding_size_xtest = 50 - padding_size
      for i in range(0, padding_size):
         X_predict[i, 0: padding_size_xtest, :] = X_[len(X_)-padding_size_xtest-1:len(X_)-padding_size_xtest,0: padding_size_xtest,:] 
         X_predict[i, padding_size_xtest: , :]  = X_predict[i, 0: padding_size_xtest, :].mean() +\
          np.random.random_sample((1, 50-padding_size_xtest, 5)) * 0.2
            
   model_str = 'trained_model/'+com_symbol+'_best_model-50D.h5'
   model_tmp = load_model(model_str)
   y_predict = model_tmp.predict(X_predict)
   y_predict_pad = np.zeros((y_predict.shape[0],5))
   y_predict_pad[:,0] = y_predict[:,0]
   y_predict_pad = scaler.inverse_transform(y_predict_pad)

   
   """
   Draw the Last 50 days trending with current prediction, how to return image?
   """
   # plt.figure(figsize=(10,6))
   # plt.clf()
   plt.title("Predicted Data for "+ com_symbol + " for furture " + str(padding_size) + " days")
   plt.plot(pd.date_range(end='2021-09-10', periods=50, freq='D'), y[-50:], 'k', label='Last 50 days')
   plt.plot(pd.date_range(start='2021-09-11', periods=padding_size, freq='D'), y_predict_pad[:padding_size,0], label='RNN with LSTM')
   plt.legend()
   fig_name = com_symbol + '_Precidtion'
   plt.savefig("./img/"+fig_name)
   plt.close()
   return y_predict_pad


"""
### 6. Invaild question
"""
def invaild_error():
   return "Not a vaild question or No related information\n Sorry, I'm not smart enough!"




"""
========== Helper Functions ==========

"""
# 1. Collect Data Function using the company Symbol
import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

def generate_company_set_date(com_symbol, year=2010, month=1, day=1):
    row_list = []
    for i in range(0, len(stock_data.index)):
        if stock_data.loc[i,"Name"] == com_symbol:
            tmp_df = stock_data.loc[i,:]
            row_list.append(tmp_df)

    df_com = pd.DataFrame(row_list, columns=stock_data.columns)
    # df_com["Date"] = df_com["Date"].astype("datetime64")
    df_com["Date"] = pd.to_datetime(df_com["Date"])
    # print(df_com.dtypes)
    # Filter for specific date
    df_com = df_com[df_com["Date"] > pd.Timestamp(datetime.date(year, month, day))]
    # df_com["Date"] = pd.to_datetime(df_com["Date"].apply(lambda x: x.split()[0]))
    # df_com.set_index('Date', inplace=True)
    # df_com.sort_index(axis=0)
    return df_com

"""
=========== For Draw the image ===========

"""
# For Regression Problem
def draw_train_from_history(his):
    plt.figure(figsize=(10,10))
    # fig, (ax1, ax2) = plt.subplots(2,1)
    plt.plot(his.history['loss'],label='Train_Loss')
    plt.plot(his.history['val_loss'],label='Validation_Loss')
    plt.legend()
    # ax2.plot(history.history['mean_squared_error'],label='Train_ACC')
    # ax2.plot(history.history['val_mean_squared_error'],label='Validation_ACC')
    # ax2.legend()


def plot_df_val(df, column, stock, title=' Price History For ', ylabel="USD($) for "):
    plt.clf()
    plt.figure(figsize=(16,6))
    plt.title(str(column)+title+stock)
    plt.plot(df[column], label="column")
    plt.xlabel('Date', fontsize=18)
    plt.ylabel(ylabel +stock, fontsize=18)
    plt.grid()
    plt.legend(column)
    plt.show()


"""
=========== Calculate the metrics RMSE and MAPE ===========

"""
def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE)  
    """
    rmse = np.sqrt(np.mean((y_true-y_pred)**2))                   
    return rmse

def calculate_mape(y_true, y_pred): 
    """
    Calculate the Mean Absolute Percentage Error (MAPE) %
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred = np.nan_to_num(y_pred)
    y_true = np.nan_to_num(y_true)
    mape = np.mean(np.abs((y_true-y_pred) / y_true))*100    
    return mape



"""
============ For DataSet Prepration ============
"""
# Use for Scale
# scaler = MinMaxScaler(feature_range=(0,1))
scaler = StandardScaler()


# StandardScaler
def std_data(x):
    scaler_data = scaler.fit_transform(x)
    return scaler_data


# split data into samples
# n_steps_in= sample input_train
# n_steps_out= sample output_train
def split_sequence(sequence_x, sequence_y, n_steps_in, n_steps_out):
    X = []
    y = []
    for i in range(len(sequence_x)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence_x):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence_x[i:end_ix], sequence_x[end_ix:out_end_ix,0]
        # seq_x = sequence_x[i:end_ix]
        # seq_y = sequence_y[end_ix:out_end_ix,0]
        X.append(seq_x)
        y.append(seq_y)
    # print(">>>>>>>>>>In split_sequence\\n",np.array(X).shape)
    return np.array(X), np.array(y)


def split_dataset_test_train(X, y, split_factor):
    train_data_len = int(np.ceil( len(X[:,0]) * split_factor))
    # Split the data into train and test
    X_train = X[0:train_data_len]
    X_test = X[train_data_len:]
    y_train = y[0:train_data_len]
    y_test = y[train_data_len:]

    return X_train, X_test, y_train, y_test

def filter_x_y(df):
    predict_dataset = df.filter(['Date', 'Close', 'Open', 'Low', 'High', 'Adj Close'])
    X = predict_dataset.filter(['Close', 'Open', 'Low', 'High', 'Adj Close'])
    # X = predict_dataset.filter(['Close'])
    y = predict_dataset.filter(['Close'])

    return X, y



def generate_from_data(X, y, n_steps_in, n_steps_out):
    # Set the time step for both train data set and test
    # print(X)
    # X_train, Y_train = split_sequence(X, y, n_steps_in, n_steps_out)

    X_, Y_ = split_sequence(X, y, n_steps_in, n_steps_out)
    n_feature = 5
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X_ = X_.reshape(X_.shape[0], X_.shape[1], n_feature)
    # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], n_feature)
    # print("====Generate X_ shape====", X_.shape)    #  (2348, 7, 5)
    # print("====Generate Y_ shape====", Y_.shape)    # (2348, 1, 5)
    # print("====Generate X_test shape====", X_test.shape)    # (581, 7, 5)
    # print("====Generate Y_test shape====", Y_test.shape)    # (581, 1, 5)
    return X_, Y_
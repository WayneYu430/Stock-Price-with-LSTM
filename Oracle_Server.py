import imp
import mimetypes
from flask import Flask
from flask import request
from flask import send_file
from flask_cors import * 
import Oracle_Server_helper 
from Oracle_Server_helper import return_historical_data, ask_com, ask_com_hist, return_com_list
from Oracle_Server_helper import invaild_error
app = Flask(__name__)


"""
### 1. Ask for the company list 
### 2. Ask for the company full name
### 3. Ask for the historical data [*] - image?
### 4. ASk for the Predicted image [*]
### 5. Ask for the Predicted data [*]
### 6. Invaild question
"""

@app.route("/comList", methods=['POST', 'GET'])
@cross_origin()
def ask_com_list():
    ans_list = return_com_list()
    return str(ans_list)

@app.route("/comName", methods=['POST', 'GET'])
@cross_origin()
def ask_com_name():
    com_symbol = request.form['com_symbol']
    ans_name = ask_com(com_symbol)
    if ans_name=="" or ans_name==None:
        return error_handle()
    return str(ans_name)


@app.route("/comHist", methods=['POST', 'GET'])
@cross_origin()
def ask_com_hist_backend():
    com_symbol = request.form['com_symbol']
    filename = './img/'+com_symbol + '_History.png'
    print("===== Runing ask_com_hist=====")
    ans = ask_com_hist(com_symbol)
    # if ans == "success":
    return send_file(filename, mimetype='image/png')
    # return error_handle()


@app.route("/PredictImg", methods=['POST', 'GET'])
@cross_origin()
def ask_com_img():
    com_symbol = request.form['com_symbol']
    ask_date = request.form['ask_date']
    filename = './img/'+com_symbol + '_Precidtion.png'
    y_predict_pad = return_historical_data(com_symbol, ask_date)
    if y_predict_pad is not None:
        return send_file(filename, mimetype='image/png')
    else:
        return error_handle()

@app.route("/PredictData", methods=['POST', 'GET'])
@cross_origin()
def ask_Pred_data():
    com_symbol = request.form['com_symbol']
    ask_date = request.form['ask_date']
    res = return_historical_data(com_symbol, ask_date)
    ans = res[:,0].tolist()
    return str(ans)


@app.route("/error")
@cross_origin()
def error_handle():
    ans = invaild_error()
    return ans


@app.route("/question", methods=['POST', 'GET'])
@cross_origin()
def ask_question():
    print(request.form)
    name = request.form['name']
    date = request.form['date']
    print("Get form in server", name, "   ", date)
    predict_array = return_historical_data(name, date, False)
    ans = str(predict_array[-1:,0].item())
    ans = "<p> The predicted price is " + ans + "</p>"

    return ans



@app.route("/image", methods=['POST', 'GET'])
@cross_origin()
def ask_image():
    filename= 'model.png'
    return send_file(filename, mimetype='image/png')

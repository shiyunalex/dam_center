# coding=utf-8
import suanpan
from suanpan.app import app
from suanpan.app.arguments import String,Json
import numpy as np
import json
import datetime
import urllib3



def read_json():
    filename = r'大坝中心数据examineData.json'
    with open(filename,'r',encoding = 'utf-8') as read_json:
        data = json.load(read_json)

    if len(data['values']) == 0:
        return 'No data!'
    else:
        num_old = 0
        num_new = 0
        for i in range(len(data['values'])):
            if data['values'][i]['status2'] == 0:
                num_old += 1
            if data['values'][i]['status2'] == None:
                num_new += 1
            if (num_old + num_new) > 150 and num_old > 3:
                break

        if i < (len(data['values']) - 2):
            del data['values'][i + 1:]

        if data['standardDeviation'] == None:
            return 'Has not been detected!'
        elif num_new < 1:
            return 'No new data for detecting!'
        elif (num_old + num_new) < 45:
            return 'No enough detected historic data!'
        else:
            return data

@app.input(String(key="inputData1"))
@app.output(Json(key="outputData1"))
def HelloWorld(context):
    args = context.args
    if args.inputData1:
        return read_json()



if __name__ == "__main__":
    suanpan.run(app)

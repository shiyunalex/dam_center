# coding=utf-8
import suanpan
from suanpan.app import app
from suanpan.app.arguments import String,Json,Csv,Table
import json

def data_read_url(point_info,inputdata):
    url = 'http://api.dam.com.cn:80/api/openAI/damExamine/codeId/' + \
          str(point_info[0]) + '/vectorId/' + str(point_info[1]) + \
          '/examineData?startTime=' + str(point_info[3]) + '%20' + str('00:00:00') + '&endTime=' + \
          str(point_info[4]) + '%20' + str('23:59:59') + '&valueIsAuto=' + str(point_info[2])

    #http = urllib3.PoolManager()
    #r = http.request('GET', url)

    if False:#(r.status) != 200:
        return 'Date reading error!', 0
    else:
        data = inputdata#json.loads(r.data.decode('utf-8'))
        if data['values'] == None:
            return 'No data!', 0
        elif (len(data['values'])) < 35:
            import pdb; pdb.set_trace()
            return 'No enough data!', 0
        else:
            wc_url = 'http://api.dam.com.cn:80/api/openAI/damExamine/codeId/' + str(point_info[0]) + \
                     '/vectorId/' + str(point_info[1]) + '/monitorError?watchTime=' + str(point_info[4]) + '%20' + str(
                '00:00:00')
            #r = http.request('GET', wc_url)
            if True:#(r.status) != 200:
                zwc = 1.0
            else:
                if r.data != b'':
                    if (json.loads(r.data.decode('utf-8')))['value'] != None:
                        zwc = (json.loads(r.data.decode('utf-8')))['value']
                    else:
                        zwc = 1.0
                else:
                    zwc = 1.0
            return data, zwc

@app.input(Json(key="inputData1"))
@app.output(Json(key="outputData1"))
def HelloWorld(context):
    point_info = '144719718,144700108,1,2015-07-13,2020-07-13,sunfuting002'
    point_info = point_info.split(',')
    args = context.args
    if args.inputData1:
        return data_read_url(point_info,args.inputData1)


if __name__ == "__main__":
    suanpan.run(app)

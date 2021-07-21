# coding=utf-8
import suanpan
from suanpan.app import app
from suanpan.app.arguments import String,Json
import datetime
import numpy as np
import urllib3
from error_detector import detector
import json

def zt_xz(zt0, zshuju, bzc, zvmean, zwc, wcbl=0.0):
    z_zt = zt0.copy()
    zvalue = []
    ktime = []
    #vmean = 0
    ztime0 = 0
    for j in range(len(z_zt)):
        if z_zt[j] == 0:
            t0 = datetime.datetime.strptime(zshuju['values'][j]['watchTime'], '%Y-%m-%d %H:%M:%S')
            if abs(datetime.datetime.timestamp(t0) / 86400 - ztime0) > 2.499:
                zvalue.append(zshuju['values'][j]['value'])
                ktime.append(datetime.datetime.timestamp(t0) / 86400)
                ztime0 = datetime.datetime.timestamp(t0) / 86400

    if len(zvalue) > 3:
        zmean = np.mean(np.array(zvalue))
    else:
        zmean = 0


    zv0 = [10.0, 30.0]
    zs0 = [0.5, 1.5, 5.0, 15.0]
    for j in range(len(z_zt)):
        if abs(zshuju['values'][j]['value'] - zmean) > 7.5 * bzc:
            z_zt[j] = 1
        if z_zt[j] != 0:
            zz0 = zshuju['values'][j]['value']
            t0 = datetime.datetime.strptime(zshuju['values'][j]['watchTime'], '%Y-%m-%d %H:%M:%S')
            zzt0 = datetime.datetime.timestamp(t0) / 86400
            zzl = 1e6
            zzr = 1e6
            zztl = 0
            zztr = 0
            for k in range(j - 1, -1, -1):
                if z_zt[k] == 0:
                    zzl = zshuju['values'][k]['value']
                    if (j - 1) >= 0:
                        t0 = datetime.datetime.strptime(zshuju['values'][j - 1]['watchTime'], '%Y-%m-%d %H:%M:%S')
                        zztl = datetime.datetime.timestamp(t0) / 86400
                    break
            for k in range(j + 1, len(z_zt)):
                if z_zt[k] == 0:
                    zzr = zshuju['values'][k]['value']
                    if (j + 1) <= (len(z_zt) - 1):
                        t0 = datetime.datetime.strptime(zshuju['values'][j + 1]['watchTime'], '%Y-%m-%d %H:%M:%S')
                        zztr = datetime.datetime.timestamp(t0) / 86400
                    break
            if zztl == 0:
                zvpl = 1e6
            elif abs(zzt0 - zztl) < 1e-6:
                zvpl = 1e6
            else:
                zvpl = (zz0 - zzl) / (zzt0 - zztl)
            if zztr == 0:
                avpr = 1e6
            elif abs(zzt0 - zztr) < 1e-6:
                zvpr = 1e6
            else:
                zvpr = (zz0 - zzr) / (zzt0 - zztr)

            if (zz0 - zzl) * (zz0 - zzr) <= 0:
                z_zt[j] = 0
            if (abs(zz0 - zzl)) <= wcbl * zwc or (abs(zz0 - zzr)) <= wcbl * zwc:
                z_zt[j] = 0
            elif (abs(zz0 - zzl)) <= zs0[0] * bzc or (abs(zz0 - zzr)) <= zs0[0] * bzc:
                z_zt[j] = 0
            elif (abs(zz0 - zzl)) > zs0[0] * bzc and (abs(zz0 - zzr)) > zs0[0] * bzc \
                    and (abs(zz0 - zzl)) <= zs0[1] * bzc or (abs(zz0 - zzr)) <= zs0[1] * bzc:
                z_zt[j] = 11
                if zzl == 1e6:
                    if abs(zvpr) <= zv0[1] * zvmean:
                        z_zt[j] = 0
                elif zzr == 1e6:
                    if abs(zvpl) <= zv0[1] * zvmean:
                        z_zt[j] = 0
                else:
                    if abs(zvpl) <= zv0[1] * zvmean and abs(zvpr) <= zv0[1] * zvmean:
                        z_zt[j] = 0
            elif (abs(zz0 - zzl)) > zs0[1] * bzc and (abs(zz0 - zzr)) > zs0[1] * bzc \
                    and (abs(zz0 - zzl)) <= zs0[2] * bzc or (abs(zz0 - zzr)) <= zs0[2] * bzc:
                z_zt[j] = 12
                if zzl == 1e6:
                    if abs(zvpr) <= zv0[0] * zvmean:
                        z_zt[j] = 0
                    elif abs(zvpr) <= zv0[1] * zvmean:
                        z_zt[j] = 9
                elif zzr == 1e6:
                    if abs(zvpl) <= zv0[0] * zvmean:
                        z_zt[j] = 0
                    elif abs(zvpl) <= zv0[1] * zvmean:
                        z_zt[j] = 9
                else:
                    if abs(zvpl) <= zv0[0] * zvmean and abs(zvpr) <= zv0[0] * zvmean:
                        z_zt[j] = 0
                    elif abs(zvpl) <= zv0[1] * zvmean and abs(zvpr) <= zv0[1] * zvmean:
                        z_zt[j] = 9
            elif (abs(zz0 - zzl)) > zs0[2] * bzc and (abs(zz0 - zzr)) > zs0[2] * bzc \
                    and (abs(zz0 - zzl)) <= zs0[3] * bzc or (abs(zz0 - zzr)) <= zs0[3] * bzc:
                z_zt[j] = 13
                if zzl == 1e6:
                    if abs(zvpr) <= zv0[0] * zvmean:
                        z_zt[j] = 0
                    elif abs(zvpr) <= zv0[1] * zvmean:
                        z_zt[j] = 9
                elif zzr == 1e6:
                    if abs(zvpl) <= zv0[0] * zvmean:
                        z_zt[j] = 0
                    elif abs(zvpl) <= zv0[1] * zvmean:
                        z_zt[j] = 9
                else:
                    if abs(zvpl) <= zv0[0] * zvmean and abs(zvpr) <= zv0[0] * zvmean:
                        z_zt[j] = 0
                    elif abs(zvpl) <= zv0[1] * zvmean and abs(zvpr) <= zv0[1] * zvmean:
                        z_zt[j] = 9
            else:
                z_zt[j] = 14
                if zzl == 1e6:
                    if abs(zvpr) <= zv0[1] * zvmean:
                        z_zt[j] = 9
                elif zzr == 1e6:
                    if abs(zvpl) <= zv0[1] * zvmean:
                        z_zt[j] = 9
                else:
                    if abs(zvpl) <= zv0[1] * zvmean and abs(zvpr) <= zv0[1] * zvmean:
                        z_zt[j] = 9

            if z_zt[j] != 0:
                if (abs(zz0 - zzl)) < wcbl * zwc or (abs(zz0 - zzr)) < wcbl * zwc:
                    z_zt[j] = 15
    return z_zt


def user_check(userId):
    zb = 0
    ss = []
    ss.append('sunfuting002')
    ss.append('wuwei2020001')
    ss.append('dam_manager001')
    for i in range(len(ss)):
        if userId == ss[i]:
            zb = 1
            break

    return zb


def main_func(point_info,zshuju):
    point_info = point_info.split(',')

    Pinfo = [[point_info[0], point_info[1], point_info[2]], [point_info[3]]]

    data_num = np.zeros(5)
    qx = user_check(Pinfo[1][0])

    if qx < 0.5:
        return '您无权限使用！'
    else:
        #zshuju = data_read_url(Pinfo)
        #        zshuju['values'][0]['value']=100
        bq = []
        if type(zshuju) == type('*'):
            infile = open('log_file/' + str(point_info[3]) + '_new.log', 'a+')
            infile.write(str(datetime.datetime.now()).split('.')[0] + ' codeId: ' + str(Pinfo[0][0]) + \
                         '  vectorId: ' + str(Pinfo[0][1]) + ' isAuto: ' + str(Pinfo[0][2]) + '  message: ' + str(
                zshuju) + '\n')
            infile.close()
            return zshuju
        else:
            t0 = datetime.datetime.now()
            t0 = str(t0).split()
            #wc_url = 'http://api.dam.com.cn:80/api/openAI/damExamine/codeId/' + str(Pinfo[0][0]) + \
            #         '/vectorId/' + str(Pinfo[0][1]) + '/monitorError?watchTime=' + str(t0[0]) + '%20' + str('00:00:00')
            #http = urllib3.PoolManager()
            #r = http.request('GET', wc_url)
            if True:#(r.status) != 200:
                zwc = 0
            else:
                if r.data != b'':
                    if (json.loads(r.data.decode('utf-8')))['value'] != None:
                        zwc = (json.loads(r.data.decode('utf-8')))['value']
                    else:
                        zwc = 0
                else:
                    zwc = 0

            ztime = 0
            zvalue = []

            for j in range(len(zshuju['values'])):
                if zshuju['values'][j]['status2'] != None:
                    ndd = j
                    break
            for j in range(len(zshuju['values']) - 1):
                t0 = datetime.datetime.strptime(zshuju['values'][j]['watchTime'], '%Y-%m-%d %H:%M:%S')
                t1 = datetime.datetime.strptime(zshuju['values'][j + 1]['watchTime'], '%Y-%m-%d %H:%M:%S')
                ztime += datetime.datetime.timestamp(t1) - datetime.datetime.timestamp(t0)
                zvalue.append(zshuju['values'][j]['value'])
            ave_time = abs(ztime / (len(zshuju['values']) - 1)) / 86400

            zvalue.sort()
            zmean = np.mean(np.array(zvalue))

            bq.append(0)
            for j in range(1, len(zshuju['values']) - 1):
                t0 = datetime.datetime.strptime(zshuju['values'][j]['watchTime'], '%Y-%m-%d %H:%M:%S')
                t1 = datetime.datetime.strptime(zshuju['values'][j + 1]['watchTime'], '%Y-%m-%d %H:%M:%S')
                dtime = datetime.datetime.timestamp(t1) - datetime.datetime.timestamp(t0)
                dvalue = abs(zshuju['values'][j + 1]['value'] - zshuju['values'][j]['value'])
                if abs(dtime / 86400) > 30 * ave_time:
                    bq.append(j + 1)
            bq.append(len(zshuju['values']))

            z_zt0, z_zt = [], []
            for j in range(len(bq) - 1):
                data = []
                for k in range(bq[j], bq[j + 1]):
                    data.append([zshuju['values'][k]['watchTime'], zshuju['values'][k]['value']])
                data_zt = detector(data)
                z_zt0.append(data_zt)
            for j in range(len(z_zt0)):
                for k in range(len(z_zt0[j])):
                    z_zt.append(z_zt0[j][k])

            z_zt = zt_xz(z_zt, zshuju, zshuju['standardDeviation'], zshuju['changeRate'], zwc)
            z_zt = zt_xz(z_zt, zshuju, zshuju['standardDeviation'], zshuju['changeRate'], zwc)

            z_zt = zt_xz(z_zt, zshuju, zshuju['standardDeviation'], zshuju['changeRate'], zwc)

            for k in range(len(z_zt)):
                if z_zt[k] == 11:
                    data_num[0] += 1
                    zshuju['values'][k]['status2'] = 1
                    zshuju['values'][k]['abnormalityContent'] = 'AI判定为一级异常'
                elif z_zt[k] == 12:
                    data_num[1] += 1
                    zshuju['values'][k]['status2'] = 2
                    zshuju['values'][k]['abnormalityContent'] = 'AI判定为二级异常'
                elif z_zt[k] == 13:
                    data_num[2] += 1
                    zshuju['values'][k]['status2'] = 3
                    zshuju['values'][k]['abnormalityContent'] = 'AI判定为三级异常'
                elif z_zt[k] == 14:
                    data_num[3] += 1
                    zshuju['values'][k]['status2'] = 4
                    zshuju['values'][k]['abnormalityContent'] = 'AI判定为错误数据'
                elif z_zt[k] == 15:
                    zshuju['values'][k]['status2'] = 5
                    zshuju['values'][k]['abnormalityContent'] = 'AI判定小于误差限'
                elif z_zt[k] == 9:
                    data_num[4] += 1
                    zshuju['values'][k]['status2'] = 9
                    zshuju['values'][k]['abnormalityContent'] = 'AI判定状态待确认'
                else:
                    zshuju['values'][k]['status2'] = 0
            for k in range(len(z_zt)):
                if (z_zt[k] > 9 and z_zt[k] < 14):
                    data_num[4] += 1
                    if z_zt[k] == 11:
                        data_num[0] -= 1
                    if z_zt[k] == 12:
                        data_num[1] -= 1
                    if z_zt[k] == 13:
                        data_num[2] -= 1
                    zshuju['values'][k]['status2'] = 9
                    zshuju['values'][k]['abnormalityContent'] = 'AI判定状态待确认'
                else:
                    break

            infile = open('log_file/' + str(point_info[3]) + '_new.log', 'a+')
            infile.write(str(datetime.datetime.now()).split('.')[0] + ' codeId: ' + str(Pinfo[0][0]) + \
                         '  vectorId: ' + str(Pinfo[0][1]) + ' isAuto: ' + str(Pinfo[0][2]) + '  data_length: ' + str(
                len(zshuju['values'])) + \
                         '    一级异常：' + str(int(data_num[0])) + '  二级异常：' + str(int(data_num[1])) + \
                         '  三级异常：' + str(int(data_num[2])) + '  明显错误：' + str(int(data_num[3])) + \
                         '  无法确定：' + str(int(data_num[4])) + '\n')
            infile.close()

            del zshuju['values'][ndd:]
            zshuju1 = json.dumps(zshuju).encode('utf-8')

            #r = http.request('POST', 'http://api.dam.com.cn:80/api/openAI/monitor/sunUpdate', body=zshuju1, \
            #                 headers={'Content-Type': 'application/json'})

            return zshuju1


@app.input(Json(key="inputData1"))
@app.output(Json(key="outputData1"))
def HelloWorld(context):
    args = context.args
    point_info = '144719712,144700108,1,sunfuting002'
    zshuju1 =  main_func(point_info,args.inputData1)
    return json.loads(zshuju1)





if __name__ == "__main__":
    suanpan.run(app)

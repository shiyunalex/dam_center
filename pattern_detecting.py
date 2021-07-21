# coding=utf-8
import suanpan
from suanpan.app import app
from suanpan.app.arguments import String,Json
import random
import numpy as np
import datetime
import tensorflow as tf
import json
from sklearn.linear_model import LinearRegression
import urllib3

def random_num(n0, n1):
    a = [0, n0 - 1]
    for i in range(1000000):
        zz = random.randint(1, n0 - 2)
        if a.count(zz) == 0:
            a.append(zz)
        if len(a) == n1:
            break
    a.sort()
    return a

def data_chuli(zdata):
    num0 = 500
    num1 = 128

    shuju = []
    for i in range(len(zdata['values'])):
        if zdata['values'][i]['status2'] == None or (
                zdata['values'][i]['status2'] < 1 or zdata['values'][i]['status2'] > 8):
            t0 = datetime.datetime.strptime(zdata['values'][i]['watchTime'], '%Y-%m-%d %H:%M:%S')
            t1 = datetime.datetime.timestamp(t0) / 86400
            shuju.append([t1, zdata['values'][i]['value']])

    shuju.reverse()
    shuju = (np.array(shuju)).reshape(-1, 2)
    nn = shuju.shape[0]

    zt = shuju[nn - 1, 0] - shuju[0, 0]
    zdt = zt / (num1 - 1)
    zx1 = np.arange(shuju[0, 0], shuju[nn - 1, 0] + 1e-6, zdt)
    zy1 = np.interp(zx1, shuju[:, 0], shuju[:, 1])

    shuju1 = np.array([zx1, zy1])
    return np.transpose(shuju1)

def data_normal(x):
    nn = len(x)
    x = (np.array(x)).reshape(-1, 2)
    z1min = min(x[:, 0])
    z1max = max(x[:, 0])

    z2min = min(x[:, 1])
    z2max = max(x[:, 1])

    if z1max - z1min > 1e-6:
        for i in range(nn):
            x[i, 0] = (x[i, 0] - z1min) / (z1max - z1min)
    else:
        x[i, 0] = 0

    if z2max - z2min > 1e-6:
        for i in range(nn):
            x[i, 1] = (x[i, 1] - z2min) / (z2max - z2min)
    else:
        x[i, 1] = 0

    return x

def user_check(userId):
    zb = 0
    ss = []
    ss.append('sunfuting002')
    ss.append('dam_manager001')
    for i in range(len(ss)):
        if userId == ss[i]:
            zb = 1
            break

    return zb

def style_detector(x):
    tf.reset_default_graph()

    x = data_normal(x)

    x = (np.array(x - 0.5)).reshape(1, 1, 128, 2)

    xs = tf.placeholder(tf.float32, [None, 1, 128, 2])
    ys = tf.placeholder(tf.float32, [None, 14])
    keep_prob = tf.placeholder(tf.float32)

    w_conv1 = tf.Variable(tf.truncated_normal([1, 7, 2, 16], stddev=1.0), name='w_conv1')
    b_conv1 = tf.Variable(tf.constant(0.0, shape=[16]), name='b_conv1')
    zL1 = tf.nn.conv2d(xs, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
    zL1 = tf.nn.relu(tf.nn.bias_add(zL1, b_conv1))
    w_conv1_1 = tf.Variable(tf.truncated_normal([1, 5, 16, 16], stddev=0.1), name='w_conv1_1')
    b_conv1_1 = tf.Variable(tf.constant(0.0, shape=[16]), name='b_conv1_1')
    zL1 = tf.nn.conv2d(zL1, w_conv1_1, strides=[1, 1, 1, 1], padding='SAME')
    zL1 = tf.nn.relu(tf.nn.bias_add(zL1, b_conv1_1))
    zL1 = tf.nn.max_pool(zL1, ksize=[1, 1, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_conv2 = tf.Variable(tf.truncated_normal([1, 5, 16, 32], stddev=0.1), name='w_conv2')
    b_conv2 = tf.Variable(tf.constant(0.0, shape=[32]), name='b_conv2')
    zL2 = tf.nn.conv2d(zL1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
    zL2 = tf.nn.relu(tf.nn.bias_add(zL2, b_conv2))
    w_conv2_1 = tf.Variable(tf.truncated_normal([1, 5, 32, 32], stddev=0.1), name='w_conv2_1')
    b_conv2_1 = tf.Variable(tf.constant(0.0, shape=[32]), name='b_conv2_1')
    zL2 = tf.nn.conv2d(zL2, w_conv2_1, strides=[1, 1, 1, 1], padding='SAME')
    zL2 = tf.nn.relu(tf.nn.bias_add(zL2, b_conv2_1))
    zL2 = tf.nn.max_pool(zL2, ksize=[1, 1, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_conv3 = tf.Variable(tf.truncated_normal([1, 5, 32, 64], stddev=0.1), name='w_conv3')
    b_conv3 = tf.Variable(tf.constant(0.0, shape=[64]), name='b_conv3')
    zL3 = tf.nn.conv2d(zL2, w_conv3, strides=[1, 1, 1, 1], padding='SAME')
    zL3 = tf.nn.relu(tf.nn.bias_add(zL3, b_conv3))

    zL3 = tf.nn.max_pool(zL3, ksize=[1, 1, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_conv4 = tf.Variable(tf.truncated_normal([1, 3, 64, 128], stddev=0.1), name='w_conv4')
    b_conv4 = tf.Variable(tf.constant(0.0, shape=[128]), name='b_conv4')
    zL4 = tf.nn.conv2d(zL3, w_conv4, strides=[1, 1, 1, 1], padding='SAME')
    zL4 = tf.nn.relu(tf.nn.bias_add(zL4, b_conv4))
    zL4 = tf.nn.dropout(zL4, keep_prob)
    zL4 = tf.nn.max_pool(zL4, ksize=[1, 1, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_conv5 = tf.Variable(tf.truncated_normal([1, 3, 128, 256], stddev=0.1), name='w_conv5')
    b_conv5 = tf.Variable(tf.constant(0.0, shape=[256]), name='b_conv5')
    zL5 = tf.nn.conv2d(zL4, w_conv5, strides=[1, 1, 1, 1], padding='SAME')
    zL5 = tf.nn.relu(tf.nn.bias_add(zL5, b_conv5))
    zL5 = tf.nn.dropout(zL5, keep_prob)
    zL5 = tf.nn.max_pool(zL5, ksize=[1, 1, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_conv6 = tf.Variable(tf.truncated_normal([1, 4, 256, 1024], stddev=0.1), name='w_conv6')
    zL6 = tf.nn.conv2d(zL5, w_conv6, strides=[1, 1, 1, 1], padding='VALID')
    zL6 = tf.nn.dropout(zL6, keep_prob)

    w_relu = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1), name='w_relu')
    b_relu = tf.Variable(tf.constant(0.0, shape=[1024]), name='b_relu')
    zL_relu = tf.reshape(zL6, [-1, 1024])
    zL_relu = tf.nn.relu(tf.nn.bias_add(tf.matmul(zL_relu, w_relu), b_relu))
    zL_relu = tf.nn.dropout(zL_relu, keep_prob)
    zL_relu = tf.reshape(zL_relu, [-1, 1, 1, 1024])

    w_conv7 = tf.Variable(tf.truncated_normal([1, 1, 1024, 14], stddev=0.1), name='w_conv7')
    zL7 = tf.nn.conv2d(zL_relu, w_conv7, strides=[1, 1, 1, 1], padding='VALID')

    w_soft = tf.Variable(tf.truncated_normal([14, 14], stddev=0.1), name='w_soft')
    b_soft = tf.Variable(tf.constant(0.0, shape=[14]), name='b_soft')
    zL_soft = tf.reshape(zL7, [-1, 14])
    prediction = tf.nn.softmax(tf.nn.bias_add(tf.matmul(zL_soft, w_soft), b_soft))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, 'parameters/Trainmodel0.988_0.977_0.075_0.951-84')
        y_pre = sess.run(prediction, feed_dict=
        {xs: x, keep_prob: 1})
    sess.close()

    zypre = np.argmax(y_pre, axis=1)

    return zypre

def main_func(point_info, inputdata, wcbl=0.0, num0=500, num1=128):
    stype1 = ['无明显规律', '周期型', '正向发散型', '负向发散型', '正向收敛型', '负向收敛型', '正向匀速型', \
              '负向匀速型', '周期+正向发散型', '周期+负向发散型', '周期+正向收敛型', '周期+负向收敛型', \
              '周期+正向匀速型', '周期+负向匀速型']
    stype = ['测值稳定', '周期性变化', '正向收敛变化', '负向收敛变化', '正向持续变化', \
             '负向持续变化', '正向加速变化', '负向加速变化', '其他']
    index_type = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    point_info = point_info.split(',')
    qx = user_check(point_info[5])
    if qx < 0.5:
        return json.dumps('No permission').encode('utf-8')
    else:
        zshuju, wc = inputdata#data_read_url(point_info)

        if type(zshuju) == type('*'):
            infile = open('log_file/' + str(point_info[5]) + '.log', 'a')
            infile.write(str(datetime.datetime.now()).split('.')[0] + '  ' + str(zshuju) + '\n')
            infile.close()
            return json.dumps(zshuju).encode('utf-8')
        else:
            zshuju1 = data_chuli(zshuju)
            ztype = style_detector(zshuju1)
            infile = open('log_file/' + str(point_info[5]) + '.log', 'a')
            infile.write(str(datetime.datetime.now()).split('.')[0] + '  data_length: ' + str(len(zshuju['values'])) + \
                         '    类型：' + str(stype1[int(ztype)]) + '\n')
            infile.close()

            if ztype == 0:
                index_type[8] = 1
            elif ztype == 1:
                index_type[1] = 1
            elif ztype == 2:
                index_type[6] = 1
            elif ztype == 3:
                index_type[7] = 1
            elif ztype == 4:
                index_type[2] = 1
            elif ztype == 5:
                index_type[3] = 1
            elif ztype == 6:
                index_type[4] = 1
            elif ztype == 7:
                index_type[5] = 1
            elif ztype == 8:
                index_type[1] = 1
                index_type[6] = 1
            elif ztype == 9:
                index_type[1] = 1
                index_type[7] = 1
            elif ztype == 10:
                index_type[1] = 1
                index_type[2] = 1
            elif ztype == 11:
                index_type[1] = 1
                index_type[3] = 1
            elif ztype == 12:
                index_type[1] = 1
                index_type[4] = 1
            elif ztype == 13:
                index_type[1] = 1
                index_type[5] = 1

            dd = max(zshuju1[:, 1]) - min(zshuju1[:, 1])
            reg = LinearRegression()

            reg.fit(zshuju1[:, 0].reshape(-1, 1), zshuju1[:, 1].reshape(-1, 1))
            ddv = (reg.predict([[365]]) - reg.predict([[0]]))

            if dd < wcbl * wc * 2:
                index_type[1:9] = [0, 0, 0, 0, 0, 0, 0, 0]
                index_type[0] = 1
            if abs(float(ddv)) < wcbl * wc:
                index_type[2:8] = [0, 0, 0, 0, 0, 0]
            if index_type.count(0) == 9:
                index_type[8] = 1

            ls = ' '
            for i in range(len(index_type)):
                if index_type[i] == 1:
                    ls += str(stype[i]) + '  '

            zdata = {}
            zdata['codeId'] = point_info[0]
            zdata['vectorId'] = point_info[1]
            zdata['autoId'] = point_info[2]
            zdata['startTime'] = point_info[3]
            zdata['finalTime'] = point_info[4]
            zdata['Type'] = ls
            zdata['Type_id'] = index_type

            zdata = json.dumps(zdata).encode('utf-8')

            return zdata

@app.input(Json(key="inputData1"))
@app.output(Json(key="outputData1"))
def HelloWorld(context):
    args = context.args
    point_info = '144719718,144700108,1,2015-07-13,2020-07-13,sunfuting002'
    if args.inputData1:
        zypre = main_func(point_info, args.inputData1, wcbl=0.0, num0=500, num1=128)
        return json.loads(zypre)

if __name__ == "__main__":
    suanpan.run(app)

# coding=utf-8
import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest
import urllib3
import json
import random
import math
import datetime
import time
import matplotlib.pyplot as plt


# definition of data normalization function

def normalization(x):
    x1 = []
    n = len(x)
    x = np.array(x)
    pj = np.mean(x)
    bzc = np.std(x)
    if bzc < 1e-6:
        bzc = 1e-6
    for i in range(0, n):
        x1.append((x[i] - pj) / bzc)
    #        print(x[i],pj,bzc)
    return x1


# transform data to polar coordinate
def cartesian_to_polar(x, y):
    y1 = []
    n = len(x)
    for i in range(0, n):
        y1.append([math.cos(x[i] / 365 * 2 * np.pi) * y[i], math.sin(x[i] / 365 * 2 * np.pi) * y[i]])
    return y1


# 中值加权滤波
def lvbo(data):
    data0 = data.copy()
    data1 = []
    n = 7
    bn = int(0.5 * n)
    num = len(data)
    if num < n:
        return data
    else:
        for i in range(0, num):
            idex1 = i - bn
            idex2 = i + bn
            if idex1 < 0:
                idex1 = 0
            if idex2 > num - 1:
                idex2 = num - 1
            xsum = sum(data0[idex1:idex2 + 1])
            yy = (xsum - max(data0[idex1:idex2 + 1]) - min(data0[idex1:idex2 + 1])) / (idex2 - idex1 - 1)

            data0[i] = yy
            data1.append(yy)
        return data1


# 中值滤波
def lvbo1(data):
    data0 = data.copy()
    data1 = []
    n = 7
    bn = int(0.5 * n)
    num = len(data)
    if num < n:
        return data
    else:
        for i in range(0, bn):
            data1.append(data[i])
        for i in range(bn, num - bn):
            aa = data0[i - bn:i + bn + 1]
            aa.sort()
            data1.append(aa[bn])
        for i in range(num - bn, num):
            data1.append(data[i])
        return data1


def itree(x_train, rng=np.random.RandomState(2)):
    if len(x_train) > 256:
        clf = IsolationForest(n_estimators=100, max_samples=256, random_state=rng,
                              #behaviour='new',
                              contamination=0.1)
    else:
        clf = IsolationForest(n_estimators=100, max_samples=1.0, random_state=rng,
                              #behaviour='new',
                              contamination=0.1)
    clf.fit(x_train)
    return clf


# split data to supervised detecting series
def data_series(x):
    n0 = len(x)
    n = int(n0 / 35)
    n1 = n0 - n * 35
    xt_forward = []
    xt_backward = []
    zmin = min(x)
    zmax = max(x)
    zjx = zmax - zmin
    if zjx < 1e-6:
        zjx = 1e-6
    for i in range(0, n):
        xt_forward.append(x[35 * i:35 * (i + 1)])
        xt_backward.append(x[n0 - (i + 1) * 35:n0 - i * 35])
    if n1 > 0:
        xt_forward.append(x[n0 - 35:n0])
        xt_backward.append(x[0:35])
    xt_forward = (np.array(xt_forward).reshape(-1, 1, 35, 1)).astype(np.float32)
    xt_backward = (np.array(xt_backward).reshape(-1, 1, 35, 1)).astype(np.float32)

    if n1 > 0:
        n2 = n + 1
    else:
        n2 = n
    for i in range(0, n2):
        #        zmin_forward=min(xt_forward[i,0,:,0])
        #        zmax_forward=max(xt_forward[i,0,:,0])
        xt_forward[i, 0, :, 0] = (xt_forward[i, 0, :, 0] - zmin) / \
                                 (zjx)

        #        zmin_backward=min(xt_backward[i,0,:,0])
        #        zmax_backward=max(xt_backward[i,0,:,0])
        xt_backward[i, 0, :, 0] = (xt_backward[i, 0, :, 0] - zmin) / \
                                  (zjx)
    return xt_forward, xt_backward


# definition of random series obtain
def one_series(x, zt, k):
    a = []
    b = []
    kk = 0
    kk1 = 0
    wz = 0
    wz1 = 0
    zloc = np.random.randint(0, 35)

    #    for i in range(0,len(x)):
    #        if zt[i]<1:
    #            b.append(x[i])

    #    for i in range(0,len(x)):
    #        if i==int(k):
    #            a.append(x[i])
    #            wz=kk
    #            kk=kk+1
    #            continue
    #        if zt[i]<1 or zt[i]>4:
    #            a.append(x[i])
    #            kk=kk+1

    for i in range(k - 1, -1, -1):
        if zt[i] < 1 or zt[i] > 4:
            a.append(x[i])
            kk = kk + 1
            if kk > 125:
                break
    a.reverse()
    a.append(x[k])
    wz = kk

    for i in range(k + 1, len(x)):
        if zt[i] < 1 or zt[i] > 4:
            a.append(x[i])
            kk1 = kk1 + 1
            if kk1 > 125:
                break

    kk = len(a)
    nd1 = wz
    nd2 = kk - 1 - wz
    zmin = min(a)
    zmax = max(a)
    zjx = zmax - zmin
    if zjx < 1e-6:
        zjx = 1e-6

    a = (np.array(a)).reshape(-1).astype(np.float32)

    for i in range(0, kk):
        a[i] = (a[i] - zmin) / (zjx)

    if nd1 < zloc:
        b = a[0:35]
        wz1 = wz
    elif nd2 < (34 - zloc):
        b = a[kk - 35:kk]
        wz1 = 34 - nd2
    else:
        b = a[wz - zloc:wz + 35 - zloc]
        wz1 = zloc
    #    print(kk,nd1,nd2,zloc,len(b))
    if len(b) < 35:
        zzt = 0
    else:
        b = b.reshape(1, 1, 35, 1)
        zzt = 1

    return b, wz1, zzt


def representation(x, y, ng, zt):
    dd = 1e-8
    a = []
    for i in range(0, ng * ng):
        a.append([0])

    x_min = 1e8
    x_max = -1e8
    y_min = 1e8
    y_max = -1e8
    for i in range(len(x)):
        if zt[i] == 0:
            x_min = min(x_min, x[i])
            x_max = max(x_max, x[i])
            y_min = min(y_min, y[i])
            y_max = max(y_max, y[i])
    x_max += dd
    x_min += -dd
    y_max += dd
    y_min += -dd

    dx = (x_max - x_min) / ng + 1e-8
    dy = (y_max - y_min) / ng + 1e-8

    #    print('************')
    for i in range(0, len(x)):
        if zt[i] == 0:
            nx = int((x[i] - x_min) / dx)
            ny = int((y[i] - y_min) / dy)
            zb = int(nx * ng + ny)
            a[zb].append(i)
    #        else:
    #            print(y[i])

    return a


def shuju_add(zx0, x0, y0):
    zx, x, y = [], [], []
    for i in range(len(x0)):
        zx.append([zx0[i]])
        x.append([x0[i]])
        y.append([y0[i]])

    dtime = 0
    for i in range(len(x0) - 1):
        dtime += (x[i + 1][0] - x[i][0])
    av_dtime = dtime / (len(x0) - 1)

    for i in range(len(x0) - 1):
        if (x[i + 1][0] - x[i][0]) > 250 * av_dtime:
            dn = int((x[i + 1][0] - x[i][0]) / av_dtime) - 1
            print(dtime, av_dtime, dn)
            ddx = (x[i + 1][0] - x[i][0]) / (dn + 1)
            #            ddy=(y[i+1][0]-y[i][0])/(dn+1)
            ddy = 0
            for j in range(dn):
                zx[i].append(zx[i][0])
                x[i].append(x[i][0] + (j + 1) * ddx)
                y[i].append(y[i][0] + (j + 1) * ddy)
    zx1, x1, y1, tfz = [], [], [], []
    for i in range(len(x0)):
        ni = len(x[i])
        for j in range(ni):
            zx1.append(zx[i][j])
            x1.append(x[i][j])
            y1.append(y[i][j])
            if j == 0:
                tfz.append(i)
            else:
                tfz.append(-1)
    return zx1, x1, y1, tfz


def detector(data0):
    tf.reset_default_graph()
    znum = len(data0)
    t0_0 = []
    t1_0 = []
    Ua_0 = []
    for i in range(znum):
        t0 = data0[i][0]
        t1 = datetime.datetime.strptime(t0, '%Y-%m-%d %H:%M:%S')
        t0_0.append(t1)
        t1_0.append(datetime.datetime.timestamp(t1) / 86400)
        Ua_0.append(float(data0[i][1]))

    t0_0.reverse()
    t1_0.reverse()
    Ua_0.reverse()

    #    t0_0,t1_0,Ua_0,tfz=shuju_add(t0_0,t1_0,Ua_0)
    #    znum=len(t1_0)
    zt_0 = np.zeros([znum])

    ng = 500
    #    wz0=representation(t1_0,Ua_0,ng)
    #    t0=[]
    #    t1=[]
    #    Ua=[]
    #    kk=0
    #    for i in range(0,ng*ng):
    #        if len(wz0[i])>1:
    #            t0.append(t0_0[wz0[i][1]])
    #            t1.append(t1_0[wz0[i][1]])
    #            Ua.append(Ua_0[wz0[i][1]])

    #    pnum=len(t0)
    if znum < 3:
        zt_0 += 9
    else:
        #    print(znum,pnum)
        #        zt=np.zeros([pnum])

        # normal-0  suspected-1
        #    print('Data normalization and iForest:')
        # data normalization+iForest->suspected
        for i in range(0, 5):
            kk = 0
            index = []
            index.append(0)
            dtime = 183
            zt0 = t1_0[0] + dtime
            for j in range(len(t1_0) - 1):
                if t1_0[j] <= zt0 and t1_0[j + 1] > zt0:
                    index.append(j)
                    zt0 = zt0 + dtime
            if (t1_0[-1] - t1_0[index[-1]]) > (0.5 * dtime):
                index.append(len(t1_0) - 1)
            else:
                index[-1] = len(t1_0) - 1

            for j in range(len(index) - 1):
                index0 = index[j]
                index1 = index[j + 1]
                x_t = []
                x_u = []
                x_num = []
                for k in range(index0, index1):
                    if zt_0[k] == 0:
                        x_t.append(t1_0[k])
                        x_u.append(Ua_0[k])
                        x_num.append(k)
                x_u = normalization(x_u)
                x_train = cartesian_to_polar(x_t, x_u)
                if len(x_train) == 0:
                    continue
                x_train = np.array(x_train).reshape(-1, 2)

                clf = itree(x_train)
                ss = -clf.score_samples(x_train)
                for k in range(0, len(ss)):
                    if ss[k] > 0.75:
                        if zt_0[int(x_num[k])] == 0:
                            zt_0[int(x_num[k])] = 1
                            kk = kk + 1
            #        print(kk)
            if kk == 0:
                break

    if znum > 37:
        wz0 = representation(t1_0, Ua_0, ng, zt_0)
        t0 = []
        t1 = []
        Ua = []
        kk = 0
        for i in range(0, ng * ng):
            if len(wz0[i]) > 1:
                t0.append(t0_0[wz0[i][1]])
                t1.append(t1_0[wz0[i][1]])
                Ua.append(Ua_0[wz0[i][1]])

        pnum = len(t0)
        zt = np.zeros([pnum])
        #    x_normal1=[]
        #    x_suspected1=[]

        #    for i in range(0,pnum):
        #        if zt[i]==0:
        #            x_normal1.append([t0[i],Ua[i]])
        #        if zt[i]==1:
        #            x_suspected1.append([t0[i],Ua[i]])

        #    x_normal1=np.array(x_normal1).reshape(-1,2)
        #    x_suspected1=np.array(x_suspected1).reshape(-1,2)

        # normal-0  suspected-2
        #    print('Data smoothing and iForest:')
        # smoothing+iForest->suspected
        for i in range(0, 12):
            kk1 = 0
            x_t = []
            x_u = []
            x_u1 = []
            x_num = []
            for j in range(0, pnum):
                if zt[j] == 0:
                    x_t.append(t1[j])
                    x_u.append(Ua[j])
                    x_num.append(j)
            a = lvbo(x_u)
            b = lvbo1(x_u)
            for j in range(len(a)):
                if i < 0:
                    x_u1.append(b[j])
                else:
                    x_u1.append(a[j])
            x_u = np.array(x_u).reshape(-1)
            x_u1 = np.array(x_u1).reshape(-1)
            dx_u = (abs(x_u1 - x_u) + 1e-5)
            x_train = cartesian_to_polar(x_t, dx_u)
            x_train = np.array(x_train).reshape(-1, 2)

            clf = itree(x_train)
            ss = -clf.score_samples(x_train)

            for j in range(0, len(ss)):
                if i < 0:
                    jd = 0.72
                else:
                    jd = 0.75
                if ss[j] > jd:
                    if zt[int(x_num[j])] == 0:
                        zt[int(x_num[j])] = 2
                        kk1 = kk1 + 1

            #        print(kk1)
            if kk1 == 0:
                break

        #    x_normal2=[]
        #    x_suspected2=[]

        #    for i in range(0,pnum):
        #        if zt[i]==0:
        #            x_normal2.append([t0[i],Ua[i]])
        #        if zt[i]==2:
        #            x_suspected2.append([t0[i],Ua[i]])

        #    x_normal2=np.array(x_normal2).reshape(-1,2)
        #    x_suspected2=np.array(x_suspected2).reshape(-1,2)

        # ******************************************************************************
        # CNN network
        tf.reset_default_graph()
        global prediction
        num = 35
        # 参数占坑
        xs = tf.placeholder(tf.float32, [None, 1, num, 1])
        ys = tf.placeholder(tf.float32, [None, num + 1])
        keep_prob = tf.placeholder(tf.float32)

        # CNN NETWORK
        # cnn size35-18
        w_conv1 = tf.Variable(tf.truncated_normal([1, 3, 1, 32], stddev=0.1), name='w_conv1')
        zl1 = tf.nn.conv2d(xs, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        zl1 = tf.nn.max_pool(zl1, ksize=(1, 1, 3, 1), strides=[1, 1, 2, 1], padding='SAME')

        # cnn size18-9
        w_conv2 = tf.Variable(tf.truncated_normal([1, 3, 32, 64], stddev=0.1), name='w_conv2')
        zl2 = tf.nn.conv2d(zl1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
        zl2 = tf.nn.max_pool(zl2, ksize=(1, 1, 3, 1), strides=[1, 1, 2, 1], padding='SAME')

        # cnn size9-5
        w_conv3 = tf.Variable(tf.truncated_normal([1, 3, 64, 128], stddev=0.1), name='w_conv3')
        zl3 = tf.nn.conv2d(zl2, w_conv3, strides=[1, 1, 1, 1], padding='SAME')
        zl3 = tf.nn.max_pool(zl3, ksize=(1, 1, 3, 1), strides=[1, 1, 2, 1], padding='SAME')

        # cnn size5-3
        w_conv4 = tf.Variable(tf.truncated_normal([1, 3, 128, 256], stddev=0.1), name='w_conv4')
        zl4 = tf.nn.conv2d(zl3, w_conv4, strides=[1, 1, 1, 1], padding='SAME')
        zl4 = tf.nn.max_pool(zl4, ksize=(1, 1, 3, 1), strides=[1, 1, 2, 1], padding='SAME')

        # cnn size3-1
        w_conv5 = tf.Variable(tf.truncated_normal([1, 3, 256, 512], stddev=0.1), name='w_conv5')
        zl5 = tf.nn.conv2d(zl4, w_conv5, strides=[1, 1, 1, 1], padding='VALID')
        zl5 = tf.nn.dropout(zl5, keep_prob)

        # relu
        zl5 = tf.reshape(zl5, [-1, 512])
        w_relu = tf.Variable(tf.truncated_normal([512, 512], stddev=0.1), name='w_relu')
        b_relu = tf.Variable(tf.constant(0.1, shape=[512]), name='b_relu')
        zl_relu = tf.nn.tanh(tf.matmul(zl5, w_relu) + b_relu)
        zl_relu = tf.reshape(zl_relu, [-1, 1, 1, 512])
        zl_relu = tf.nn.dropout(zl_relu, keep_prob)

        # cnn
        w_conv6 = tf.Variable(tf.truncated_normal([1, 1, 512, 512], stddev=0.1), name='w_conv6')
        zl6 = tf.nn.conv2d(zl_relu, w_conv6, strides=[1, 1, 1, 1], padding='VALID')

        # soft_max
        w_soft = tf.Variable(tf.truncated_normal([512, 36], stddev=0.1), name='w_soft')
        b_soft = tf.Variable(tf.constant(0.1, shape=[36]), name='b_soft')
        zl6 = tf.reshape(zl6, [-1, 512])
        prediction = tf.nn.softmax(tf.matmul(zl6, w_soft) + b_soft)
        # ***************************************************************************

        saver = tf.train.Saver([w_conv1, w_conv2, w_conv3, w_conv4, w_conv5, w_conv6, w_relu, b_relu, w_soft, b_soft])
        # saver=tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, 'parameter_file/parameter0.988_0.988.ckpt')

        #    saver.restore(sess,'parameter_file/parameter0.987_0.986.ckpt')
        # 定义预测值输出程序
        def pic_prediction(x_test, sess):
            global prediction
            znum = 50
            nnum = int(len(x_test) / znum)
            za = np.zeros([len(x_test), 36])
            for i in range(0, nnum):
                y_pre = sess.run(prediction, feed_dict=
                {xs: x_test[(i * znum):((i + 1) * znum), :, :, :], keep_prob: 1})
                za[(i * znum):((i + 1) * znum)] = y_pre
            y_pre = sess.run(prediction, feed_dict={xs: x_test[(nnum * znum):len(x_test), :, :, :], keep_prob: 1})
            za[(nnum * znum):len(x_test)] = y_pre
            zypre = np.array(za).reshape(len(x_test), 36).astype(np.float32)
            return zypre

        # normal-0  suspected-3
        #    print('Supervised detecting:')
        # supervised detecting->abnormal
        for i in range(0, 12):
            kk2 = 0
            x_t = []
            x_u = []
            x_num = []
            for j in range(0, pnum):
                if zt[j] == 0:
                    x_t.append(t1[j])
                    x_u.append(Ua[j])
                    x_num.append(j)
            if (len(x_u)) < 35:
                break
            x_test1, x_test2 = data_series(x_u)
            y1 = pic_prediction(x_test1, sess)
            y2 = pic_prediction(x_test2, sess)

            n_data = len(x_u)
            n_length = len(y1)
            score = np.zeros([n_data])
            n_res = n_data - 35 * (n_length - 1)

            for j in range(0, n_length - 1):
                score[j * 35:(j + 1) * 35] += y1[j, 0:35]
            if n_res > 0:
                score[n_data - n_res:n_data] += y1[n_length - 1, 35 - n_res:35]
            for j in range(0, n_length - 1):
                score[n_data - (j + 1) * 35:n_data - j * 35] += y2[j, 0:35]
            if n_res > 0:
                score[0:n_res] += y2[n_length - 1, 0:n_res]

            score[:] = score[:] * 0.5

            for j in range(0, n_data):
                if score[j] > 0.5:
                    zt[int(x_num[j])] = 3
                    kk2 = kk2 + 1

            #        print(kk2)
            if kk2 == 0:
                break

        #    x_normal3=[]
        #    x_suspected3=[]
        #    for i in range(0,pnum):
        #        if zt[i]==0:
        #            x_normal3.append([t0[i],Ua[i]])
        #        if zt[i]==3:
        #            x_suspected3.append([t0[i],Ua[i]])

        #    x_normal3=np.array(x_normal3).reshape(-1,2)
        #    x_suspected3=np.array(x_suspected3).reshape(-1,2)

        # normal-0  suspected-4
        #    print('Supervised y-axis detecting')
        kf1 = 0
        kf11 = 0
        kk1 = 0
        kk2 = 0
        for i in range(0, ng):
            kf2 = 0
            for j in range(0, ng):
                if len(wz0[i * ng + j]) > 1:
                    kf2 += 1
            if kf2 > 3:
                kk1 += kf2
                for j in range(0, ng):
                    if len(wz0[i * ng + j]) > 1:
                        if zt[kf1] == 0:
                            zt[kf1] = 4
                        kf1 += 1
                for j in range(0, ng):
                    if len(wz0[i * ng + j]) > 1:
                        if zt[kf11] == 4:
                            x_test, wz, zzt = one_series(Ua, zt, kf11)
                            if zzt == 1:
                                y_test = pic_prediction(x_test, sess)
                                if y_test[0, wz] < 0.5:
                                    zt[kf11] = 0
                                    kk2 += 1
                        kf11 += 1

            else:
                kf1 += kf2
                kf11 += kf2

        #    print(kk1-kk2)

        #    x_normal3_1=[]
        #    x_suspected3_1=[]
        #    for i in range(0,pnum):
        #        if zt[i]==0:
        #            x_normal3_1.append([t0[i],Ua[i]])
        #        if zt[i]==4:
        #            x_suspected3_1.append([t0[i],Ua[i]])

        #    x_normal3_1=np.array(x_normal3_1).reshape(-1,2)
        #    x_suspected3_1=np.array(x_suspected3_1).reshape(-1,2)

        #    print('Supervised dtermination:')
        # supervised detecting->suspected determination(recall)
        kk3 = 0
        for i in range(0, pnum):
            if zt[i] > 0 and zt[i] < 5:
                for j in range(100):
                    x_test1, wz1, zzt1 = one_series(Ua, zt, i)
                    if wz1 > 7 and wz1 < 27:
                        break
                for j in range(100):
                    x_test2, wz2, zzt2 = one_series(Ua, zt, i)
                    if wz2 > 7 and wz2 < 27:
                        break
                if zzt1 == 0 or zzt2 == 0:
                    continue
                else:
                    y_test1 = pic_prediction(x_test1, sess)
                    y_test2 = pic_prediction(x_test2, sess)
                    if (y_test1[0, wz1] + y_test2[0, wz2]) < 0.5:
                        zt[i] = 5
                        kk3 += 1
        #    print(kk3)

        #    x_normal4=[]
        #    x_suspected4=[]
        #    x_suspected5=[]
        #    for i in range(0,pnum):
        #        if zt[i]==0:
        #            x_normal4.append([t0[i],Ua[i]])
        #        elif zt[i]==5:
        #            x_suspected4.append([t0[i],Ua[i]])
        #        else:
        #            x_suspected5.append([t0[i],Ua[i]])

        #    x_normal4=np.array(x_normal4).reshape(-1,2)
        #    x_suspected4=np.array(x_suspected4).reshape(-1,2)
        #    x_suspected5=np.array(x_suspected5).reshape(-1,2)

        sess.close()

        # representation back
        kk = 0
        for i in range(0, ng * ng):
            if len(wz0[i]) > 1:
                for j in range(1, len(wz0[i])):
                    zt_0[wz0[i][j]] = zt[kk]
                kk += 1

    #    zls=[]
    #    for i in range(0,znum):
    #        if zt_0[i]==0 or zt_0[i]==5:
    #            zls.append(Ua_0[i])
    #    bzc=np.std(np.array(zls))

    #    for i in range(0,znum):
    #        if zt_0[i]>0 and zt_0[i]<5:
    #            zz0=Ua_0[i]
    #            zzl=1e6
    #            zzr=1e6
    #            for j in range(i,0,-1):
    #                if zt_0[j]==0 or zt_0[j]==5:
    #                    zzl=Ua_0[j]
    #                    break
    #            for j in range(i,znum):
    #                if zt_0[j]==0 or zt_0[j]==5:
    #                    zzr=Ua_0[j]
    #                    break
    #            dzz=0.5*abs((zz0-zzl)+(zz0-zzr))
    #            print(bzc,zz0,zzl,zzr)

    #            if (abs(zz0-zzl))<=0.5*bzc or (abs(zz0-zzr))<=0.5*bzc:
    #                zt_0[i]=5
    #            elif (abs(zz0-zzl))>0.5*bzc and (abs(zz0-zzr))>0.5*bzc \
    #                 and (abs(zz0-zzl))<=1.5*bzc or (abs(zz0-zzr))<=1.5*bzc:
    #                zt_0[i]=11
    #            elif (abs(zz0-zzl))>1.5*bzc and (abs(zz0-zzr))>1.5*bzc \
    #                 and (abs(zz0-zzl))<=5.0*bzc or (abs(zz0-zzr))<=5.0*bzc:
    #                zt_0[i]=12
    #            elif (abs(zz0-zzl))>3.0*bzc and (abs(zz0-zzr))>3.0*bzc:
    #            else:
    #                zt_0[i]=13

    #                print('*')
    #    print()
    #    print('&&&&&&&&&&&&&&&&&&&&&&&')

    #    print(znum,kk1)
    #    x_normal=[]
    #    for i in range(0,znum):
    #        if zt_0[i]<1 or zt_0[i]>4:
    #            x_normal.append([t0_0[i],Ua_0[i]])
    #    x_normal=np.array(x_normal).reshape(-1,2)

    data_labeled = []
    #    x_normal,x_abnormal1,x_abnormal2,x_abnormal3=[],[],[],[]
    for i in range(0, znum):
        #        if zt_0[i]==0 or zt_0[i]==5:
        #            data_labeled.append(0)
        #            x_normal.append([t0_0[i],Ua_0[i]])
        #        elif zt_0[i]==11:
        #            data_labeled.append(1)
        #            x_abnormal1.append([t0_0[i],Ua_0[i]])
        #        elif zt_0[i]==12:
        #            data_labeled.append(2)
        #            x_abnormal2.append([t0_0[i],Ua_0[i]])
        #        else:
        #            data_labeled.append(3)
        #            x_abnormal3.append([t0_0[i],Ua_0[i]])

        #        if zt_0[i]==11:
        #            data_labeled.append(1)
        #            x_abnormal1.append([t0_0[i],Ua_0[i]])
        #        elif zt_0[i]==12:
        #            data_labeled.append(2)
        #            x_abnormal2.append([t0_0[i],Ua_0[i]])
        #        elif zt_0[i]==13:
        #            data_labeled.append(3)
        #            x_abnormal3.append([t0_0[i],Ua_0[i]])
        #        else:
        #            data_labeled.append(0)
        #            x_normal.append([t0_0[i],Ua_0[i]])
        #        if zt_0[i]>0 and zt_0[i]<5:
        #            print(zt_0[i],data_labeled[-1])
        if zt_0[i] == 0 or zt_0[i] == 5:
            data_labeled.append(0)
        else:
            data_labeled.append(1)
    data_labeled.reverse()

    #    x_normal=np.array(x_normal).reshape(-1,2)
    #    x_abnormal1=np.array(x_abnormal1).reshape(-1,2)
    #    x_abnormal2=np.array(x_abnormal2).reshape(-1,2)
    #    x_abnormal3=np.array(x_abnormal3).reshape(-1,2)

    #    fig=plt.figure()

    #    ax1=plt.axes([0.1,0.05,0.8,0.275])
    #    ax1.plot(x_normal[:,0],x_normal[:,1],'o-',markersize=2,
    #             markerfacecolor='white',markeredgecolor='blue')
    #    ax1.plot(x_abnormal1[:,0],x_abnormal1[:,1],'o',markersize=3,
    #             markerfacecolor='white',markeredgecolor='darkorange')
    #    ax1.plot(x_abnormal2[:,0],x_abnormal2[:,1],'o',markersize=3,
    #             markerfacecolor='white',markeredgecolor='red')
    #    ax1.plot(x_abnormal3[:,0],x_abnormal3[:,1],'o',markersize=3,
    #             markerfacecolor='white',markeredgecolor='black')

    #    ax2=plt.axes([0.1,0.375,0.8,0.275])
    #    ax2.plot(x_normal[:,0],x_normal[:,1],'o-',markersize=2,
    #             markerfacecolor='white',markeredgecolor='blue')
    #    ax2.plot(x_abnormal1[:,0],x_abnormal1[:,1],'o',markersize=3,
    #             markerfacecolor='white',markeredgecolor='darkorange')
    #    ax2.plot(x_abnormal2[:,0],x_abnormal2[:,1],'o',markersize=3,
    #             markerfacecolor='white',markeredgecolor='red')

    #    ax3=plt.axes([0.1,0.7,0.8,0.275])
    #    ax3.plot(x_normal[:,0],x_normal[:,1],'o-',markersize=2,
    #             markerfacecolor='white',markeredgecolor='blue')

    #    plt.show()

    return data_labeled


if __name__ == '__main__':
    print('Hello')

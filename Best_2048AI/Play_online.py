from numpy.random import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math
import random
import os
from selenium.common.exceptions import TimeoutException
from selenium.webdriver import Chrome
from selenium.webdriver import ChromeOptions
import pyautogui
import re
import time
import numpy as np


# 新建浏览器访问2048网站
option = ChromeOptions()
option.add_experimental_option('excludeSwitches', ['enable-automation'])
driver = Chrome(options=option)
driver.set_page_load_timeout(10)

url = r'https://play2048.co/'
try :
    driver.get(url)
except TimeoutException:
    print('页面加载超时，停止加载')
    driver.execute_script('window.stop()')


def Board8(image):
    tempB = np.zeros([4, 4])
    for i in range(16):
        t_num = 0
        if image[i // 4][i % 4] != 0:
            t_num = int(math.log2(image[i // 4][i % 4]))
        tempB[i // 4][i % 4] = t_num

    g0 = tempB  # 原始
    g1 = g0[::-1, :]  # 上下180度颠倒
    g2 = g0[:, ::-1]  # 左右180度翻转
    g3 = g2[::-1, :]  # 先上下颠倒，再左右翻转
    r0 = g0.swapaxes(0, 1)  # 转置
    r1 = r0[::-1, :]  # 转置后上下颠倒
    r2 = r0[:, ::-1]  # 转置后左右翻转
    r3 = r2[::-1, :]  # 转置后，先上下颠倒，再左右翻转

    inputB = np.zeros([8, 4, 4, 16])
    #  输入的维度， 8指的是这8个形态（g0到g3，r0到r3）
    #  4， 4 指的是棋盘 4*4的大小
    #  这里输入有意思的是，将0-15位置1表示这个数（其余位就是0），也就是说，理论上这个游戏的一格的上限就是2^15
    gcount = 0
    for g in [g0, r2, g3, r1, g2, r0, g1, r3]:
        for i in range(16):
            inputB[gcount][i // 4][i % 4][int(g[i // 4][i % 4])] = 1
        gcount += 1
    P = np.zeros([4], dtype=np.float32)
    Pcount = np.zeros([4])

    prev = sess.run(p, feed_dict={x_image: inputB})

    B0(prev[0][:], P, Pcount)
    B1(prev[1][:], P, Pcount)
    B2(prev[2][:], P, Pcount)
    B3(prev[3][:], P, Pcount)
    B4(prev[4][:], P, Pcount)
    B5(prev[5][:], P, Pcount)
    B6(prev[6][:], P, Pcount)
    B7(prev[7][:], P, Pcount)

    return P, Pcount


def B0(pre, P, Pcount):  # 0123
    for i in range(4):
        P[i] += pre[i]
    Pcount[np.argmax(pre)] += 1


def B1(pre, P, Pcount):  # 12840
    tempP = np.zeros([4])
    tempP[0] = pre[1]
    tempP[1] = pre[2]
    tempP[2] = pre[3]
    tempP[3] = pre[0]
    Pcount[np.argmax(tempP)] += 1
    for i in range(4):
        P[i] += tempP[i]


def B2(pre, P, Pcount):  # 15141312
    tempP = np.zeros([4])
    tempP[0] = pre[2]
    tempP[1] = pre[3]
    tempP[2] = pre[0]
    tempP[3] = pre[1]
    Pcount[np.argmax(tempP)] += 1
    for i in range(4):
        P[i] += tempP[i]


def B3(pre, P, Pcount):  # 371115
    tempP = np.zeros([4])
    tempP[0] = pre[3]
    tempP[1] = pre[0]
    tempP[2] = pre[1]
    tempP[3] = pre[2]
    Pcount[np.argmax(tempP)] += 1
    for i in range(4):
        P[i] += tempP[i]


def B4(pre, P, Pcount):  # 3210
    tempP = np.zeros([4])
    tempP[0] = pre[0]
    tempP[1] = pre[3]
    tempP[2] = pre[2]
    tempP[3] = pre[1]
    Pcount[np.argmax(tempP)] += 1
    for i in range(4):
        P[i] += tempP[i]


def B5(pre, P, Pcount):  # 04812
    tempP = np.zeros([4])
    tempP[0] = pre[3]
    tempP[1] = pre[2]
    tempP[2] = pre[1]
    tempP[3] = pre[0]
    Pcount[np.argmax(tempP)] += 1
    for i in range(4):
        P[i] += tempP[i]


def B6(pre, P, Pcount):  # 12131415
    tempP = np.zeros([4])
    tempP[0] = pre[2]
    tempP[1] = pre[1]
    tempP[2] = pre[0]
    tempP[3] = pre[3]
    Pcount[np.argmax(tempP)] += 1
    for i in range(4):
        P[i] += tempP[i]


def B7(pre, P, Pcount):  # 151173
    tempP = np.zeros([4])
    tempP[0] = pre[1]
    tempP[1] = pre[0]
    tempP[2] = pre[3]
    tempP[3] = pre[2]
    Pcount[np.argmax(tempP)] += 1
    for i in range(4):
        P[i] += tempP[i]


x_image = tf.placeholder(tf.float32, [None, 4, 4, 16])
W_conv1 = tf.Variable(tf.truncated_normal([2, 2, 16, 222], stddev=0.1))
h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
b_conv1 = tf.Variable(tf.constant(0.1, shape=[222]))
relu1 = tf.nn.relu(h_conv1 + b_conv1)

W_conv2 = tf.Variable(tf.truncated_normal([2, 2, 222, 222], stddev=0.1))
h_conv2 = tf.nn.conv2d(relu1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
b_conv2 = tf.Variable(tf.constant(0.1, shape=[222]))
relu2 = tf.nn.relu(h_conv2 + b_conv2)

W_conv3 = tf.Variable(tf.truncated_normal([2, 2, 222, 222], stddev=0.1))
h_conv3 = tf.nn.conv2d(relu2, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
b_conv3 = tf.Variable(tf.constant(0.1, shape=[222]))
relu3 = tf.nn.relu(h_conv3 + b_conv3)

W_conv4 = tf.Variable(tf.truncated_normal([2, 2, 222, 222], stddev=0.1))
h_conv4 = tf.nn.conv2d(relu3, W_conv4, strides=[1, 1, 1, 1], padding='SAME')
b_conv4 = tf.Variable(tf.constant(0.1, shape=[222]))
relu4 = tf.nn.relu(h_conv4 + b_conv4)

W_conv5 = tf.Variable(tf.truncated_normal([2, 2, 222, 222], stddev=0.1))
h_conv5 = tf.nn.conv2d(relu4, W_conv5, strides=[1, 1, 1, 1], padding='SAME')
b_conv5 = tf.Variable(tf.constant(0.1, shape=[222]))
relu5 = tf.nn.relu(h_conv5 + b_conv5)
relu5_flat = tf.reshape(relu5, [-1, 4 * 4 * 222])
w0 = tf.Variable(tf.zeros([4 * 4 * 222, 4]))
b0 = tf.Variable(tf.zeros([4]))
p = tf.nn.softmax(tf.matmul(relu5_flat, w0) + b0)

t = tf.placeholder(tf.float32, [None, 4])
loss = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer().minimize(loss)

# 正答率
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=None)
saver.restore(sess, r'modle/c5-600000000')  #  读取模型


#  输入网页源码，输出棋盘格 4*4的ndarray
def get_board(source):
    pattern = r'tile tile-\d+ tile-position-\d+-\d+'
    strlist = re.findall(pattern, source)
    board = np.zeros((4, 4))
    for s in strlist:
        value, col, row = re.findall(r'\d+', s)
        value, col, row = int(value), int(col), int(row)
        board[row-1][col-1] = max(board[row-1][col-1],  value)
    return board

#  game begin
while True:
    time.sleep(0.4)
    #  从网页获取 4*4的棋盘
    source = driver.page_source
    board = get_board(source)

    #  通过模型获取结果
    prev, Pcount = Board8(board)
    # select 0123分别代表 上右下左
    select = -1
    pmax = -1
    for i in range(4):
        if pmax < Pcount[i]:
            pmax = Pcount[i]
            select = i
        elif pmax == Pcount[i] and prev[select] < prev[i]:
            pmax = Pcount[i]
            select = i

    # 行为
    pyautogui.press(['up', 'right','down', 'left'][select])
    print(['up', 'right','down', 'left'][select])


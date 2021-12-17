from selenium.common.exceptions import TimeoutException
from selenium.webdriver import Chrome
from selenium.webdriver import ChromeOptions
import pyautogui
import re
import time
import numpy as np
from tensorflow.keras.models import load_model


# 新建浏览器访问2048网站
option = ChromeOptions()
option.add_experimental_option('excludeSwitches', ['enable-automation'])
driver = Chrome(options=option)
driver.set_page_load_timeout(3)

url = r'https://play2048.co/'
try :
    driver.get(url)
except TimeoutException:
    print('页面加载超时，停止加载')
    driver.execute_script('window.stop()')

#  加载AI
model = load_model('whale_dqn2048.h5')


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


orginal_board = np.zeros((4, 4))
orginal_action = 0

while True:
    time.sleep(1)

    source = driver.page_source
    state = get_board(source)

    if np.sum(np.abs(state - orginal_board)) == 0:
        print('over and operation is {}'.format(orginal_action))
        break
    else:
        orginal_board = state

    state = state.reshape([-1, ])
    state = state[np.newaxis, :]
    state = np.log(state + 1) / 16
    action = model.predict(state).argmax()

    pyautogui.press(['up', 'down', 'left', 'right'][action])
    orginal_action = action


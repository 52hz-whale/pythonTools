from selenium.webdriver import Chrome
from selenium.webdriver import ChromeOptions
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
import keyboard
import pyperclip
import time
import sys
from PyQt5.QtWidgets import QWidget, QMessageBox, QApplication




# app = QApplication(sys.argv)
# window = TranslationBox('')
# window.show()
# sys.exit(app.exec())

option = ChromeOptions()
option.add_argument('headless')  # 设置浏览器隐藏
driver = Chrome(options=option)
driver.set_page_load_timeout(5)

url = 'https://translate.google.cn/?sl=en&tl=zh-CN&op=translate'  #  中英互译的网址
try :
    driver.get(url)
except TimeoutException:
    print('页面加载超时，停止加载')
    driver.execute_script('window.stop()')

original_text = ''  #  判断翻译结果界面有没有更新
print('realy to translate!!!')
while True:
    keyboard.wait(hotkey='ctrl+c')

    #  等待剪贴板写入
    time.sleep(0.3)
    english_text = pyperclip.paste()

    english_text = english_text.replace('\r\n', '')  #  去掉pdf中的换行符

    input_elem = driver.find_element(By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[3]/c-wiz[1]/span/span/div/textarea')
    input_elem.clear()
    input_elem.send_keys(english_text)

    #  不断检查页面元素，判断是否翻译完成
    while True:
        try:
            t = driver.find_element(By.XPATH,'//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[3]/c-wiz[2]/div[6]/div/div[1]/span[1]/span[1]/span')
            t = t.get_attribute('innerHTML')
            if t == original_text:
                time.sleep(0.1)
                continue
            break
        except:
            pass
        else:
            time.sleep(0.1)

    chinese_text = ""
    try:
        i = 1
        while True:
            chinese_text += driver.find_element(By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[3]/c-wiz[2]/div[6]/div/div[1]/span[1]/span[' \
                                                + str(i) + ']/span').get_attribute('innerHTML')
            if i == 1:
                original_text = chinese_text
            chinese_text += '\r\n'
            i += 1
    except:
        pass
    else:
        pass

    print(chinese_text)
    print('translation over')


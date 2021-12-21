from PyQt5.QtWidgets import QWidget, QMessageBox, QApplication
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtCore import Qt, pyqtSignal
import sys
import time
from window import *
from selenium.webdriver import Chrome
from selenium.webdriver import ChromeOptions
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
import keyboard
import pyperclip
from PyQt5.Qt import QThread, QMutex
import PyHook3 as pyHook
import pythoncom

ctrl_c_flag= True

class KeyBoardThread(QThread):
    ctrl_c_signal = pyqtSignal()

    def __init__(self):
        super().__init__()

    def run(self) -> None:
        hm = pyHook.HookManager()
        hm.KeyDown = self.onKeyboardEvent
        hm.HookKeyboard()
        pythoncom.PumpMessages()

    def onKeyboardEvent(self, event):
        if event.Ascii == 3:
            if ctrl_c_flag:
                print('ctrl + c' + '你好呀')
                self.ctrl_c_signal.emit()
        return True


class MyMainWindow(QWidget, Ui_Form):
    def __init__(self):
        super(QWidget, self).__init__()
        self.setupUi(self)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        font = QtGui.QFont()
        font.setPixelSize(30)
        font.setFamily('方正静蕾简体加粗版')
        font.setBold(True)
        self.textEdit.setFont(font)
        self.textEdit.setStyleSheet("background-color:transparent; color: rgb(255, 66, 93)")
        self.textEdit.setText('Welcome~···\nWaiting for initialization···\n')
        self.original_text = ''  # 判断翻译结果界面有没有更新

    def init(self):
        #  初始化浏览器
        option = ChromeOptions()
        option.add_argument('headless')  # 设置浏览器隐藏
        self.driver = Chrome(options=option)
        self.driver.set_page_load_timeout(10)
        #  访问翻译api
        url = 'https://translate.google.cn/?sl=en&tl=zh-CN&op=translate'  # 中英互译的网址

        try:
            self.driver.get(url)
        except TimeoutException:
            print('页面加载超时，停止加载')
            self.driver.execute_script('window.stop()')

        # 判断能不能翻译了
        try:
            self.driver.find_element(By.XPATH,
                                         '//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[3]/c-wiz[1]/span/span/div/textarea')
        except:  # 如果发生错误，也就是找不到
            self.textEdit.setText('ERROR\n' + 'cannot link to \n' + 'https://translate.google.cn/' + '\nplease check the internet and try again')
            return


        self.thread1 = KeyBoardThread()
        self.thread1.ctrl_c_signal.connect(self.showResults)
        self.thread1.start()

        self.textEdit.setText('initialized successfully!!!\npress ctrl+v and the clipboard will be translated :)')

    def showResults(self):
        global ctrl_c_flag
        ctrl_c_flag = False
        time.sleep(0.3)  #  等待剪贴板写入
        english_text = pyperclip.paste()
        english_text = english_text.replace('\r\n', '')  # 去掉pdf中的换行符

        input_elem = self.driver.find_element(By.XPATH,
                                         '//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[3]/c-wiz[1]/span/span/div/textarea')
        input_elem.clear()
        input_elem.send_keys(english_text)
        #  不断检查页面元素，判断是否翻译完成
        while True:
            try:
                t = self.driver.find_element(By.XPATH,
                                        '//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[3]/c-wiz[2]/div[6]/div/div[1]/span[1]/span[1]/span')
                t = t.get_attribute('innerHTML')
                if t == self.original_text:
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
                chinese_text += self.driver.find_element(By.XPATH,
                                                    '//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[3]/c-wiz[2]/div[6]/div/div[1]/span[1]/span[' \
                                                    + str(i) + ']/span').get_attribute('innerHTML')
                if i == 1:
                    self.original_text = chinese_text
                chinese_text += '\r\n'
                i += 1
        except:
            pass
        else:
            pass

        self.textEdit.setText(chinese_text)
        ctrl_c_flag = True


    # def keyPressEvent(self, QKeyEvent):
    #     print(QKeyEvent.key())
    #     if QKeyEvent.modifiers() == Qt.ControlModifier and QKeyEvent.key() == Qt.Key_C:  # 按下的是 ctrl + c
    #         print('按键捕捉了')
    #         self.showResults()

if __name__ == '__main__':

    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    myWin.init()
    sys.exit(app.exec())

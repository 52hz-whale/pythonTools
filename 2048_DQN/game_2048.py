'''
@Author:Drake
2048 游戏环境
存活：+1
消去组合一对数：+10
死局：-5

'''

import numpy as np


HEIGHT = 4
WIDTH = 4

CLEAN_REWARD = 1
SURVIVE_REWARD = 1


class Game2048(object):
    def __init__(self):
        self.actions = [0, 1, 2, 3]  # 操作：0123分别代表 上下左右
        self.n_actions = len(self.actions)  # int 4, 四种操作
        self.n_features = HEIGHT * WIDTH
        self.board = np.zeros(shape=[HEIGHT, WIDTH], dtype=np.int32)
        self.reset()

    def reset(self):
        init_places = np.random.choice(a=np.array([i for i in range(16)], dtype=np.int32), size=2, replace=False)
        init_digitals = np.random.choice(a=np.array([2, 4], dtype=np.int32), size=2, replace=True)
        self.board = np.zeros(shape=[HEIGHT, WIDTH], dtype=np.int32)
        self.n_step = 0
        self.score = 0
        for i in range(2):
            self.board[init_places[i] // HEIGHT][init_places[i] % WIDTH] = init_digitals[i]

    #  输入的op为 0123，对应上下左右
    def step(self, op):

        done = False

        #  先移动，得到一个新板子
        next_board, clear_score = self._move(self.board.copy(), op)
        if np.sum(np.abs(next_board - self.board)) == 0:
            reward = -20
            done = True
            return next_board, reward, done

        #  空位有没有！有的话加！
        x_t = np.where(next_board == 0)[0]  # 为0的空位list
        y_t = np.where(next_board == 0)[1]
        if len(x_t) == 0:  # 没有空位0了
            reward = -20
            done = True
            return next_board, reward, done

        tmp = np.random.choice(a=len(x_t), size=1)[0]  # 选择其中一个空位
        xx = x_t[tmp]
        yy = y_t[tmp]
        new_score = np.random.choice(a=np.array([2, 4]), size=1)[0]
        next_board[xx][yy] = new_score
        self.board = next_board.copy()

        reward = clear_score * CLEAN_REWARD  + SURVIVE_REWARD
        self.n_step += 1
        self.score += reward

        return next_board, reward, done

    #  四个方向试着动一动，看看能不能更新board，如果都不能更新board，那就是结束了（动不了了）
    def _terminal(self, board):
        for i in range(self.n_actions):
            tmp_board, _ = self._move(board.copy(), i)
            if np.sum(np.abs(tmp_board - board)) != 0:
                return False
        return True

    def _move(self, board, op_num):
        valid = np.ones(shape=[HEIGHT, WIDTH])
        clear_score = 0
        if op_num == 0:  # 上
            for i in range(1, HEIGHT):
                for j in range(0, WIDTH):
                    # 找到一个非0的格子
                    # self.show_game()
                    if board[i][j] == 0:
                        continue
                    cur_y = i
                    while cur_y > 0:
                        if board[cur_y - 1][j] == 0:
                            board[cur_y - 1][j] = board[cur_y][j]
                            board[cur_y][j] = 0
                            cur_y -= 1
                        else:
                            break

                    if board[cur_y][j] == board[cur_y - 1][j] and valid[cur_y - 1][j] == 1:
                        valid[cur_y - 1][j] = 0
                        board[cur_y - 1][j] *= 2
                        board[cur_y][j] = 0
                        clear_score += board[cur_y - 1][j]
            # print('valid:\n', valid, '\n')

        elif op_num == 1:  # 下
            for i in range(1, HEIGHT):
                for j in range(0, WIDTH):
                    # 找到一个非0的格子
                    # self.show_game()
                    if board[HEIGHT - i - 1][j] == 0:
                        continue
                    cur_y = HEIGHT - i - 1
                    while cur_y < HEIGHT - 1:
                        if board[cur_y + 1][j] == 0:
                            board[cur_y + 1][j] = board[cur_y][j]
                            board[cur_y][j] = 0
                            cur_y += 1
                        else:
                            break
                    if cur_y < HEIGHT - 1 and board[cur_y][j] == board[cur_y + 1][j] and valid[cur_y + 1][j] == 1:
                        valid[cur_y + 1][j] = 0
                        board[cur_y + 1][j] *= 2
                        board[cur_y][j] = 0
                        clear_score += board[cur_y + 1][j]
            # print('valid:\n', valid, '\n')

        elif op_num == 2:  # 左
            for j in range(1, WIDTH):
                for i in range(0, HEIGHT):
                    # 找到一个非0的格子
                    # self.show_game()
                    if board[i][j] == 0:
                        continue
                    cur_x = j
                    while cur_x > 0:
                        if board[i][cur_x - 1] == 0:
                            board[i][cur_x - 1] = board[i][cur_x]
                            board[i][cur_x] = 0
                            cur_x -= 1
                        else:
                            break

                    if board[i][cur_x] == board[i][cur_x - 1] and valid[i][cur_x - 1] == 1:
                        valid[i][cur_x - 1] = 0
                        board[i][cur_x - 1] *= 2
                        board[i][cur_x] = 0
                        clear_score += board[i][cur_x - 1]
            # print('valid:\n', valid, '\n')

        elif op_num == 3:  # 右
            for j in range(1, WIDTH):
                for i in range(0, HEIGHT):
                    # 找到一个非0的格子
                    # self.show_game()
                    if board[i][WIDTH - j - 1] == 0:
                        continue
                    cur_x = WIDTH - j - 1
                    while cur_x < WIDTH - 1:
                        if board[i][cur_x + 1] == 0:
                            board[i][cur_x + 1] = board[i][cur_x]
                            board[i][cur_x] = 0
                            cur_x += 1
                        else:
                            break
                    if cur_x < WIDTH - 1 and board[i][cur_x] == board[i][cur_x + 1] and valid[i][cur_x + 1] == 1:
                        valid[i][cur_x + 1] = 0
                        board[i][cur_x + 1] *= 2
                        board[i][cur_x] = 0
                        clear_score += board[i][cur_x + 1]
            # print('valid:\n', valid, '\n')
        return board, clear_score

    def get_plat_state(self):
        return self.board.reshape([-1,])

    def get_state(self):
        return self.board.copy()

    def get_score(self):
        return self.score

    def show_game(self):
        print(self.board)

    def play_human(self):
        while True:
            self.show_game()
            digit = input('8246分别是上下左右')
            if digit == '8':
                self.step(0)
            if digit == '2':
                self.step(1)
            if digit == '4':
                self.step(2)
            if digit == '6':
                self.step(3)

            print('score:', self.get_score())

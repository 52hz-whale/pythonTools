from game_2048 import Game2048
import numpy as np
from tensorflow.keras.models import load_model
import time
import keyboard

model = load_model('whale_dqn2048.h5')
game = Game2048()
game.reset()
s = game.get_state()
game_step = 0

while True:
    print(game.board)

    state = s.reshape([-1, ])
    state = state[np.newaxis, :]
    state = np.log(state + 1) / 16
    action = model.predict(state).argmax()

    s_, r, done = game.step(action)
    s = s_

    if done:
        print('final:\n', game.board)
        print('score:', game.get_score(), ' board sum:', np.sum(game.board), ' play step:', game.n_step)
        break

    game_step += 1
    keyboard.wait(hotkey='ctrl')

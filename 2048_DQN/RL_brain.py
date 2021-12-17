import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.keras.activations import sigmoid, relu, softmax
from tensorflow.keras.optimizers import RMSprop
import numpy as np


class DeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 lr=0.01,
                 reward_decay=0.9,
                 epsilon=0.96,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 train_epochs=10,
                 epsilon_increment=None,
                 ):
        self.n_actions = n_actions  #  4个，上下左右
        self.n_features = n_features  # 16个，4*4的网格展平了
        self.lr = lr
        self.gamma = reward_decay
        self.epsilon_max = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = epsilon_increment
        self.epsilon = 0 if epsilon_increment is not None else self.epsilon_max
        self.train_epochs = train_epochs

        self.learn_step_counter = 0
        self._build_net()
        # 存储空间下标：[s, a, r, s_]  状态是16维，展平了的数组, a是1维， r是1维
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 1 + 1))

    def preprocess_state(self, state):  # 预处理成 0-1之间的小数
        return np.log(state + 1) / 16

    def choose_action(self, state):
        state = self.preprocess_state(state)
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action_list = self.q_eval_model.predict(state)
            action = action_list.argmax()
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def _build_net(self):
        self.q_eval_model = Sequential(name='evaluate_net')
        self.q_eval_model.add(Dense(input_shape=[self.n_features], units=32, activation='relu'))
        self.q_eval_model.add(Dense(self.n_actions))

        self.q_target_model = Sequential(name='target_net')
        self.q_target_model.add(Dense(input_shape=[self.n_features], units=32, activation='relu'))
        self.q_target_model.add(Dense(self.n_actions))

        self.q_eval_model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=self.lr), metrics=['accuracy'])
        self.q_target_model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=self.lr), metrics=['accuracy'])

    def target_replace_op(self):
        p = self.q_eval_model.get_weights()
        self.q_target_model.set_weights(p)

    def store_memory(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        s = self.preprocess_state(s)
        s_ = self.preprocess_state(s_)
        memory = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = memory
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_replace_op()
            print('target_params_replaced!')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        s = batch_memory[:, 0:self.n_features]
        s_ = batch_memory[:, -self.n_features:]
        a = batch_memory[:, self.n_features].astype(np.int32)
        r = batch_memory[:, self.n_features+1]

        q_next = self.q_target_model.predict(s_)
        q_eval = self.q_eval_model.predict(s)

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, a] = r + self.gamma * np.max(q_next, axis=1)
        self.q_eval_model.fit(s, q_target, epochs=self.train_epochs, verbose=0)

        self.learn_step_counter += 1

    def save_model(self):
        self.q_eval_model.save('whale_dqn2048.h5')

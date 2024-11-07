import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import torch.multiprocessing as mp
import torch.optim as optim
from torch.autograd import Variable
from collections import deque
import hashlib
import json
import random
from random import randint
import math
import argparse
import pickle
from collections import namedtuple
from itertools import count
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import asyncio
from structure.point import Point
from structure.point_set import PointSet
from structure.constant import EQN2
from structure import constant
from structure.hyperplane import Hyperplane
from structure.hyperplane_set import HyperplaneSet
from structure import others
import time
import random


# one_layer_nn and Agent are for insertion
class one_layer_nn(nn.Module):

    def __init__(self, ALPHA, input_size, hidden_size1, output_size, is_gpu=True):
        super(one_layer_nn, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1, bias=False)
        self.layer2 = nn.Linear(hidden_size1, output_size, bias=False)

        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA) # learning rate
        self.loss = nn.MSELoss()
        if is_gpu:
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        # self.device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # sm = nn.Softmax(dim=0)
        sm = nn.SELU() # SELU activation function

        # Variable is used for older torch
        x = Variable(state, requires_grad=False)
        y = self.layer1(x)
        y_hat = sm(y)
        z = self.layer2(y_hat)
        scores = z
        return scores


class Agent():
    def __init__(self, gamma, epsilon, alpha, maxMemorySize, batch_size, action_space_size, dim, is_gpu, epsEnd=0.0, replace=5):
        # epsEnd was set to be 0.05 initially
        self.GAMMA = gamma  # discount factor
        self.EPSILON = epsilon  # epsilon greedy to choose action
        self.EPS_END = epsEnd  # minimum epsilon
        self.memSize = maxMemorySize  # memory size
        self.batch_size = batch_size
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.memCntr = 0
        self.replace_target_cnt = replace
        self.action_space_size = action_space_size
        self.Q_eval = one_layer_nn(alpha, (action_space_size + 3) * dim + 1, 64, action_space_size, is_gpu)
        self.Q_target = one_layer_nn(alpha, (action_space_size + 3) * dim + 1, 64, action_space_size, is_gpu)
        self.state_action_pairs = []
        self.loss_history = []

    def storeTransition(self, state, action, reward, state_):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.memCntr % self.memSize] = [state, action, reward, state_]
        self.memCntr = self.memCntr + 1

    def sample(self):
        if len(self.memory) < self.batch_size:
            return self.memory
        else:
            return random.sample(self.memory, self.batch_size)

    def chooseAction(self, observation):
        observation = torch.FloatTensor(observation).to(self.Q_eval.device)
        rand = np.random.random()
        actions = self.Q_eval.forward(observation)
        if rand < 1 - self.EPSILON:
            # action = torch.argmax(actions).item()
            useless_value, action = torch.max(actions[:self.action_space_size], 0)
        else:
            # action = np.random.choice(self.actionSpace)
            action = randint(0, self.action_space_size - 1)
        self.steps = self.steps + 1
        return int(action)

    def learn(self):
        if len(self.memory) <= 0:
            return

        self.Q_eval.optimizer.zero_grad()
        if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

        state = []
        action = []
        reward = []
        next_state = []
        for transition in self.sample():
            state.append(list(transition[0]))
            action.append(transition[1])
            reward.append(transition[2])
            next_state.append(list(transition[3]))
        state_action_values = self.Q_eval.forward(torch.FloatTensor(state).to(self.Q_eval.device))
        next_state_action_values = self.Q_target.forward(torch.FloatTensor(next_state).to(self.Q_eval.device))

        # maxA = torch.argmax(Qnext, dim=0)
        max_value, maxA = torch.max(next_state_action_values, 1)  # 1 represents max based on each column
        actions = torch.tensor(action, dtype=torch.int64)
        actions = Variable(actions, requires_grad=False)
        rewards = torch.FloatTensor(reward)
        rewards = Variable(rewards, requires_grad=False)
        state_action_values_target = state_action_values.clone()

        for i in range(len(maxA)):
            temp = rewards[i] + self.GAMMA * (max_value[i])
            state_action_values_target[i, actions[i]] = temp

        if self.EPSILON > self.EPS_END:
            self.EPSILON *= 0.99

        loss = self.Q_eval.loss(state_action_values, state_action_values_target.detach()).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter = self.learn_step_counter + 1

        self.loss_history.append(loss.item())
        # 定期打印loss和replay memory大小
        if self.learn_step_counter % 100 == 0:
            print(f"Step: {self.learn_step_counter}, Loss: {loss.item()}, Replay Memory Size: {len(self.memory)}")

        # be careful of the input format
        # loss = self.Q_eval.loss(Qpred, Qtarget.detach()).to(self.Q_eval.device)


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

async def highRL_Train(pset: PointSet, u: Point, epsilon, dataset_name, train=True, trainning_epoch=1000, action_space_size=5,
                       gamma=0.8, epsilonTrain=0.5, alpha=0.05, maxMemorySize=5000, batch_size=64, websocket=None, session_id=None):

    start_time = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    dim = pset.points[0].dim

    # init agent
    brain = Agent(gamma=gamma, epsilon=epsilonTrain, alpha=alpha, maxMemorySize=maxMemorySize, batch_size=batch_size,
                  action_space_size=action_space_size, dim=dim, is_gpu=True)
    torch.save(brain.Q_eval.state_dict(), '_high_model_dim_' + dataset_name + '_' + str(epsilon) + '_' + str(trainning_epoch) + '_' + str(action_space_size) + '_' + '.mdl')
    brain.Q_eval.load_state_dict(torch.load('_high_model_dim_' + dataset_name + '_' + str(epsilon) + '_' + str(trainning_epoch) + '_' + str(action_space_size) + '_' + '.mdl'))

    if train:  # training
        num_question_history = []
        time_history = []
        for epo in range(trainning_epoch):
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                data = json.loads(message)
                if data.get("type") == "stop_training":
                    print("Received stop signal, ending training.")
                    await websocket.send(json.dumps({"type": "training_stopped"}))
                    return  # 退出训练
            except asyncio.TimeoutError:
                pass

            trainning_question = 0
            utility_range = HyperplaneSet(dim)
            state = utility_range.get_state_high()  # part of state
            dist = np.linalg.norm(utility_range.upper_point.coord - utility_range.lower_point.coord)
            pivot_pt = pset.find_top_k(utility_range.in_center, 1)[0]
            while dist > epsilon * np.sqrt(dim) * 2:  # stopping condition
                h_cand = utility_range.find_candidate_hyperplanes(pset, action_space_size, pivot_pt, epsilon)
                if len(h_cand) <= 0:
                    break
                while len(h_cand) < action_space_size:
                    h_cand.append(h_cand[0])
                for i in range(action_space_size):
                    state = np.concatenate((state, h_cand[i].norm))
                action = brain.chooseAction(state)
                brain.state_action_pairs.append([state, action])

                # interaction
                trainning_question += 1
                value1 = u.dot_prod(h_cand[action].p1)
                value2 = u.dot_prod(h_cand[action].p2)
                if value1 > value2:
                    h = Hyperplane(p1=h_cand[action].p2, p2=h_cand[action].p1)
                    utility_range.hyperplanes.append(h)
                    pivot_pt = h_cand[action].p1
                else:
                    h = Hyperplane(p1=h_cand[action].p1, p2=h_cand[action].p2)
                    utility_range.hyperplanes.append(h)
                    pivot_pt = h_cand[action].p2

                utility_range.set_ext_pts()
                state = utility_range.get_state_high()  # part of state
                dist = np.linalg.norm(utility_range.upper_point.coord - utility_range.lower_point.coord)

            # include the last state
            state = np.append(state, np.zeros(action_space_size * dim))
            brain.state_action_pairs.append([state, -1])

            # compute reward
            ind = 0
            while ind < len(brain.state_action_pairs) - 2:
                brain.storeTransition(brain.state_action_pairs[ind][0], brain.state_action_pairs[ind][1], -2, brain.state_action_pairs[ind + 1][0])
                ind = ind + 1
            brain.storeTransition(brain.state_action_pairs[ind][0], brain.state_action_pairs[ind][1], 100, brain.state_action_pairs[ind + 1][0])
            brain.learn()
            brain.state_action_pairs = []
            if (epo + 1) % 10 == 0:
                print("Model saved")
                folder_path = "models/high_model"
                os.makedirs(folder_path, exist_ok=True)
                file_path = os.path.join(folder_path, '_high_model_dim_' + dataset_name + '_' + str(epsilon) + '_' + str(epo + 1) + '_' + str(action_space_size) + '_' + '.mdl')
                torch.save(brain.Q_eval.state_dict(), file_path)

                ##################################################
                ##################################################
                # Inference
                brain_test = Agent(gamma=0.80, epsilon=0.0, alpha=0.003, maxMemorySize=5000, batch_size=64, action_space_size=action_space_size, dim=dim, is_gpu=False)
                brain_test.Q_eval.load_state_dict(torch.load('models/high_model/_high_model_dim_' + dataset_name + '_' + str(epsilon) + '_' + str(epo + 1) + '_' + str(action_space_size) + '_' + '.mdl'))
                num_question_test = 0
                utility_range_test = HyperplaneSet(dim)
                state_test = utility_range_test.get_state_high() # part of state
                dist_test = np.linalg.norm(utility_range_test.upper_point.coord - utility_range_test.lower_point.coord)
                pivot_pt_test = pset.find_top_k(utility_range_test.in_center, 1)[0]

                print(pivot_pt_test)
                while dist_test > epsilon * np.sqrt(dim) * 2: # stopping condition
                    h_cand_test = utility_range_test.find_candidate_hyperplanes(pset, action_space_size, pivot_pt_test, epsilon)
                    if len(h_cand_test) <= 0:
                        break
                    while len(h_cand_test) < action_space_size:
                        h_cand_test.append(h_cand_test[0])
                    for i in range(action_space_size):
                        state_test = np.concatenate((state_test, h_cand_test[i].norm))
                    action_test = brain_test.chooseAction(state_test)

                    # interaction
                    num_question_test += 1
                    value1_test = u.dot_prod(h_cand_test[action_test].p1)
                    value2_test = u.dot_prod(h_cand_test[action_test].p2)
                    if value1_test > value2_test:
                        h = Hyperplane(p1=h_cand_test[action_test].p2, p2=h_cand_test[action_test].p1)
                        utility_range_test.hyperplanes.append(h)
                        pivot_pt_test = h_cand_test[action_test].p1
                    else:
                        h = Hyperplane(p1=h_cand_test[action_test].p1, p2=h_cand_test[action_test].p2)
                        utility_range_test.hyperplanes.append(h)
                        pivot_pt_test = h_cand_test[action_test].p2

                    state_test = utility_range_test.get_state_high()  # part of state
                    dist_test = np.linalg.norm(utility_range_test.upper_point.coord - utility_range_test.lower_point.coord)

                # replay memory, loss, time, number of questions
                print(num_question_test)
                time2 = time.time() - start_time
                time_history.append(time2)
                num_question_history.append(num_question_test)
                memory_serializable = [
                    [transition[0].tolist(), transition[1], transition[2], transition[3].tolist()]
                    for transition in brain.memory
                ]
                data = {
                    "type": "training_data",
                    "time_history": time_history,
                    "num_question_history": num_question_history,
                    "memory": memory_serializable
                }
                await websocket.send(json.dumps(data))


        plt.plot(num_question_history)
        plt.xlabel("Training Steps")
        plt.ylabel("Questions")
        plt.show()
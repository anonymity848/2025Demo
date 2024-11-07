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
import copy
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
from structure.agent import get_response

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
    def __init__(self, gamma, epsilon, alpha, maxMemorySize, batch_size, action_space_size, dim, is_gpu, epsEnd=0.1, replace=20):
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

        if self.steps > 400000:
            if self.EPSILON > self.EPS_END:
                self.EPSILON = self.EPSILON * 0.99
            else:
                self.EPSILON = self.EPS_END

        loss = self.Q_eval.loss(state_action_values, state_action_values_target.detach()).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter = self.learn_step_counter + 1

        # be careful of the input format
        # loss = self.Q_eval.loss(Qpred, Qtarget.detach()).to(self.Q_eval.device)


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class TreeNode:
    def __init__(self, utility_range):
        self.utilityR = copy.deepcopy(utility_range)
        self.left = None
        self.right = None
        self.p1 = None
        self.p2 = None

    # p1, p2 are the results in the last node presented to the user
    def generate_tree(self, p1, p2, pivot_pt, pset, brain, action_space_size, epsilon): # better point is p1
        dim = pivot_pt.dim
        if p1 is not None and p2 is not None:
            h = Hyperplane(p1=p1, p2=p2)
            self.utilityR.hyperplanes.append(h)
        state = self.utilityR.get_state_high()  # part of state
        dist = np.linalg.norm(self.utilityR.upper_point.coord - self.utilityR.lower_point.coord)
        print("dist: ", dist)
        if dist < epsilon * np.sqrt(dim) * 2:  # stopping condition
            cor_center = Point(dim)
            cor_center.coord = (self.utilityR.upper_point.coord + self.utilityR.lower_point.coord) / 2
            pset_org = PointSet(f'carOrg.txt')
            self.p1 = pset_org.points[pset.find_top_k(cor_center, 1)[0].id]
            return
        h_cand = self.utilityR.find_candidate_hyperplanes(pset, action_space_size, pivot_pt, epsilon)
        if len(h_cand) <= 0:
            return
        while len(h_cand) < action_space_size:
            h_cand.append(h_cand[0])
        for i in range(action_space_size):
            state = np.concatenate((state, h_cand[i].norm))
        action = brain.chooseAction(state)
        pset_org = PointSet(f'carOrg.txt')
        self.p1 = pset_org.points[h_cand[action].p1.id]
        self.p2 = pset_org.points[h_cand[action].p2.id]

        self.left = TreeNode(self.utilityR)
        self.right = TreeNode(self.utilityR)
        self.left.generate_tree(h_cand[action].p2, h_cand[action].p1, h_cand[action].p1, pset, brain, action_space_size, epsilon)
        self.right.generate_tree(h_cand[action].p1, h_cand[action].p2, h_cand[action].p2, pset, brain, action_space_size, epsilon)

        return

# 将二叉树转为字典，便于序列化
def tree_to_dict(node):
    # Check if the node is actually an instance of the expected class
    if node is None:
        return None
    if not isinstance(node, TreeNode):
        return node  # This will avoid the recursive call on a dictionary

    return {
        'inner_center': node.utilityR.in_center.coord.tolist(),
        'inner_radius': node.utilityR.in_radius,
        'upper_point': node.utilityR.upper_point.coord.tolist(),
        'lower_point': node.utilityR.lower_point.coord.tolist(),
        'p1': node.p1.coord.tolist() if node.p1 else None,
        'p2': node.p2.coord.tolist() if node.p2 else None,
        'left': tree_to_dict(node.left) if node.left else None,
        'right': tree_to_dict(node.right) if node.right else None
    }

# 存储二叉树
def store_tree(tree, filename='binary_tree.json'):
    open(filename, 'w').close()

    with open(filename, 'w') as file:
        json.dump(tree_to_dict(tree), file)

# 读取二叉树
def load_tree(filename='binary_tree.json'):
    with open(filename, 'r') as file:
        return json.load(file)




single_round = (
            "We want to learn the user's weight vector u on attributes. Now we show two points p1 and p2 to the user, "
            "and ask him which one he prefers. Based on the user's choice, we can learn that u.dot_prod(p1) > u.dot_prod(p2) "
            "or u.dot_prod(p1) < u.dot_prod(p2). In this way, we can narrow the domain of the user's weight vector u. Now "
            "the two points are p1 = {p1_coord} and p2 = {p2_coord}. Please analyze what information "
            "of the user's weight vector u we may have a chance to learn.")

async def highRL(pset: PointSet, u: Point, epsilon, dataset_name, train=False, trainning_epoch=10000, action_space_size=5, websocket=None, session_id=None):
    start_time = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    dim = pset.points[0].dim
    # action_space_size = 5

    # use the trained model to interact
    brain = Agent(gamma=0.80, epsilon=0.0, alpha=0.003, maxMemorySize=5000, batch_size=64, action_space_size=action_space_size, dim=dim, is_gpu=False)
    brain.Q_eval.load_state_dict(torch.load('_high_model_dim_' + dataset_name + '_' + str(epsilon) + '_' + str(trainning_epoch) + '_' + str(action_space_size) + '_' + '.mdl'))
    utility_range = HyperplaneSet(dim)
    state = utility_range.get_state_high()  # part of state
    dist = np.linalg.norm(utility_range.upper_point.coord - utility_range.lower_point.coord)
    print("dist: ", dist)
    pivot_pt = pset.find_top_k(utility_range.in_center, 1)[0]

    root = TreeNode(utility_range)
    root.generate_tree(None, None, pivot_pt, pset, brain, action_space_size, epsilon)
    tree_dict = tree_to_dict(root)
    store_tree(root)
    tree = load_tree()
    await websocket.send(json.dumps({"tree": tree}))

    ########################################Interaction############################################
    num_question = 0
    current_path = []
    dist_history = []
    message = json.dumps({"path": current_path})
    await websocket.send(message)
    utility_range = HyperplaneSet(dim)
    state = utility_range.get_state_high()  # part of state
    dist = np.linalg.norm(utility_range.upper_point.coord - utility_range.lower_point.coord)
    dist_history.append(dist)
    vertices = []
    for e in utility_range.ext_pts:
        vertices.append(e.coord.tolist())
    hyperplanes = []
    for hyp in utility_range.hyperplanes:
        hyperplanes.append(hyp.norm.tolist())
    message = {"dist_history": dist_history,
               "vertices": vertices,
               "hyperplanes": hyperplanes}
    await websocket.send(json.dumps(message))
    pivot_pt = pset.find_top_k(utility_range.sample_vector(), 1)[0]
    while dist > epsilon * np.sqrt(dim) * 2:  # stopping condition
        print(epsilon * np.sqrt(dim) * 2)
        h_cand = utility_range.find_candidate_hyperplanes(pset, action_space_size, pivot_pt, epsilon)
        if len(h_cand) <= 0:
            break
        while len(h_cand) < action_space_size:
            h_cand.append(h_cand[0])
        for i in range(action_space_size):
            state = np.concatenate((state, h_cand[i].norm))
        action = brain.chooseAction(state)

        # interaction
        num_question += 1
        value1 = u.dot_prod(h_cand[action].p1)
        value2 = u.dot_prod(h_cand[action].p2)

        choice = await pset.question_for_interaction(websocket, h_cand[action].p1, h_cand[action].p2, num_question)
        if choice == 1:
            h = Hyperplane(p1=h_cand[action].p2, p2=h_cand[action].p1)
            utility_range.hyperplanes.append(h)
            pivot_pt = h_cand[action].p1
            pset.printMiddleSelection(num_question, "AA", h_cand[action].p1, h_cand[action].p2, 1, session_id)
            current_path.append("left")
        else:
            h = Hyperplane(p1=h_cand[action].p1, p2=h_cand[action].p2)
            utility_range.hyperplanes.append(h)
            pivot_pt = h_cand[action].p2
            pset.printMiddleSelection(num_question, "AA", h_cand[action].p1, h_cand[action].p2, 2, session_id)
            current_path.append("right")

        utility_range.set_ext_pts()
        state = utility_range.get_state_high()  # part of state
        dist = np.linalg.norm(utility_range.upper_point.coord - utility_range.lower_point.coord)
        dist_history.append(dist)
        vertices = []
        for e in utility_range.ext_pts:
            vertices.append(e.coord.tolist())
        hyperplanes = []
        for hyp in utility_range.hyperplanes:
            hyperplanes.append(hyp.norm.tolist())
        message = {"dist_history": dist_history,
                   "vertices": vertices,
                   "hyperplanes": hyperplanes}
        await websocket.send(json.dumps(message))

        ####################### Analysis ############################
        vertices_str = "; ".join(
            [f"Vertex {i + 1}: {vertex.coord}" for i, vertex in enumerate(utility_range.ext_pts)])
        multi_round = (
            "There are four attributes, namely Year, Price, Mileage, and Tax, that describe the cars. We want to "
            "learn the user's trade-off among attributes, i.e., the user's weight vector u = (u[0], u[1], u[2], [3]). "
            "Each u[i] represents the importance of the i-th attribute to the user.\n"
            "Currently, we have learned that the domain of the user's weight vector u is within a certain range. "
            f"The vertices of the range are shown below. \n {vertices_str} \n"
            "Based on the information of the user's weight vector u that we have learned so far, I expect you can "
            "describe the user's trade-off among attributes within 50 words."
        )

        message = {"analysis_clear": "Clear"}
        await websocket.send(json.dumps(message))
        messages = [
            {
                "role": "user",
                "content": multi_round
            }
        ]
        async for description_partial in get_response(messages):
            print(description_partial)
            await websocket.send(json.dumps({"analysis_update": description_partial}))
        message = {"analysis_finish": "Finish"}
        await websocket.send(json.dumps(message))
        message = json.dumps({"path": current_path})
        await websocket.send(message)

    # print results
    cor_center = Point(dim)
    cor_center.coord = (utility_range.upper_point.coord + utility_range.lower_point.coord) / 2
    result = pset.find_top_k(cor_center, 1)[0]
    pset_org = PointSet(f'carOrg.txt')
    message = {
        "message": "final result",
        "data_group_1": pset_org.points[result.id].coord.tolist()
    }
    await websocket.send(json.dumps(message))
    ####################### Description ############################
    messages = [
        {
            "role": "user",
            "content": (
                f"The car is described by attributes Year, Price, Mileage, Tax.\n"
                f"Car: {pset_org.points[result.id].coord}\n"
                f"Describe this car within 50 words respectively.\n"
            )
        }
    ]

    async for assistant_output in get_response(messages):
        print(assistant_output)
        await websocket.send(json.dumps({"description_update": assistant_output}))
    # result.printAlgResult("AA", num_question, start_time, 0)



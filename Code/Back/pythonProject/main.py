import uuid

from structure.hyperplane import Hyperplane
from structure.point import Point
from structure.point_set import PointSet
from structure import constant
import time
import highRL
import lowRL
import time
import asyncio
import websockets
import json
import highRL_Train


async def user_study(websocket, session_id):

    epsilon = 0.4
    trainning_size = 1000
    action_size = 5

    response = await websocket.recv()
    data = json.loads(response)

    print(data)

    if data.get("type") == "training":
        params = data.get("params", {})
        is_training = True
        params = data.get("params", {})
        gamma = params.get("gamma", 0.80)
        epsilonTrain = params.get("epsilonTrain", 0.5)
        alpha = params.get("alpha", 0.05)
        maxMemorySize = params.get("maxMemorySize", 5000)
        batch_size = params.get("batchSize", 64)
        action_space_size = params.get("actionSpaceSize", 5)
        training_size = params.get("trainingSize", 1000)
        epsilon = params.get("threshold", 0.1)

        # dataset
        dataset_name = 'car'
        pset = PointSet(f'{dataset_name}.txt')
        dim = pset.points[0].dim
        for i in range(len(pset.points)):
            pset.points[i].id = i
        u = Point(dim)
        u.coord = params.get("utility_vector", [0.0, 0.0, 0.0, 0.0])  # 默认值是 [0.0, 0.0, 0.0]
        # normalizing the utility vector
        summ = 0
        for i in range(dim):
            summ += u.coord[i]
        for i in range(dim):
            u.coord[i] = u.coord[i] / summ
        await highRL_Train.highRL_Train(pset, u, epsilon, dataset_name, True, training_size, action_space_size, gamma, epsilonTrain,
                                        alpha, maxMemorySize, batch_size, websocket, session_id)

    elif data.get("type") == "inference":
        # dataset
        dataset_name = 'car'
        pset = PointSet(f'{dataset_name}.txt')
        dim = pset.points[0].dim
        for i in range(len(pset.points)):
            pset.points[i].id = i
        u = Point(dim)
        await highRL.highRL(pset, u, epsilon, dataset_name, False, trainning_size, action_size, websocket, session_id)

    await websocket.close(code=1000, reason="Task completed")

# 每当有客户端连接时都会调用这个处理函数
async def handler(websocket):
    session_id = str(uuid.uuid4())  # 生成唯一标识符
    print(f"Client connected with session ID: {session_id}")
    try:
        await user_study(websocket, session_id)  # 启动客户端的输入处理
    except websockets.ConnectionClosed:
        print("Connection closed")
    except Exception as e:
        print(f"Error: {e}")


# 启动 WebSocket 服务器
async def main():
    async with websockets.serve(handler, "localhost", 8000):
        print("Server started at ws://localhost:8000")
        await asyncio.Future()  # 保持服务器运行

if __name__ == "__main__":
    asyncio.run(main())
from algorithms.actor_critic_model import ActorCriticModel
import numpy as np
import os

import zstd, json


def read_replay(file):
    with open(file, 'rb') as f:
        data = zstd.loads(f.read())
        data = json.loads(data.decode())
        return data


replay = read_replay('test_file.hlt')
print(f'replay: {replay}')

# X = np.array([[(0.13, 0, 0), (0.03, 0, 0), (0.07, 0, 0)], [(0.03, 0, 0), (0.0, 0.0, 1), (0.06, 0, 0)], [(0.01, 0, 0), (0.07, 0, 0), (0.02, 0, 0)]])
# print(f'shape: {X.shape}')
#
#
# X2 = np.array([X])
# X3 = np.array(X).reshape(-1, len(X), len(X), 3)
#
# print(f'shape: {X2.shape}')
# print(f'shape: {X3.shape}')
# print(f'shape: {X2}')
# print(f'shape: {X3}')
#
# model = ActorCriticModel(radius=1, output_number=5, load_model=False)
# print(f'init')
#
# for i in range(150):
#     a = model.predict(X3)[0]
#     model.store_memory(X, a, 1.0, X)
#
# b = model.predict(np.array([X, X, X]))
# print(f'b {b}')
#
# for i in range(50):
#     model.train()
#     b = model.predict(np.array([X, X, X]))
#     print(f'b {b}')

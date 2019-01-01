from algorithms.actor_critic_model import ActorCriticModel
from algorithms.dql_model import DQLModel
import numpy as np
import os

X = np.array([[(0.13, 0, 0), (0.03, 0, 0), (0.07, 0, 0)], [(0.03, 0, 0), (0.0, 0.0, 1), (0.06, 0, 0)], [(0.01, 0, 0), (0.07, 0, 0), (0.02, 0, 0)]])
print(f'shape: {X.shape}')


X2 = np.array([X])
X3 = np.array(X).reshape(-1, len(X), len(X), 3)

print(f'shape: {X2.shape}')
print(f'shape: {X3.shape}')

model = DQLModel(radius=1, output_number=5, load_model=False)

for i in range(150):
    a = model.predict(X3)[0]
    model.store_memory(X, a, 1.0, X)

b = model.predict(np.array([X, X, X]))
# print(f'b {b}')

for i in range(1):
    model.train()
    # b = model.predict(np.array([X, X, X]))
    # print(f'b {b}')

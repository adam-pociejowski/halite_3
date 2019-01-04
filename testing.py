from algorithms.actor_critic_model import ActorCriticModel
from algorithms.dql_model import DQLModel
from hlt.bot_utils import Utils, Dropout

import numpy as np
import os

from algorithms.supervised_model import SupervisedModel


# model = SupervisedModel(radius=10, output_number=5, model_name='supervised_cnn_phase2')
model = SupervisedModel(radius=16, output_number=5, model_name='cnn')

zeros = np.zeros(model.input_dim)
Y = model.predict_choice_actions(np.array([zeros]))
print(f'Y: {Y}, dim: {model.input_dim}')

# MAP_CONFIG = {
#     32: {'finish': 7, 'stop_produce_ships': 200,
#          'dropouts': [{'turn': 200, 'position': Position(-8, -8)}, {'turn': 250, 'position': Position(8, 8)}]},
#     40: {'finish': 8, 'stop_produce_ships': 210,
#          'dropouts': [{'turn': 200, 'position': Position(-8, -8)}, {'turn': 250, 'position': Position(8, 8)}]},
#     48: {'finish': 9, 'stop_produce_ships': 220,
#          'dropouts': [{'turn': 200, 'position': Position(-8, -8)}, {'turn': 250, 'position': Position(8, 8)}]},
#     56: {'finish': 10, 'stop_produce_ships': 230,
#          'dropouts': [{'turn': 200, 'position': Position(-8, -8)}, {'turn': 250, 'position': Position(8, 8)}]},
#     64: {'finish': 11, 'stop_produce_ships': 240,
#          'dropouts': [{'turn': 200, 'position': Position(-8, -8)}, {'turn': 250, 'position': Position(8, 8)}]},
# }


#
# dropouts.append(Dropout(200, -8, -8))
# dropouts.append(Dropout(200, 8, 8))
dropouts = [Dropout(200, -8, -8), Dropout(200, 8, 8)]





#
# X = np.array([[(0.13, 0, 0), (0.03, 0, 0), (0.07, 0, 0)], [(0.03, 0, 0), (0.0, 0.0, 1), (0.06, 0, 0)], [(0.01, 0, 0), (0.07, 0, 0), (0.02, 0, 0)]])
# print(f'shape: {X.shape}')
#
#
# X2 = np.array([X])
# X3 = np.array(X).reshape(-1, len(X), len(X), 3)
#
# print(f'shape: {X2.shape}')
# print(f'shape: {X3.shape}')
#
# model = DQLModel(radius=1, output_number=5, load_model=False)
#
# for i in range(150):
#     a = model.predict(X3)[0]
#     model.store_memory(X, a, 1.0, X)
#
# b = model.predict(np.array([X, X, X]))
# print(f'b {b}')
#
# for i in range(1):
#     model.train()
    # b = model.predict(np.array([X, X, X]))
    # print(f'b {b}')

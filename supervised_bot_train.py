import hlt
from hlt import constants
from hlt.bot_utils import *
from hlt.positionals import Direction
import logging
import random
import numpy as np
import sys
import os

ship_states = {}
ships_created = 0

radius = 10
episode = 0
turn_limit = 0

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from algorithms.supervised_model import SupervisedModel

model = SupervisedModel(radius=radius, output_number=5)
summary_writer = SummaryWriter()

sys.stderr = stderr

game = hlt.Game()
game.ready("SupervisedBot")

if len(sys.argv) >= 2:
    logging.info(f'Argument List: {str(sys.argv[1])}')
    episode = int(sys.argv[1])
    turn_limit = int(sys.argv[2])

max_map_halite, min_map_halite = Utils.get_max_min_map_halite(game.game_map)
logging.info(f'[MAP_HALITE]: max: {max_map_halite}, min: {min_map_halite}')

DEPOSIT_HALITE_AMOUNT = 600
MIN_TO_COLLECT_HALITE_AMOUNT = 200
STOP_COLLECT_HALITE_AMOUNT = 20

while True:
    command_queue = []
    game.update_frame()
    game_map = game.game_map
    me = game.me

    movement_choices = []
    for ship in me.get_ships():
        surroundings = Utils.get_surroundings_new(ship, game_map, radius, me, max_map_halite, min_map_halite)
        predicted_action = model.predict(np.array([surroundings]))[0]
        predicted_move = Utils.directions[predicted_action]
        command_queue.append(ship.move(predicted_move))

    if game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())
        ships_created += 1

    game.end_turn(command_queue)

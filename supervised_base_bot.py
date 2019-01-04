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
game.ready("SupervisedBaseBot")

if len(sys.argv) >= 2:
    logging.info(f'Argument List: {str(sys.argv[1])}')
    episode = int(sys.argv[1])
    turn_limit = int(sys.argv[2])

max_halite, min_halite = Utils.get_max_min_map_halite(game.game_map)
logging.info(f'[MAP_HALITE]: max: {max_halite}, min: {min_halite}')

DEPOSIT_HALITE_AMOUNT = 600
MIN_TO_COLLECT_HALITE_AMOUNT = 200
STOP_COLLECT_HALITE_AMOUNT = 20

while True:
    command_queue = []
    game.update_frame()
    game_map = game.game_map
    me = game.me
    my_structures_positions = [d.position for d in list(me.get_dropoffs()) + [me.shipyard]]
    my_ships_positions = [s.position for s in list(me.get_ships())]

    movement_choices = []
    for ship in me.get_ships():
        if ship.id not in ship_states:
            ship_states[ship.id] = 'search'

        if ship.position == me.shipyard.position:
            ship_states[ship.id] = 'search'
        elif ship.halite_amount > DEPOSIT_HALITE_AMOUNT:
            ship_states[ship.id] = 'deposit'

        if ship_states[ship.id] == 'search':
            surroundings = Utils.get_surroundings_new(my_structures_positions, my_ships_positions, ship, game_map, radius, max_halite, min_halite)
            predicted_action = model.predict(np.array([surroundings]))[0]
            predicted_move = Utils.directions[predicted_action]
            move = game_map.naive_navigate(ship, ship.position.directional_offset(predicted_move))
        else:
            move = game_map.naive_navigate(ship, me.shipyard.position)

        command_queue.append(ship.move(move))

    if game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())
        ships_created += 1

    if game.turn_number == turn_limit:
        total_halite_collected_norm = ((ships_created * constants.SHIP_COST) + me.halite_amount - 5000) / 1000.0
        step = (episode * turn_limit) + game.turn_number
        summary_writer.add_summary(0.0, total_halite_collected_norm, step, 0.0, 'supervised_base_bot')

    game.end_turn(command_queue)

import hlt
from hlt import constants
from hlt.bot_utils import *
from hlt.positionals import Direction, Position
import logging
import random
import numpy as np

import sys
import os

episode = 0
turn_limit = 0
loadModel = False
train = False
logging.info(sys.argv)

if len(sys.argv) >= 4:
    episode = int(sys.argv[1])
    turn_limit = int(sys.argv[2])
    if int(sys.argv[3]) == 1:
        loadModel = True
    if int(sys.argv[4]) == 1:
        train = True

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from algorithms.actor_critic_model import ActorCriticModel
from algorithms.dql_model import DQLModel

radius = 10
# model = ActorCriticModel(radius=radius, output_number=4, load_model=False)
model = DQLModel(radius=radius, output_number=5, load_model=loadModel, episode=episode)
summary_writer = SummaryWriter()

sys.stderr = stderr

# {"search", "collect", "deposit"}
ship_states = {}
ship_experience = {}
episode_reward = 0
last_summary_turn = 1
ships_created = 0
DEPOSIT_HALITE_AMOUNT = 600
MIN_TO_COLLECT_HALITE_AMOUNT = 200
STOP_COLLECT_HALITE_AMOUNT = 20

game = hlt.Game()
game.ready("ReinforcementBot")


while True:
    command_queue = []
    game.update_frame()
    game_map = game.game_map
    me = game.me

    movement_choices = []
    dropoffs_positions = [d.position for d in list(me.get_dropoffs()) + [me.shipyard]]
    current_positions = [s.position for s in list(me.get_ships())]

    for ship in me.get_ships():
        if ship.id not in ship_states:
            ship_states[ship.id] = 'search'

        if ship.position == me.shipyard.position and ship_states[ship.id] == 'deposit':
            ship_states[ship.id] = 'search'
            logging.info(f'ship: {ship.id}, DEPOSIT SUCCESSFUL')

        surroundings = Utils.get_surroundings(ship, game_map, radius, dropoffs_positions, current_positions)
        if ship.halite_amount > DEPOSIT_HALITE_AMOUNT and ship_states[ship.id] != 'deposit':
            ship_states[ship.id] = 'deposit'
            logging.info(f'ship: {ship.id}, STARTED DEPOSIT')
        elif game_map[ship.position].halite_amount > MIN_TO_COLLECT_HALITE_AMOUNT and ship_states[ship.id] != 'collect':
            ship_states[ship.id] = 'collect'
            logging.info(f'ship: {ship.id}, STARTED COLLECT')
            if ship.id in ship_experience:
                model.store_memory(ship_experience[ship.id]['observation'],
                                   ship_experience[ship.id]['action'],
                                   Utils.norm_halite(game_map[ship.position].halite_amount),
                                   surroundings)
                del ship_experience[ship.id]
        elif ship_states[ship.id] == 'search' or (ship_states[ship.id] == 'collect'
                                                  and game_map[ship.position].halite_amount < STOP_COLLECT_HALITE_AMOUNT):
            ship_states[ship.id] = 'search'
            if ship.id in ship_experience:
                model.store_memory(ship_experience[ship.id]['observation'],
                                   ship_experience[ship.id]['action'],
                                   0.0,
                                   surroundings)

        if ship_states[ship.id] == 'search':
            predicted_action = model.predict(np.array([surroundings]))[0]
            predicted_move = Utils.directions[predicted_action]
            move_choice = game_map.naive_navigate(ship, ship.position.directional_offset(predicted_move))

            ship_experience[ship.id] = {
                'action': Utils.actions[move_choice],
                'observation': surroundings,
                'halite': Utils.norm_halite(ship.halite_amount)
            }
            logging.info(f'ship: {ship.id}, SEARCHING, ML action: {predicted_action}, final action: {Utils.actions[move_choice]}')
        elif ship_states[ship.id] == 'deposit':
            move_choice = game_map.naive_navigate(ship, me.shipyard.position)
            logging.info(f'ship: {ship.id}, DEPOSITING, {ship.halite_amount}')
        else:
            move_choice = Direction.Still
            logging.info(f'ship: {ship.id}, COLLECTING, {ship.halite_amount}, CELL_HALITE: {game_map[ship.position].halite_amount}')

        command_queue.append(ship.move(move_choice))

    if game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())
        ships_created += 1

    if train:
        model.train()
        if game.turn_number % 20 == 0:
            model.save()
        if game.turn_number == turn_limit:
            total_halite_collected_norm = ((ships_created * constants.SHIP_COST) + me.halite_amount - 5000) / 1000.0
            step = (episode * turn_limit) + game.turn_number
            summary_writer.add_summary(episode_reward, total_halite_collected_norm, step, model.epsilon, model.model_name)

    game.end_turn(command_queue)


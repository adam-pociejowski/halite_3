import hlt
from hlt import constants
from hlt.bot_utils import Utils
from hlt.positionals import Direction, Position
import logging
import random
import numpy as np

import sys
import os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from algorithms.actor_critic_model import ActorCriticModel

radius = 16
model = ActorCriticModel(radius=radius, output_number=4, load_model=True)

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
game.ready("ActorCriticBot2")

episode = 0
turn_limit = 0
if len(sys.argv) >= 2:
    logging.info(f'Argument List: {str(sys.argv[1])}')
    episode = int(sys.argv[1])
    turn_limit = int(sys.argv[2])


with tf.name_scope('Summaries'):
    mean_reward_placeholder = tf.placeholder(tf.float32, shape=None, name='mean_reward')
    mean_reward = tf.summary.scalar('mean_reward', mean_reward_placeholder)

    halite_placeholder = tf.placeholder(tf.float32, shape=None, name='halite_value')
    halite_value = tf.summary.scalar('halite_value', halite_placeholder)

    merged = tf.summary.merge_all()


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
                logging.info(f"COLLECTING: reward: {Utils.norm_halite(game_map[ship.position].halite_amount)}, "
                             f"action: {ship_experience[ship.id]['action']}")
                del ship_experience[ship.id]
        elif ship_states[ship.id] == 'search' or (ship_states[ship.id] == 'collect'
                                                  and game_map[ship.position].halite_amount < STOP_COLLECT_HALITE_AMOUNT):
            ship_states[ship.id] = 'search'
            if ship.id in ship_experience:
                model.store_memory(ship_experience[ship.id]['observation'],
                                   ship_experience[ship.id]['action'],
                                   0.0,
                                   surroundings)
                logging.info(f"SEARCHING: reward: 0.0, action: {ship_experience[ship.id]['action']}")

        if ship_states[ship.id] == 'search':
            if random.random() > 0.8:
                predicted_action = np.random.randint(4)
            else:
                predicted_action = model.predict(np.array([surroundings]))[0]
            ship_experience[ship.id] = {
                'action': predicted_action,
                'observation': surroundings,
                'halite': Utils.norm_halite(ship.halite_amount)
            }
            move_choice = Utils.directions[predicted_action]
            logging.info(f'ship: {ship.id}, SEARCHING, {ship.halite_amount}')
        elif ship_states[ship.id] == 'deposit':
            move_choice = game_map.naive_navigate(ship, me.shipyard.position)
            logging.info(f'ship: {ship.id}, DEPOSITING, {ship.halite_amount}')
        else:
            move_choice = Direction.Still
            logging.info(f'ship: {ship.id}, COLLECTING, {ship.halite_amount}, CELL_HALITE: {game_map[ship.position].halite_amount}')

        command_queue.append(ship.move(move_choice))

    if turn_limit > 0:
        model.train()
        if game.turn_number % 20 == 0:
            model.save()
            if game.turn_number == turn_limit:
                sess = tf.Session()
                summary_writer = tf.summary.FileWriter(f'summary/{model.model_name}')

                total_halite_collected = (ships_created * constants.SHIP_COST) + me.halite_amount - 5000
                summary, _, _ = sess.run([merged, mean_reward, halite_value], feed_dict={mean_reward_placeholder: episode_reward,
                                                                                         halite_placeholder: total_halite_collected / 1000.0})
                summary_writer.add_summary(summary, (episode * turn_limit) + game.turn_number)

    if game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        if len(me.get_ships()) == 0:
            command_queue.append(me.shipyard.spawn())
            ships_created += 1

    game.end_turn(command_queue)


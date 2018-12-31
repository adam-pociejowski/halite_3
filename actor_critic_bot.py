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

# model = ActorCriticModel(radius=radius, output_number=5, load_model=False)
model = ActorCriticModel(radius=radius, output_number=5, load_model=True)

sys.stderr = stderr

# {"alive", "dead"}
ship_states = {}
ship_experience = {}
episode_reward = 0
last_summary_turn = 1

game = hlt.Game()
game.ready("ActorCriticBot")

episode = 0
turn_limit = 0
if len(sys.argv) >= 2:
    logging.info(f'Argument List: {str(sys.argv[1])}')
    episode = int(sys.argv[1])
    turn_limit = int(sys.argv[2])


def set_reward(ship_id, reward, type):
    logging.info(f'ship: {ship_id}, reward: {reward}, type: {type}')
    ship_experience[ship_id]['reward'] = reward
    return reward


with tf.name_scope('Summaries'):
    mean_reward_placeholder = tf.placeholder(tf.float32, shape=None, name='mean_reward')
    mean_reward = tf.summary.scalar('mean_reward', mean_reward_placeholder)
    merged = tf.summary.merge_all()


while True:
    command_queue = []
    game.update_frame()
    game_map = game.game_map
    me = game.me

    movement_choices = []
    dropoffs_positions = [d.position for d in list(me.get_dropoffs()) + [me.shipyard]]
    current_positions = [s.position for s in list(me.get_ships())]
    current_ships = [s.id for s in list(me.get_ships())]

    for key in ship_states:
        if key not in current_ships and ship_states[key] == 'alive':
            ship_states[key] = 'dead'
            # episode_reward += set_reward(key, -1.0, 'COLLISION')
            # model.store_memory(ship_experience[key]['observation'],
            #                    ship_experience[key]['action'],
            #                    ship_experience[key]['reward'],
            #                    ship_experience[key]['observation'])
            del ship_experience[key]

    for ship in me.get_ships():
        if ship.id not in ship_states:
            ship_states[ship.id] = 'alive'

        surroundings = Utils.get_surroundings(ship, game_map, radius)
        if ship.id in ship_experience:
            if ship.position == me.shipyard.position:
                logging.info(ship.halite_amount)
                episode_reward += set_reward(ship.id, ship_experience[ship.id]['halite'] / constants.MAX_HALITE, 'DEPOSIT')

            model.store_memory(ship_experience[ship.id]['observation'],
                               ship_experience[ship.id]['action'],
                               ship_experience[ship.id]['reward'],
                               surroundings)

        predicted_action = model.predict(np.array([surroundings]))[0]
        command_queue.append(ship.move(Utils.directions[predicted_action]))
        ship_experience[ship.id] = {
            'action': predicted_action,
            'observation': surroundings,
            'reward': 0.0,
            'halite': ship.halite_amount
        }

    if turn_limit > 0:
        model.train()
        if game.turn_number % 20 == 0:
            model.save()
            if game.turn_number == turn_limit:
                sess = tf.Session()
                summary_writer = tf.summary.FileWriter(f'summary/{model.model_name}')
                logging.info(f'Episode {episode} reward: {episode_reward}')
                summary, _ = sess.run([merged, mean_reward], feed_dict={mean_reward_placeholder: episode_reward})
                summary_writer.add_summary(summary, (episode * turn_limit) + game.turn_number)
                last_summary_turn = game.turn_number
                episode_reward = 0

    if game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    game.end_turn(command_queue)


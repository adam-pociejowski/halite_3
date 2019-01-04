import hlt
from hlt import constants
from hlt.positionals import Direction, Position
import tensorflow as tf
import numpy as np
import logging


class SummaryWriter:

    def __init__(self):
        with tf.name_scope('Summaries'):
            self.mean_reward_placeholder = tf.placeholder(tf.float32, shape=None, name='ships')
            self.mean_reward = tf.summary.scalar('ships', self.mean_reward_placeholder)

            self.halite_placeholder = tf.placeholder(tf.float32, shape=None, name='halite_value')
            self.halite_value = tf.summary.scalar('halite_value', self.halite_placeholder)

            self.randomness_placeholder = tf.placeholder(tf.float32, shape=None, name='game_size')
            self.randomness = tf.summary.scalar('game_size', self.randomness_placeholder)

            self.merged = tf.summary.merge_all()

    def add_summary(self, ships, halite, step, game_size, model_name):
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(f'summary/{model_name}')
        summary, _, _, _ = sess.run([self.merged, self.mean_reward, self.halite_value, self.randomness],
                                    feed_dict={self.mean_reward_placeholder: ships,
                                               self.halite_placeholder: halite,
                                               self.randomness_placeholder: game_size})
        summary_writer.add_summary(summary, step)


class Dropout:
    def __init__(self, turn, x, y):
        self.turn = turn
        self.position = Position(x, y, normalize=False)


class Utils:
    directions = [Direction.North, Direction.South, Direction.East, Direction.West, Direction.Still]
    actions = {Direction.North: 0, Direction.South: 1, Direction.East: 2, Direction.West: 3, Direction.Still: 4}
    finish_dict = {32: 7, 40: 8, 48: 9, 56: 10, 64: 11}
    MAP_CONFIG = {
        2: {
            32: {'finish': 5, 'stop_produce_ships': 250, 'dropouts': [Dropout(200, 0, -8), Dropout(250, 0, 8)]},
            40: {'finish': 6, 'stop_produce_ships': 260, 'dropouts': [Dropout(200, 0, -9), Dropout(250, 0, 9)]},
            48: {'finish': 7, 'stop_produce_ships': 270, 'dropouts': [Dropout(200, 0, -10), Dropout(250, 0, 10)]},
            56: {'finish': 8, 'stop_produce_ships': 280, 'dropouts': [Dropout(200, 0, -11), Dropout(250, 0, 11)]},
            64: {'finish': 9, 'stop_produce_ships': 290, 'dropouts': [Dropout(200, 0, -12), Dropout(250, 0, 12)]},
        },
        4: {
            32: {'finish': 7, 'stop_produce_ships': 200, 'dropouts': [Dropout(200, -8, -8), Dropout(250, 8, 8)]},
            40: {'finish': 8, 'stop_produce_ships': 210, 'dropouts': [Dropout(200, -8, -8), Dropout(250, 8, 8)]},
            48: {'finish': 9, 'stop_produce_ships': 220, 'dropouts': [Dropout(200, -8, -8), Dropout(250, 8, 8)]},
            56: {'finish': 10, 'stop_produce_ships': 230, 'dropouts': [Dropout(200, -8, -8), Dropout(250, 8, 8)]},
            64: {'finish': 11, 'stop_produce_ships': 240, 'dropouts': [Dropout(200, -8, -8), Dropout(250, 8, 8)]},
        }
    }

    SHIPS_LIMIT = 30
    DEPOSIT_HALITE_AMOUNT = 600
    MIN_TO_COLLECT_HALITE_AMOUNT = 200
    STOP_COLLECT_HALITE_AMOUNT = 20

    @staticmethod
    def can_make_ship(turn_number, me, game_map, config):
        return turn_number <= config['stop_produce_ships'] and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied and \
               len(me.get_ships()) < Utils.SHIPS_LIMIT

    @staticmethod
    def find_closest_structure(game_map, ship, my_structures_positions):
        closest_structure_pos = None
        closest_distance = 1000
        for pos in my_structures_positions:
            distance = game_map.calculate_distance(ship.position, pos)
            if closest_distance > distance:
                closest_structure_pos = pos
                closest_distance = distance
        return closest_structure_pos

    @staticmethod
    def find_closest_ship(game_map, ships, location):
        closest_ship = None
        closest_ship_dist = 1000
        for ship in ships:
            distance = game_map.calculate_distance(ship.position, location)
            if closest_ship_dist > distance:
                closest_ship = ship
                closest_ship_dist = distance
        return closest_ship

    @staticmethod
    def select_ship_to_make_dropout(game_map, building_dropout, ship_states, ships, shipyard_position, dropout):
        dropout_position = shipyard_position + dropout.position
        ship = Utils.find_closest_ship(game_map, ships, dropout_position)
        if ship is not None:
            ship_states[ship.id] = 'B'
            building_dropout = {'ship_id': ship.id, 'dropout_position': dropout_position}
            logging.info(f'[CREATE_DROPOUT]: ship: {ship.id}, building_dropout: {building_dropout}')
        return building_dropout, ship_states

    @staticmethod
    def make_dropout(game_map, ships, ship_states, config, game_turn, building_dropout, shipyard_position):
        dropouts_to_make = config['dropouts']
        if building_dropout is not None:
            if building_dropout['ship_id'] not in ship_states:
                building_dropout, ship_states = Utils.select_ship_to_make_dropout(game_map, building_dropout, ship_states, ships, shipyard_position, dropouts_to_make[0])
        elif len(dropouts_to_make) > 0 and dropouts_to_make[0].turn <= game_turn:
            building_dropout, ship_states = Utils.select_ship_to_make_dropout(game_map, building_dropout, ship_states, ships, shipyard_position, dropouts_to_make[0])
        return ship_states, config, building_dropout

    @staticmethod
    def update_ship_states(ships, ship_states):
        new_ship_states = {}
        my_ships_positions = []
        for ship in ships:
            my_ships_positions.append(ship.position)
            if ship.id in ship_states:
                new_ship_states[ship.id] = ship_states[ship.id]
            else:
                new_ship_states[ship.id] = 'S'

        return new_ship_states, my_ships_positions

    @staticmethod
    def norm_halite(halite_amount):
        return round(halite_amount / constants.MAX_HALITE, 2)

    @staticmethod
    def get_max_halite_move(halite_dict):
        if not halite_dict:
            return Direction.Still

        return max(halite_dict, key=halite_dict.get)

    @staticmethod
    def make_surrounding_dict(ship, game_map, movement_choices, current_positions):
        # {(0,1): (12,23)}
        position_dict = {}
        # {(0,1): 500}
        halite_dict = {}
        ship_positions = ship.position.get_surrounding_cardinals() + [ship.position]
        for n, direction in enumerate([Direction.North, Direction.South, Direction.East, Direction.West, Direction.Still]):
            position_dict[direction] = ship_positions[n]
            if position_dict[direction] not in movement_choices and position_dict[direction] not in current_positions:
                halite_amount = game_map[position_dict[direction]].halite_amount
                halite_dict[direction] = halite_amount

        return position_dict, halite_dict

    @staticmethod
    def get_surroundings(ship, game_map, radius, dropoffs_positions, current_positions):
        _surroundings = []
        for y in range(-1 * radius, radius + 1):
            row = []
            for x in range(-1 * radius, radius + 1):
                current_cell = game_map[ship.position + Position(x, y)]
                if current_cell.position in dropoffs_positions:
                    drop_friend_foe = 1
                else:
                    drop_friend_foe = -1

                if current_cell.position in current_positions:
                    ship_friend_foe = 1
                else:
                    ship_friend_foe = -1

                halite = round(current_cell.halite_amount / constants.MAX_HALITE, 2)
                cell_ship = current_cell.ship
                structure = current_cell.structure

                if halite is None:
                    halite = 0

                if cell_ship is None:
                    cell_ship = 0
                else:
                    cell_ship = ship_friend_foe * round(cell_ship.halite_amount / constants.MAX_HALITE * 0.8, 2) + 0.2

                if structure is None:
                    structure = 0
                else:
                    structure = drop_friend_foe
                row.append((halite, cell_ship, structure))
            _surroundings.append(row)
        return _surroundings

    @staticmethod
    def get_max_min_map_halite(game_map):
        max_halite = 0
        min_halite = constants.MAX_HALITE
        for x in range(game_map.width):
            for y in range(game_map.height):
                halite_amount = game_map[Position(x, y)].halite_amount
                if max_halite < halite_amount:
                    max_halite = halite_amount
                if min_halite > halite_amount:
                    min_halite = halite_amount

        return max_halite, min_halite

    @staticmethod
    def get_surroundings_new(friendly_structures_positions, friendly_ships_positions, ship, game_map, radius, max_map_halite, min_map_halite):
        _surroundings = []
        for y in range(-1 * radius, radius + 1):
            row = []
            for x in range(-1 * radius, radius + 1):
                current_cell = game_map[ship.position + Position(x, y)]
                halite = (current_cell.halite_amount - min_map_halite) / (max_map_halite - min_map_halite)
                if current_cell.ship is None:
                    cell_ship = 0
                elif current_cell.position in friendly_ships_positions:
                    cell_ship = 1
                else:
                    cell_ship = -1

                if current_cell.structure is None:
                    structure = 0
                elif current_cell.position in friendly_structures_positions:
                    structure = 1
                else:
                    structure = -1

                row.append((halite, cell_ship, structure, 0.0))
            _surroundings.append(row)
        return _surroundings

    @staticmethod
    def get_best_move(predicted_actions, game_map, ship):
        for i in range(len(predicted_actions)):
            predicted_action = predicted_actions[i]
            predicted_move = Utils.directions[predicted_action]
            move = game_map.naive_navigate(ship, ship.position.directional_offset(predicted_move))
            if move == predicted_move:
                return move

        return Direction.Still

    @staticmethod
    def choice_actions(probabilities):
        action_choices = []
        actions_indexes = np.arange(probabilities.shape[1], dtype=np.int32)

        for i in range(probabilities.shape[1]):
            action = [np.random.choice(probabilities.shape[1], 1, p=probabilities[i])[0] for i in range(probabilities.shape[0])][0]
            probabilities = np.delete(probabilities, action, 1)
            probabilities = probabilities / np.sum(probabilities)
            action_choices.append(actions_indexes[action])
            actions_indexes = np.delete(actions_indexes, action)

        return action_choices

    @staticmethod
    def navigate_to_finish_game(friendly_structures_positions, game_map, ship, destination):
        for direction in game_map.get_unsafe_moves(ship.position, destination):
            target_pos = ship.position.directional_offset(direction)
            if not game_map[target_pos].is_occupied or target_pos in friendly_structures_positions:
                game_map[target_pos].mark_unsafe(ship)
                return direction

        return Direction.Still

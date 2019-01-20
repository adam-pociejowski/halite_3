import hlt
from hlt import constants
from hlt.positionals import Direction, Position
import tensorflow as tf
import numpy as np
import logging
import random


class SummaryWriter:

    def __init__(self):
        with tf.name_scope('Summaries'):
            self.ships_at_the_end_placeholder = tf.placeholder(tf.float32, shape=None, name='ships_at_the_end')
            self.ships_at_the_end = tf.summary.scalar('ships_at_the_end', self.ships_at_the_end_placeholder)

            self.halite_placeholder = tf.placeholder(tf.float32, shape=None, name='halite_value')
            self.halite_value = tf.summary.scalar('halite_value', self.halite_placeholder)

            self.game_size_placeholder = tf.placeholder(tf.float32, shape=None, name='game_size')
            self.game_size = tf.summary.scalar('game_size', self.game_size_placeholder)

            self.ships_produced_placeholder = tf.placeholder(tf.float32, shape=None, name='ships_produced')
            self.ships_produced = tf.summary.scalar('ships_produced', self.ships_produced_placeholder)

            self.merged = tf.summary.merge_all()

    def add_summary(self, ships, halite, step, game_size, ships_produces, model_name):
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(f'summary/{model_name}')
        summary, _, _, _, _ = sess.run([self.merged, self.ships_at_the_end, self.halite_value, self.game_size, self.ships_produced],
                                       feed_dict={self.ships_at_the_end_placeholder: ships,
                                                  self.halite_placeholder: halite,
                                                  self.game_size_placeholder: game_size,
                                                  self.ships_produced_placeholder: ships_produces})
        summary_writer.add_summary(summary, step)


class Dropout:
    def __init__(self, turn, x, y, halite=0, search_best_place_radius=4):
        self.turn = turn
        self.center = Position(x, y, normalize=False)
        self.position = self.center
        self.halite = halite
        self.search_best_place_radius = search_best_place_radius


class Utils:
    DIRECTIONS = [Direction.North, Direction.South, Direction.East, Direction.West, Direction.Still]
    MAP_CONFIG = {
        2: {
            32: {'finish': 5, 'stop_produce_ships': 260, 'dropouts': [Dropout(140, -6, -7), Dropout(195, -6, 7)]},
            40: {'finish': 6, 'stop_produce_ships': 270, 'dropouts': [Dropout(140, -6, -8), Dropout(195, -6, 8)]},
            48: {'finish': 7, 'stop_produce_ships': 280, 'dropouts': [Dropout(140, -7, -9), Dropout(190, -7, 9)]},
            56: {'finish': 8, 'stop_produce_ships': 290, 'dropouts': [Dropout(135, -7, -10), Dropout(185, -7, 10)]},
            64: {'finish': 9, 'stop_produce_ships': 300, 'dropouts': [Dropout(130, -8, -11), Dropout(180, -8, 11)]},
        },
        4: {
            32: {'finish': 7, 'stop_produce_ships': 200, 'dropouts': [Dropout(140, -4, -4)]},
            40: {'finish': 8, 'stop_produce_ships': 210, 'dropouts': [Dropout(140, -7, 2), Dropout(195, 2, -7)]},
            48: {'finish': 9, 'stop_produce_ships': 220, 'dropouts': [Dropout(140, -8, 2), Dropout(190, 2, -8)]},
            56: {'finish': 10, 'stop_produce_ships': 230, 'dropouts': [Dropout(135, -9, 2), Dropout(185, 2, -9)]},
            64: {'finish': 11, 'stop_produce_ships': 240, 'dropouts': [Dropout(130, -10, 2), Dropout(180, 2, -10)]},
        }
    }
    DROPOUT_MOD = {
        2: {
            0: {'x': 1, 'y': 1},
            1: {'x': -1, 'y': -1}
        },
        4: {
            0: {'x': 1, 'y': 1},
            1: {'x': -1, 'y': 1},
            2: {'x': 1, 'y': -1},
            3: {'x': -1, 'y': -1}
        }
    }

    ########### DROPOUT
    @staticmethod
    def find_best_place_for_dropout(game_map, dropout):
        radius = dropout.search_best_place_radius
        logging.info(f'DROPOUT CENTER: {dropout.center}')
        for i in range(dropout.center.x - radius, dropout.center.x + radius + 1):
            for j in range(dropout.center.y - radius, dropout.center.y + radius + 1):
                position = game_map.normalize(Position(i, j))
                if game_map[position].halite_amount > dropout.halite:
                    dropout.halite = game_map[position].halite_amount
                    dropout.position = position
        logging.info(f'CHOSEN DROPOUT: {dropout.position} {dropout.halite}')

        return dropout

    @staticmethod
    def select_ship_to_make_dropout(game_map, data, ships, dropout_to_make):
        dropout_to_make = Utils.find_best_place_for_dropout(game_map, dropout_to_make)
        dropout_position = dropout_to_make.position
        ship = Utils.find_closest_ship(game_map, ships, dropout_position)
        if ship is not None:
            data['states'][ship.id] = 'B'
            data['dropout'] = {'ship_id': ship.id, 'dropout_position': dropout_position}
        return data

    @staticmethod
    def make_dropout(game_map, ships, data, config, game_turn):
        dropouts_to_make = config['dropouts']
        if data['dropout'] is not None:
            if data['dropout']['ship_id'] not in data['states']:
                data = Utils.select_ship_to_make_dropout(game_map, data, ships, dropouts_to_make[0])
        elif len(dropouts_to_make) > 0 and dropouts_to_make[0].turn <= game_turn:
            data = Utils.select_ship_to_make_dropout(game_map, data, ships, dropouts_to_make[0])
        return data, config

    @staticmethod
    def has_halite_to_make_dropout(game, ship):
        return game.me.halite_amount >= Utils.get_cost_to_make_dropout(game, ship)

    @staticmethod
    def get_cost_to_make_dropout(game, ship):
        return constants.DROPOFF_COST - ship.halite_amount - game.game_map[ship.position].halite_amount

    @staticmethod
    def can_make_ship(turn_number, me, game_map, config):
        return turn_number <= config['stop_produce_ships'] and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied

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
    def update_ship_states(ships, data):
        new_ship_states = {}
        data['ships'] = []
        for ship in ships:
            data['ships'].append(ship.position)
            if ship.id in data['states']:
                new_ship_states[ship.id] = data['states'][ship.id]
            else:
                new_ship_states[ship.id] = 'S'
                
        data['states'] = new_ship_states
        return data

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
        position_dict = {}
        halite_dict = {}
        ship_positions = ship.position.get_surrounding_cardinals() + [ship.position]
        for n, direction in enumerate([Direction.North, Direction.South, Direction.East, Direction.West, Direction.Still]):
            position_dict[direction] = ship_positions[n]
            if position_dict[direction] not in movement_choices and position_dict[direction] not in current_positions:
                halite_amount = game_map[position_dict[direction]].halite_amount
                halite_dict[direction] = halite_amount

        return position_dict, halite_dict

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
    def get_surroundings(positions, ship, game_map, radius, max_map_halite, min_map_halite):
        _surroundings = []
        for y in range(-1 * radius, radius + 1):
            row = []
            for x in range(-1 * radius, radius + 1):
                current_cell = game_map[ship.position + Position(x, y)]
                halite = (current_cell.halite_amount - min_map_halite) / (max_map_halite - min_map_halite)
                if current_cell.ship is None:
                    cell_ship = 0
                elif current_cell.position in positions['ships']:
                    cell_ship = 1
                else:
                    cell_ship = -1

                if current_cell.structure is None:
                    structure = 0
                elif current_cell.position in positions['structures']:
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
            predicted_move = Utils.DIRECTIONS[predicted_action]
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
    def navigate(game_map, ship, destination, data):
        for direction in game_map.get_unsafe_moves(ship.position, destination):
            target_pos = ship.position.directional_offset(direction)
            cell_ship = game_map[target_pos].ship
            if cell_ship is None or (target_pos in data['structures'] and cell_ship.id not in data['states'].keys()):
                game_map[target_pos].mark_unsafe(ship)
                return direction

        return Direction.Still

    @staticmethod
    def navigate_to_finish_game(friendly_structures_positions, game_map, ship, destination):
        for direction in game_map.get_unsafe_moves(ship.position, destination):
            target_pos = ship.position.directional_offset(direction)
            if not game_map[target_pos].is_occupied or target_pos in friendly_structures_positions:
                game_map[target_pos].mark_unsafe(ship)
                return direction

        return Direction.Still

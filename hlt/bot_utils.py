import hlt
from hlt import constants
from hlt.positionals import Direction, Position
import tensorflow as tf
import logging


class SummaryWriter:

    def __init__(self):
        with tf.name_scope('Summaries'):
            self.mean_reward_placeholder = tf.placeholder(tf.float32, shape=None, name='mean_reward')
            self.mean_reward = tf.summary.scalar('mean_reward', self.mean_reward_placeholder)

            self.halite_placeholder = tf.placeholder(tf.float32, shape=None, name='halite_value')
            self.halite_value = tf.summary.scalar('halite_value', self.halite_placeholder)

            self.randomness_placeholder = tf.placeholder(tf.float32, shape=None, name='randomness')
            self.randomness = tf.summary.scalar('randomness', self.randomness_placeholder)

            self.merged = tf.summary.merge_all()

    def add_summary(self, reward, halite, step, randomness, model_name):
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(f'summary/{model_name}')
        summary, _, _, _ = sess.run([self.merged, self.mean_reward, self.halite_value, self.randomness],
                                     feed_dict={self.mean_reward_placeholder: reward,
                                                self.halite_placeholder: halite,
                                                self.randomness_placeholder: randomness})
        summary_writer.add_summary(summary, step)


class Utils:
    directions = [Direction.North, Direction.South, Direction.East, Direction.West, Direction.Still]
    actions = {Direction.North: 0, Direction.South: 1, Direction.East: 2, Direction.West: 3, Direction.Still: 4}

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
    def get_surroundings_new(ship, game_map, radius, me, max_map_halite, min_map_halite):
        friendly_structures_positions = [d.position for d in list(me.get_dropoffs()) + [me.shipyard]]
        friendly_ships_positions = [s.position for s in list(me.get_ships())]
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

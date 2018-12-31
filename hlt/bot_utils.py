import hlt
from hlt import constants
from hlt.positionals import Direction, Position
import tensorflow as tf


class Utils:
    directions = [Direction.North, Direction.South, Direction.East, Direction.West, Direction.Still]

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

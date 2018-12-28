import hlt
from hlt import constants
from hlt.positionals import Direction


class Utils:

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

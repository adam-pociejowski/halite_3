import hlt
from hlt import constants
from hlt.bot_utils import Utils
from hlt.positionals import Direction
import logging
import random

game = hlt.Game()
game.ready("MyPythonBot")
ship_states = {}


def _get_best_safe_move(_ship, _movement_choices):
    safe_halite_dict = {}
    for key in halite_dict:
        if position_dict[key] not in _movement_choices:
            safe_halite_dict[key] = halite_dict[key]

    return Utils.get_max_halite_move(halite_dict)


def _make_move(_ship, _movement_choices, move_choice):
    if position_dict[move_choice] in _movement_choices:
        move_choice = _get_best_safe_move(_ship, _movement_choices)

    command_queue.append(_ship.move(move_choice))
    _movement_choices.append(position_dict[move_choice])
    return _movement_choices


def make_move(_ship, _game_map, _movement_choices, _position_dict):
    move_choice = Direction.Still
    if ship_states[ship.id] == 'collect':
        if _ship.is_full:
            ship_states[ship.id] = 'deposit'
            move_choice = _game_map.naive_navigate(_ship, me.shipyard.position)
        elif _game_map[_ship.position].halite_amount < constants.MAX_HALITE / 10:
            move_choice = Utils.get_max_halite_move(halite_dict)
    elif _ship.position == me.shipyard.position:
        ship_states[ship.id] = 'collect'
        move_choice = Utils.get_max_halite_move(halite_dict)
    elif ship_states[ship.id] == 'deposit':
        move_choice = _game_map.naive_navigate(_ship, me.shipyard.position)

    return _make_move(_ship, _movement_choices, move_choice)


while True:
    command_queue = []
    game.update_frame()
    game_map = game.game_map
    me = game.me

    movement_choices = []
    current_positions = []
    for ship in me.get_ships():
        current_positions.append(ship.position)

    for ship in me.get_ships():
        if ship.id not in ship_states:
            ship_states[ship.id] = 'collect'

        position_dict, halite_dict = Utils.make_surrounding_dict(ship, game_map, movement_choices, current_positions)
        movement_choices = make_move(ship, game_map, movement_choices, position_dict)

    if game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    game.end_turn(command_queue)


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

    logging.info(f'_get_best_safe_move {max(halite_dict, key=halite_dict.get)}')
    return max(halite_dict, key=halite_dict.get)


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
            logging.info(f'ship: {ship.id} is full!')
        elif _game_map[_ship.position].halite_amount < constants.MAX_HALITE / 10:
            move_choice = max(halite_dict, key=halite_dict.get)
    elif _ship.position == me.shipyard.position:
        ship_states[ship.id] = 'collect'
        move_choice = max(halite_dict, key=halite_dict.get)
        logging.info(f'ship: {ship.id} is successfully deposited!')
    elif ship_states[ship.id] == 'deposit':
        move_choice = _game_map.naive_navigate(_ship, me.shipyard.position)

    return _make_move(_ship, _movement_choices, move_choice)


while True:
    command_queue = []
    game.update_frame()
    game_map = game.game_map
    me = game.me

    movement_choices = []
    for ship in me.get_ships():
        if ship.id not in ship_states:
            ship_states[ship.id] = 'collect'

        position_dict, halite_dict = Utils.make_surrounding_dict(ship, game_map, movement_choices)
        movement_choices = make_move(ship, game_map, movement_choices, position_dict)

    logging.info(f'movement_choices: {movement_choices}, step: {game.turn_number}, ships: {len(me.get_ships())}')

    if game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    game.end_turn(command_queue)


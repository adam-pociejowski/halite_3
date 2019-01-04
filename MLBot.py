import hlt
from hlt import constants
from hlt.bot_utils import Utils, SummaryWriter
from hlt.positionals import Direction
import numpy as np
import sys
import os
import logging
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from algorithms.supervised_model import SupervisedModel


class MLBot:
    def __init__(self, episode=0, deposit_halite_amount=800, radius=10, summary_name=None, model_name='supervised_cnn_phase2'):
        self.radius = radius
        self.model = SupervisedModel(radius=self.radius, output_number=5, model_name=model_name)
        self.deposit_halite_amount = deposit_halite_amount
        self.summary_name = summary_name
        self.episode = episode
        if summary_name is not None:
            self.summary_writer = SummaryWriter()
        sys.stderr = stderr

        self.game = hlt.Game()
        self.config = Utils.MAP_CONFIG[len(self.game.players)][self.game.game_map.height]
        self.game.ready("SupervisedBot")

    def run(self):
        # F - finish, S - search, D - deposit, B - build dropout
        ship_states = {}
        dropout = None
        ships_created = 0
        turn_limit = constants.MAX_TURNS
        max_halite, min_halite = Utils.get_max_min_map_halite(self.game.game_map)
        dropout_build_in_this_turn = False

        while True:
            self.game.update_frame()
            game_map = self.game.game_map
            turn_number = self.game.turn_number
            me = self.game.me
            command_queue = []
            my_structures_positions = [d.position for d in list(me.get_dropoffs()) + [me.shipyard]]
            ship_states, my_ships_positions = Utils.update_ship_states(me.get_ships(), ship_states)
            ship_states, config, dropout = Utils.make_dropout(game_map, me.get_ships(), ship_states, self.config, turn_number, dropout, game_map[me.shipyard].position)

            for ship in me.get_ships():
                if ship_states[ship.id] == 'F' or turn_limit - turn_number - self.config['finish'] < \
                        game_map.calculate_distance(ship.position, me.shipyard.position):
                    ship_states[ship.id] = 'F'
                elif ship.position in my_structures_positions and ship_states[ship.id] != 'B':
                    ship_states[ship.id] = 'S'
                elif ship.halite_amount > self.deposit_halite_amount and ship_states[ship.id] != 'B':
                    ship_states[ship.id] = 'D'

                if ship_states[ship.id] == 'F':
                    move = Utils.navigate_to_finish_game(my_structures_positions, game_map, ship, Utils.find_closest_structure(game_map, ship, my_structures_positions))
                elif ship_states[ship.id] == 'B':
                    if ship.position == dropout['dropout_position'] and me.halite_amount >= constants.DROPOFF_COST:
                        command_queue.append(ship.make_dropoff())
                        ship_states[ship.id] = 'S'
                        dropout = None
                        del config['dropouts'][0]
                        dropout_build_in_this_turn = True
                        continue
                    else:
                        move = game_map.naive_navigate(ship, dropout['dropout_position'])
                elif ship_states[ship.id] == 'S':
                    surroundings = Utils.get_surroundings_new(my_structures_positions, my_ships_positions, ship, game_map, self.radius, max_halite, min_halite)
                    predicted_actions = self.model.predict_choice_actions(np.array([surroundings]))
                    move = Utils.get_best_move(predicted_actions, game_map, ship)
                else:
                    move = game_map.naive_navigate(ship, Utils.find_closest_structure(game_map, ship, my_structures_positions))

                command_queue.append(ship.move(move))

            if (dropout is None or (dropout_build_in_this_turn and me.halite_amount >= constants.SHIP_COST + constants.DROPOFF_COST)) \
                    and Utils.can_make_ship(turn_number, me, game_map, config):
                command_queue.append(me.shipyard.spawn())
                ships_created += 1

            if self.summary_name is not None and turn_number == turn_limit:
                total_halite_collected_norm = ((ships_created * constants.SHIP_COST) + me.halite_amount - 5000) / 1000.0
                self.summary_writer.add_summary(len(me.get_ships()), total_halite_collected_norm, self.episode, game_map.height, self.summary_name)

            self.game.end_turn(command_queue)

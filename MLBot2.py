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


class MLBot2:
    def __init__(self, episode=0, deposit_halite_amount=800, radius=16, summary_name=None, model_name='supervised_cnn_phase2'):
        self.radius = radius
        self.model = SupervisedModel(radius=self.radius, output_number=5, model_name=model_name)
        self.deposit_halite_amount = deposit_halite_amount
        self.summary_name = summary_name
        self.episode = episode
        if summary_name is not None:
            self.summary_writer = SummaryWriter()
        sys.stderr = stderr

        self.game = hlt.Game()
        self.max_halite, self.min_halite = Utils.get_max_min_map_halite(self.game.game_map)
        self.config = Utils.MAP_CONFIG[len(self.game.players)][self.game.game_map.height]
        for dropout in self.config['dropouts']:
            dropout.center.x = dropout.center.x * Utils.DROPOUT_MOD[len(self.game.players)][self.game.my_id]['x']
            dropout.center.y = dropout.center.y * Utils.DROPOUT_MOD[len(self.game.players)][self.game.my_id]['y']
            dropout.center = self.game.game_map.normalize(dropout.center + self.game.me.shipyard.position)

        self.game.ready("SupervisedBot")

    def run(self):
        me = self.game.me
        # F - finish, S - search, D - deposit, B - build dropout
        data = {'structures': [me.shipyard], 'ships': [], 'states': {}, 'dropout': None}
        ships_created = 0
        turn_limit = constants.MAX_TURNS
        ship_building_dropout = None

        while True:
            self.game.update_frame()
            game_map = self.game.game_map
            turn_number = self.game.turn_number
            command_queue = []
            # data['structures'] = [d.position for d in list([me.shipyard] + me.get_dropoffs())]
            data = Utils.update_ship_states(me.get_ships(), data)
            data, config = Utils.make_dropout(game_map, me.get_ships(), data, self.config, turn_number)

            for ship in me.get_ships():
                if data['states'][ship.id] == 'F' or turn_limit - turn_number - self.config['finish'] < \
                        game_map.calculate_distance(ship.position, me.shipyard.position):
                    data['states'][ship.id] = 'F'
                elif ship.position in data['structures'] and data['states'][ship.id] != 'B':
                    data['states'][ship.id] = 'S'
                elif ship.halite_amount > self.deposit_halite_amount and data['states'][ship.id] != 'B':
                    data['states'][ship.id] = 'D'

                if data['states'][ship.id] == 'F':
                    move = Utils.navigate_to_finish_game(data['structures'], game_map, ship,
                                                         Utils.find_closest_structure(game_map, ship, data['structures']))
                elif data['states'][ship.id] == 'B':
                    if ship.position == data['dropout']['dropout_position'] and Utils.has_halite_to_make_dropout(self.game, ship):
                        command_queue.append(ship.make_dropoff())
                        data['states'][ship.id] = 'S'
                        data['structures'].append(ship.position)
                        data['dropout'] = None
                        del config['dropouts'][0]
                        ship_building_dropout = ship
                        continue
                    else:
                        move = Utils.navigate(game_map, ship, data['dropout']['dropout_position'], data)
                elif data['states'][ship.id] == 'S':
                    surroundings = Utils.get_surroundings(data, ship, game_map, self.radius, self.max_halite, self.min_halite)
                    predicted_actions = self.model.predict_choice_actions(np.array([surroundings]))
                    move = Utils.get_best_move(predicted_actions, game_map, ship)
                else:
                    move = Utils.navigate(game_map, ship, Utils.find_closest_structure(game_map, ship, data['structures']), data)

                command_queue.append(ship.move(move))

            if ship_building_dropout is not None:
                if me.halite_amount >= constants.SHIP_COST + Utils.get_cost_to_make_dropout(self.game, ship_building_dropout):
                    command_queue.append(me.shipyard.spawn())
                    ships_created += 1
                ship_building_dropout = None
            elif (data['dropout'] is None and Utils.can_make_ship(turn_number, me, game_map, config)) or \
                    (data['dropout'] is not None and me.halite_amount >= constants.SHIP_COST + constants.DROPOFF_COST):
                command_queue.append(me.shipyard.spawn())
                ships_created += 1

            if self.summary_name is not None and turn_number == turn_limit:
                self.summary_writer.add_summary(len(me.get_ships()),
                                                me.halite_amount / 1000.0,
                                                self.episode,
                                                game_map.height,
                                                ships_created,
                                                self.summary_name)

            self.game.end_turn(command_queue)
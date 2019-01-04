import hlt
from hlt import constants
from hlt.bot_utils import Utils, SummaryWriter
from hlt.positionals import Direction
import numpy as np
import sys
import os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from algorithms.supervised_model import SupervisedModel


class MLBot2:
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
        self.game.ready("SupervisedBot")

    def run(self):
        ship_states = {}
        ships_created = 0
        turn_limit = constants.MAX_TURNS
        max_halite, min_halite = Utils.get_max_min_map_halite(self.game.game_map)

        while True:
            self.game.update_frame()
            game_map = self.game.game_map
            turn_number = self.game.turn_number
            me = self.game.me
            command_queue = []
            my_structures_positions = [d.position for d in list(me.get_dropoffs()) + [me.shipyard]]
            my_ships_positions = [s.position for s in list(me.get_ships())]

            for ship in me.get_ships():
                if ship.id not in ship_states:
                    ship_states[ship.id] = 'search'

                if ship_states[ship.id] == 'finish' or turn_limit - turn_number - Utils.finish_dict[game_map.height] < \
                        game_map.calculate_distance(ship.position, me.shipyard.position):
                    ship_states[ship.id] = 'finish'
                elif ship.position == me.shipyard.position:
                    ship_states[ship.id] = 'search'
                elif ship.halite_amount > self.deposit_halite_amount:
                    ship_states[ship.id] = 'deposit'

                if ship_states[ship.id] == 'finish':
                    move = Utils.navigate_to_finish_game(my_structures_positions, game_map, ship, me.shipyard.position)
                elif ship_states[ship.id] == 'search':
                    surroundings = Utils.get_surroundings_new(my_structures_positions, my_ships_positions, ship, game_map, self.radius, max_halite, min_halite)
                    predicted_actions = self.model.predict_choice_actions(np.array([surroundings]))
                    move = Utils.get_best_move(predicted_actions, game_map, ship)
                else:
                    move = game_map.naive_navigate(ship, me.shipyard.position)

                command_queue.append(ship.move(move))

            if turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied \
                    and len(me.get_ships()) < Utils.SHIPS_LIMIT:
                command_queue.append(me.shipyard.spawn())
                ships_created += 1

            if self.summary_name is not None and turn_number == turn_limit:
                total_halite_collected_norm = ((ships_created * constants.SHIP_COST) + me.halite_amount - 5000) / 1000.0
                self.summary_writer.add_summary(len(me.get_ships()), total_halite_collected_norm, self.episode, game_map.height, self.summary_name)

            self.game.end_turn(command_queue)

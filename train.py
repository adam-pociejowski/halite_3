import os
from docopt import docopt

EPISODE_START = 1
EPISODES = 200
MAP_SIZE = 32
TURN_LIMIT = 200
LOAD_MODEL = 1
TRAIN = 1

for episode in range(EPISODE_START, EPISODE_START + EPISODES):
    os.system(f'halite.exe --no-logs --no-replay -vvv --turn-limit {TURN_LIMIT} --width {MAP_SIZE} --height {MAP_SIZE} --no-timeout '
              f'"python reinforcement-bot.py {episode} {TURN_LIMIT} {LOAD_MODEL} {TRAIN}" "python rule_based_bot.py {episode} {TURN_LIMIT}"')

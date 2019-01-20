import os
from docopt import docopt

EPISODE_START = 0
EPISODES = 1000
MAP_SIZE = 64
TURN_LIMIT = 500
LOAD_MODEL = 1
TRAIN = 1

for episode in range(EPISODE_START, EPISODE_START + EPISODES):
    # os.system(f'halite.exe --no-logs --no-replay -vvv --turn-limit {TURN_LIMIT} --width {MAP_SIZE} --height {MAP_SIZE} --no-timeout '
    #           f'"python reinforcement-bot.py {episode} {TURN_LIMIT} {LOAD_MODEL} {TRAIN}" "python rule_based_bot.py {episode} {TURN_LIMIT}"')
    # os.system(f'halite.exe --no-logs --no-replay -vvv --turn-limit {TURN_LIMIT} --width {MAP_SIZE} --height {MAP_SIZE} --no-timeout '
    #           f'"python supervised_bot.py {episode} {TURN_LIMIT}" "python rule_based_bot.py {episode} {TURN_LIMIT}"')

    # os.system(f'halite.exe --no-logs --no-replay -vvv --no-timeout "python supervised_bot.py {episode}" "python rule_based_bot.py {episode}"')
    os.system(f'halite.exe --replay-directory replays/ --no-logs -vvv "python SupervisedBot.py {episode}" "python SupervisedBot2.py {episode}"')

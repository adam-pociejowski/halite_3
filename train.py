import os
from docopt import docopt

EPISODE_START = 1
EPISODES = 100000
MAP_SIZE = 32
TURN_LIMIT = 200

for episode in range(EPISODE_START, EPISODE_START + EPISODES):
    os.system(f'halite.exe --no-logs --no-replay -vvv --turn-limit {TURN_LIMIT} --width {MAP_SIZE} --height {MAP_SIZE} --no-timeout '
              f'"python actor_critic_bot_2.py {episode} {TURN_LIMIT}" "python rule_based_bot.py {episode} {TURN_LIMIT}"')

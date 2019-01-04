import os
from docopt import docopt

EPISODE_START = 0
EPISODES = 200
HALITE = 800
HALITE_600 = 600
RADIUS = 10
RADIUS_2 = 16

# BOT 1
SUMMARY = 'bot_10_phase7_600'
MODEL = 'cnn-10-phase7'

# BOT 2
SUMMARY_2 = 'bot_10_phase7_800'
MODEL_2 = 'cnn-10-phase7'

# BOT 1
SUMMARY_3 = 'bot_16_phase2_600'
MODEL_3 = 'cnn'

# BOT 2
SUMMARY_4 = 'bot_16_phase7_800'
MODEL_4 = 'cnn-16-phase4'

MAP_SIZE = 64
TURN_LIMIT = 500
LOAD_MODEL = 1
TRAIN = 1

for episode in range(EPISODE_START, EPISODE_START + EPISODES):
    os.system(f'halite.exe --no-logs --no-replay -vvv '
              f'"python SupervisedBot.py {episode} {HALITE_600} {RADIUS} {SUMMARY} {MODEL_2}" '
              f'"python SupervisedBot.py {episode} {HALITE} {RADIUS} {SUMMARY_2} {MODEL_2}" '
              f'"python SupervisedBot.py {episode} {HALITE_600} {RADIUS_2} {SUMMARY_3} {MODEL_4}" '
              f'"python SupervisedBot.py {episode} {HALITE} {RADIUS_2} {SUMMARY_4} {MODEL_4}"')

import MLBot
import sys
from algorithms.supervised_model import SupervisedModel

episode = 0
deposit_halite_amount = 800
radius = 16
summary_name = 'supervised_bot_new'
model_name = 'cnn'


if len(sys.argv) >= 2:
    episode = int(sys.argv[1])
if len(sys.argv) >= 3:
    deposit_halite_amount = int(sys.argv[2])
if len(sys.argv) >= 4:
    radius = int(sys.argv[3])
if len(sys.argv) >= 5:
    summary_name = sys.argv[4]
if len(sys.argv) >= 6:
    model_name = sys.argv[5]


class SupervisedBot(MLBot.MLBot):
    def __init__(self, episode, deposit_halite_amount, radius, summary_name, model_name):
        super().__init__(episode=episode, deposit_halite_amount=deposit_halite_amount, radius=radius, summary_name=summary_name, model_name=model_name)


if __name__ == '__main__':
    bot = SupervisedBot(episode, deposit_halite_amount, radius, summary_name, model_name)
    bot.run()

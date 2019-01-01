#!/bin/sh
#./halite --replay-directory replays/ -vvv --width 32 --height 32 --turn-limit 200 --no-timeout "python reinforcement-bot.py 0 200 0 0" "python rule_based_bot.py 0 200"
#./halite --replay-directory replays/ -vvv --width 32 --height 32 --turn-limit 200 --no-timeout "python reinforcement-bot.py 0 200 0 1" "python rule_based_bot.py 0 200"
#./halite --replay-directory replays/ -vvv --width 32 --height 32 --turn-limit 200 --no-timeout "python rule_based_bot.py 0 200 0 1" "python rule_based_bot.py 0 200"
./halite --replay-directory replays/ -vvv --width 32 --height 32 --turn-limit 400 --no-timeout "python supervised_bot_train.py 0 100 0 1" "python rule_based_bot.py 0 100"

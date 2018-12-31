#!/bin/sh

./halite --replay-directory replays/ -vvv --width 32 --height 32 --turn-limit 200 --no-timeout "python actor_critic_bot_2.py 0 200" "python rule_based_bot.py 0 200"
#./halite --replay-directory replays/ -vvv --width 32 --height 32 --turn-limit 200 --no-timeout "python rule_based_bot.py 0 200" "python rule_based_bot.py 0 200"
#./halite --replay-directory replays/ -vvv --width 32 --height 32 --no-timeout "python rule_based_bot.py" "python rule_based_bot.py"
#./halite --replay-directory replays/ -vvv --width 32 --height 32 --no-timeout "python old-bot.py" "python rule_based_bot.py"

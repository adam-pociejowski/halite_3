#!/bin/sh
./halite --replay-directory replays/ -vvv "python SupervisedBot.py  --deposit_halite_amount=800 --radius=10 --summary_name=supervised_bot" "python SupervisedBot.py"

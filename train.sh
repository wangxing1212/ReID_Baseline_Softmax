#!/bin/sh
python -m visdom.server -logging_level WARNING &
bg
python main.py train
killall python
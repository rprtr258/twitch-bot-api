#!/usr/bin/env python

import sys

import main
import config


conf = config.load_config_and_init()
main.send(conf.twitch_config.sock, f"PASS {conf.twitch_config.password}")
main.send(conf.twitch_config.sock, f"NICK {conf.twitch_config.nick}")
message = ' '.join(sys.argv[1:])
for channel in conf.twitch_config.channels:
    main.send(conf.twitch_config.sock, f"JOIN #{channel}")
main.send_message(conf.twitch_config, "screamlark", message)

#!/usr/bin/env python

import logging
import re
import socket
import traceback

import commands
import config
import utils


def send(sock: socket.socket, message: str):
    sock.send((message + "\r\n").encode("utf-8"))

def send_message(twitch_config: config.TwitchConfig, channel: str, message: str):
    send(twitch_config.sock, f"PRIVMSG #{channel} :{message}")
    logging.info(utils.MessageRecord(
        username=twitch_config.nick,
        channel=channel,
        message=message,
    ))

def send_long_message(twitch_config: config.TwitchConfig, channel: str, message: str):
    message_len = len(message)
    i = 0
    while i < message_len:
        send_message(conf.twitch_config, channel, message[i:i + 500])
        i += 500

def every_message_action(conf: config.Config, message_record: utils.MessageRecord):
    logging.info(message_record)
    commands.feed(conf, message_record)

def action_on_message(conf: config.Config, message_record: utils.MessageRecord):
    every_message_action(conf, message_record)
    if message_record.message[0] != '!':
        return
    command, *param = message_record.message.split(' ', 1)
    command_message_record = utils.MessageRecord(
        username=message_record.username,
        message=None if param == [] else param[0],
        channel=message_record.channel,
    )
    # TODO: match case
    # TODO: list blab's pastas
    if command == "!блаб":
        text = commands.balaboba(conf, command_message_record)
        send_long_message(conf.twitch_config, message_record.channel, text)
    elif command == "!блаб-прочитать":
        text = commands.balaboba_read(conf, command_message_record)
        send_long_message(conf.twitch_config, message_record.channel, text)
    elif command == "!блаб-продолжить":
        text = commands.balaboba_continue(conf, command_message_record)
        send_long_message(conf.twitch_config, message_record.channel, text)
    elif command == "!say":
        send_message(conf.twitch_config, message_record.channel, commands.say(conf, command_message_record))
    elif command == "!pyth":
        send_message(conf.twitch_config, message_record.channel, commands.pyth(conf, command_message_record))
    elif command == "!feed":
        send_message(conf.twitch_config, message_record.channel, commands.feed_cmd(conf, command_message_record))
    elif command == "!commands":
        # TODO: extract somehow?
        send_message(
            conf.twitch_config,
            message_record.channel,
            f"@{username} Команды бота: " + ", ".join(
                f"!{cmd}" for cmd in ["блаб", "блаб-прочитать", "блаб-продолжить", "say", "pyth", "feed", "commands"]
            )
        )


conf = config.load_config_and_init()
send(conf.twitch_config.sock, f"PASS {conf.twitch_config.password}")
send(conf.twitch_config.sock, f"NICK {conf.twitch_config.nick}")
for channel in conf.twitch_config.channels:
    send(conf.twitch_config.sock, f"JOIN #{channel}")
# TODO: task with writing messages from bot
with conf.twitch_config.sock:
    is_running = True
    while is_running:
        try:
            conf.twitch_config.buffer += conf.twitch_config.sock.recv(2048).decode("utf-8")
            data_split = conf.twitch_config.buffer.split("\r\n")
            conf.twitch_config.buffer = data_split.pop()

            for line in data_split:
                if "PING" in line:
                    send(conf.twitch_config.sock, line.replace("PING", "PONG"))
                elif "PRIVMSG" in line:
                    match = re.match(r":([a-z0-9_]+)!\1@\1.tmi.twitch.tv PRIVMSG #([a-z0-9_]+) :(.*)", line)
                    username, channel, message = match.groups()
                    action_on_message(
                        conf=conf,
                        message_record=utils.MessageRecord(
                            username=username,
                            channel=channel,
                            message=message,
                        )
                    )
                else:
                    logging.info(line)
        except socket.timeout as e:
            pass
        except socket.error as e:
            logging.error("Socket died:", e)
            is_running = False
        except Exception:
            logging.error(traceback.format_exc())
            is_running = False

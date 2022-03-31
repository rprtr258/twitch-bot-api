#!/usr/bin/env python3

import commands
import config
import utils


def send_message(twitch_config: config.TwitchConfig, channel: str, message: str):
    print(message)
    # print(f"PRIVMSG #{channel} :{message}")

def send_long_message(twitch_config: config.TwitchConfig, channel: str, message: str):
    message_len = len(message)
    i = 0
    while i < message_len:
        send_message(twitch_config, channel, message[i:i + 500])
        i += 500

def every_message_action(conf: config.Config, message_record: utils.MessageRecord):
    commands.feed(conf, message_record)

last_hour_egg_was_open = -1

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
    if command == "!блаб":
        # TODO: fix balaboba
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
        send_message(
            conf.twitch_config,
            message_record.channel,
            f"@{message_record.username} Команды бота: " + ", ".join(
                f"!{cmd}" for cmd in ["блаб", "блаб-прочитать", "блаб-продолжить", "say", "pyth", "feed", "commands"]
            )
        )
    elif command == "!киндер":
        if param == []:
            import datetime
            now = datetime.datetime.now()
            hour = now.hour
            global last_hour_egg_was_open
            if last_hour_egg_was_open != hour:
                last_hour_egg_was_open = hour
                with open("eggs.txt", "r", encoding="utf-8") as fd:
                    eggs = [x.strip() for x in fd.readlines()]
                import random
                egg_inner = random.choice(eggs)
                send_message(
                    conf.twitch_config,
                    message_record.channel,
                    f"@{message_record.username} открыл яйцо и получил OOOO 👉 {egg_inner}"  # TODO: remove unicode
                )
            else:
                minutes = 60 - now.minute
                send_message(
                    conf.twitch_config,
                    message_record.channel,
                    f"@{message_record.username} подожди {minutes} минут"
                )
        elif message_record.username == "rprtr258":
            prize = ' '.join(param).strip()
            with open("eggs.txt", "a", encoding="utf-8") as fd:
                fd.write(prize + '\n')


def main():
    conf = config.load_config_and_init()
    while True:
        line = input()
        # print(line) # TODO: log to stderr
        username, channel, message = line.split(',', 2)
        action_on_message(
            conf=conf,
            message_record=utils.MessageRecord(
                username=username,
                channel=channel,
                message=message,
            )
        )

if __name__ == "__main__":
    main()


from dataclasses import dataclass
import datetime
import json
import logging
import os
import sqlite3
import socket
from typing import List

from quotes_generator.ngram import NGram


@dataclass
class PythConfig:
    ban_words: List[str]

def load_pyth_config() -> PythConfig:
    _BAN_COMMANDS = [
        "ban", "disconnect", "timeout", "unban", "slow", "slowoff", "followers", "followersoff", "subscribers",
        "subscribersoff", "clear", "uniquechat", "uniquechatoff", "emoteonly", "emoteonlyoff", "commercial", "host",
        "unhost", "raid", "unraid", "marker"
    ]
    BAN_COMMANDS = ["!commands"]
    for word in _BAN_COMMANDS:
        BAN_COMMANDS += ['/' + word, '.' + word]
    return PythConfig(
        ban_words=BAN_COMMANDS,
    )

@dataclass
class BalabobaConfig:
    balaboba_skip_constant: int

def load_balaboba_config() -> BalabobaConfig:
    TO_SKIP = len("please wait up to 15 seconds Без стиля".split())
    return BalabobaConfig(
        balaboba_skip_constant=TO_SKIP,
    )

@dataclass
class SayConfig:
    model: NGram
    vocabulary: List[str]

def load_say_config() -> SayConfig:
    model = NGram(3)
    model.load("model.json")
    with open("model.json", "r") as fd:
        vocabulary = json.load(fd)["prior"].keys()
    return SayConfig(
        model=model,
        vocabulary=vocabulary,
    )

@dataclass
class FeedConfig:
    FEED_TIME_LIMIT: datetime.timedelta

def load_feed_config() -> FeedConfig:
    return FeedConfig(
        FEED_TIME_LIMIT=datetime.timedelta(minutes=5),
    )

@dataclass
class TwitchConfig:
    channels: List[str]
    sock: socket.socket
    buffer: str
    nick: str
    password: str

def load_twitch_config() -> TwitchConfig:
    HOST = "irc.twitch.tv"
    PORT = 6667
    sock = socket.socket()
    sock.connect((HOST, PORT))
    sock.settimeout(1)
    NICK = os.environ["NICK"]
    PASSWORD = os.environ["PASSWORD"]
    return TwitchConfig(
        channels=["rprtr258", "screamlark"],
        sock=sock,
        buffer="",
        nick=NICK,
        password=PASSWORD,
    )

@dataclass
class DbConfig:
    conn: sqlite3.Connection

def load_db_config() -> DbConfig:
    return DbConfig(
        conn=sqlite3.connect("db.db", detect_types=sqlite3.PARSE_DECLTYPES),
    )

@dataclass
class Config:
    twitch_config: TwitchConfig
    db_config: DbConfig
    feed_config: FeedConfig
    say_config: SayConfig
    balaboba_config: BalabobaConfig
    pyth_config: PythConfig

def load_config_and_init() -> Config:
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.INFO
    )
    return Config(
        twitch_config=load_twitch_config(),
        db_config=load_db_config(),
        feed_config=load_feed_config(),
        say_config=load_say_config(),
        balaboba_config=load_balaboba_config(),
        pyth_config=load_pyth_config(),
    )

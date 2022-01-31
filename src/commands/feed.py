import datetime
from typing import Callable, List, Tuple
import sqlite3
import config

import commands.utils
import utils


def calc_weight(count: int) -> int:
    return 100 + 13 * count

def ensure_user_in_db(conn: sqlite3.Connection, username: str, channel: str, now: datetime.datetime):
    conn.execute("""
        INSERT OR IGNORE INTO feed (emote_id, username, at, count)
        SELECT id, ?, ?, 0 FROM emote WHERE emote.channel=?;
    """, (username, now, channel))
    conn.commit()

def feed_command(
    f: Callable[[config.DbConfig, config.FeedConfig, str, str, List[str]], str]
) -> Callable[[utils.MessageRecord], str]:
    def _wrapped(conf: config.Config, message_record: utils.MessageRecord) -> str:
        return f(
            db_config=conf.db_config,
            feed_config=conf.feed_config,
            channel=message_record.channel,
            username=message_record.username,
            words=message_record.message.split() if message_record.message else [],
        )
    return _wrapped

@feed_command
def feed(
    db_config: config.DbConfig,
    feed_config: config.FeedConfig,
    channel: str,
    username: str,
    words: List[str],
):
    if username == "Gekmi":
        return
    now = datetime.datetime.utcnow()
    ensure_user_in_db(db_config.conn, username, channel, now - feed_config.FEED_TIME_LIMIT)
    feedings = db_config.conn.execute("""
        SELECT emote.id, emote.emote, feed.at
        FROM feed
        JOIN emote on feed.emote_id=emote.id
        WHERE feed.username=? AND emote.channel=?;
    """, (username, channel)).fetchall()
    emotes_to_feed = set()
    for emote_id, emote, last_feed_time in feedings:
        if now - last_feed_time < feed_config.FEED_TIME_LIMIT:
            continue
        if emote not in words:
            continue
        emotes_to_feed.add(emote_id)
    if emotes_to_feed:
        emotes_id_set = ','.join(map(str, emotes_to_feed))
        db_config.conn.execute(f"""
            UPDATE feed SET count=count+1, at=? WHERE emote_id IN ({emotes_id_set}) AND username=?;
        """, (now, username))
        db_config.conn.commit()

@commands.utils.with_mention
@feed_command
def feed_cmd(
    db_config: config.DbConfig,
    feed_config: config.FeedConfig,
    channel: str,
    username: str,
    words: List[str],
):
    def _feed_info(emotes_to_show: List[Tuple[str, int]], username=None):
        user = username if username else "ты"
        return f"{user} покормил: " + "; ".join(
            f"{word} {count} раз, вес {calc_weight(count)} грамм"
            for word, count in emotes_to_show
        )

    def _list_all_emotes():
        emotes_to_show = db_config.conn.execute("""
            SELECT emote.emote, emote.word FROM emote WHERE emote.channel=?;
        """, (channel,)).fetchall()
        return "ты можешь покормить " + ", ".join(
            f"{word}: {emote} "
            for emote, word in emotes_to_show
        )

    if words == []:
        return _list_all_emotes()

    if words == ["all"]:
        emotes_to_show = db_config.conn.execute("""
            SELECT emote.word, feed.count FROM feed JOIN emote ON feed.emote_id=emote.id
            WHERE feed.username=? AND emote.channel=? AND feed.count > 0;
        """, (username, channel)).fetchall()
        if emotes_to_show:
            return _feed_info(emotes_to_show)
        return _list_all_emotes()

    if len(words) == 1:
        maybe_username = words[0].lower()
        emotes_to_show = db_config.conn.execute("""
            SELECT emote.word, feed.count FROM feed JOIN emote ON feed.emote_id=emote.id
            WHERE feed.username=? AND emote.channel=? AND feed.count > 0;
        """, (maybe_username, channel)).fetchall()
        if emotes_to_show:
            return _feed_info(emotes_to_show, username=maybe_username)

    words_sql = ','.join(map(repr, words))
    emotes_to_show = db_config.conn.execute(f"""
        SELECT emote.word, feed.count FROM feed JOIN emote ON feed.emote_id=emote.id
        WHERE emote.emote IN ({words_sql}) AND feed.username=? AND emote.channel=? AND feed.count > 0;
    """, (username, channel)).fetchall()
    if emotes_to_show:
        return _feed_info(emotes_to_show)

    return _list_all_emotes()

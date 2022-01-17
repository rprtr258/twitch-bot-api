import os
import subprocess
from typing import Tuple

import commands.utils
import config
import utils


# TODO: styles
def balabob(TO_SKIP: int, text: str) -> Tuple[bool, str]:
    # TODO: match
    if os.name == "posix":
        output = subprocess.check_output(["./balaboba"] + text.split())
    elif os.name == "nt":
        output = subprocess.check_output(["./balaboba.exe"] + text.split())
    output = output.decode("utf-8").split()
    if "на острые темы, например, про политику или религию" in ' '.join(output):
        return False, "PauseFish"
    return True, ' '.join(output[TO_SKIP:])

@commands.utils.if_empty_message("что? FeelsWeakMan")
@commands.utils.with_mention
def balaboba(conf: config.Config, message_record: utils.MessageRecord):
    to_skip = conf.balaboba_config.balaboba_skip_constant
    did_generate, response = balabob(to_skip, message_record.message)
    if did_generate:
        # TODO: autoincrement
        new_id = conf.db_config.conn.execute("""
            SELECT MAX(id) + 1 FROM balaboba;
        """).fetchone()[0]
        conf.db_config.conn.execute("""
            INSERT INTO balaboba (id, pasta) VALUES (?, ?);
        """, (new_id, response))
        return f"[{new_id}]: {response}"
    else:
        return response

@commands.utils.with_mention
def balaboba_read(conf: config.Config, message_record: utils.MessageRecord):
    if message_record.message:
        try:
            id = int(message_record.message)
        except ValueError:
            return f"Использование: !блаб-прочитать 42"
    else:
        return f"Использование: !блаб-прочитать 42"
    response = conf.db_config.conn.execute("""
        SELECT pasta FROM balaboba WHERE id=?;
    """, (id,)).fetchone()
    if response:
        return response[0]
    else:
        return f"Паста не найдена fishShy"

@commands.utils.with_mention
def balaboba_continue(conf: config.Config, message_record: utils.MessageRecord):
    if message_record.message:
        try:
            id = int(message_record.message)
        except ValueError:
            return f"Использование: !блаб-продолжить 42"
    else:
        return f"Использование: !блаб-продолжить 42"
    old_text = conf.db_config.conn.execute("""
        SELECT pasta FROM balaboba WHERE id=?;
    """, (id,)).fetchone()
    if old_text:
        old_text = old_text[0]
        to_skip = conf.balaboba_config.balaboba_skip_constant
        # TODO: continue even if text is too long
        did_generate, response = balabob(to_skip, old_text)
        if did_generate and response != old_text:
            conf.db_config.conn.execute("""
                UPDATE balaboba SET pasta=? WHERE id=?;
            """, (response, id))
            return response[len(old_text):]
        else:
            return "Продолжения не будет PauseFish"
    else:
        return f"Паста не найдена fishShy"

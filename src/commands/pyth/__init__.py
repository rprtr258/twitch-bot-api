import logging
import subprocess

import commands.utils
import config
import utils

PYTH_CMD_FILE = "c"


@commands.utils.if_empty_message("Provide pyth program: pyth.readthedocs.io")
def pyth(conf: config.Config, message_record: utils.MessageRecord):
    if message_record.username in ["passionsausage", "morglod"]:
        return f"{message_record.username} наказан за абьюз системы AAUGH"
    with open(PYTH_CMD_FILE, "w", encoding="utf-8") as fd:
        fd.write(message_record.message)
    ps = subprocess.run(['echo', '6'], check=True, capture_output=True)
    # TODO: fix paths
    # TODO: log stderr
    if message_record.username == "rprtr258":
        processNames = subprocess.run(["python3", "src/pyth/pyth.py", PYTH_CMD_FILE], input=ps.stdout, capture_output=True)
    else:
        processNames = subprocess.run(["python3", "src/pyth/pyth.py", "-s", PYTH_CMD_FILE], input=ps.stdout, capture_output=True)
    subprocess.run(["rm", PYTH_CMD_FILE])
    out = processNames.stdout.decode('utf-8').replace('\r', '').replace('\n', ' ')
    logging.info("PYTH PROGRAM: %s OUTPUT: %s", repr(message_record.message), repr(out))
    if any(cmd in out for cmd in conf.pyth_config.ban_words):
        return f"@{message_record.username}, как особо опасный хакер ты приговариваешься к расстрелу! PauseFish"
    if out.strip() == "":
        return "EMPTY_OUTPUT AAUGH"
    return out

import config
import utils


def with_mention(f):
    def _wrapped(conf: config.Config, message_record: utils.MessageRecord):
        response = f(conf, message_record)
        return f"@{message_record.username} {response}"
    return _wrapped

def if_empty_message(response: str):
    def _decorator(f):
        def _wrapped(conf: config.Config, message_record: utils.MessageRecord):
            if message_record.message:
                return f(conf, message_record)
            return response
        return _wrapped
    return _decorator

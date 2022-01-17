import math
import random
import commands

# import edlib # TODO: fix

import commands.utils
import config
import utils


def levenshteinDistance(s1, s2):
    def _helper(s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    s1, s2 = s1.lower(), s2.lower()
    # return edlib.align(s1, s2)["editDistance"]
    return _helper(s1, s2)

def get_begins(message_word, vocabulary):
    words = {}
    for prior_ngram in vocabulary:
        if ' ' not in prior_ngram: # TODO: check why priors contain not-2grams
            continue
        dist = levenshteinDistance(prior_ngram, message_word)
        if dist < len(message_word) * 1.2 or len(words) == 0:
            words[prior_ngram] = dist
    return list(sorted(words.items(), key=lambda kv: kv[1]))[:10]

# TODO: speed up
@commands.utils.with_mention
def say(conf: config.Config, message_record: utils.MessageRecord):
    # TODO: move logic to ngram, remove vocabulary from config
    model = conf.say_config.model
    vocabulary = conf.say_config.vocabulary
    if not message_record.message:
        return model.generate()
    message = message_record.message.split()
    message = sorted(message, key=len)[-5:]
    begins = sum([get_begins(w, vocabulary) for w in message], [])
    begin = random.choices([w for w, _ in begins], [math.pow(2, -p) for _, p in begins])
    return model.generate(begin)

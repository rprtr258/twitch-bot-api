CREATE TABLE emote (
    id INT PRIMARY KEY,
    emote TEXT,
    word TEXT,
    channel TEXT,
    UNIQUE(emote, channel),
    UNIQUE(word, channel)
);
INSERT OR IGNORE INTO emote (id, emote, word, channel) VALUES
    (0, 'AAUGH', 'рыбку', 'screamlark'),
    (1, 'VeryLark', 'стримера', 'screamlark'),
    (2, 'VeryPog', 'лысого из C++', 'screamlark'),
    (3, 'VerySus', 'AMOGUS', 'screamlark'),
    (4, 'VeryPag', 'анимешку', 'screamlark'),
    (5, 'SadgeCry', 'FeelsWeakMan', 'screamlark'),
    (6, 'LewdIceCream', 'бороду', 'screamlark'),
    (7, 'AAUGH', 'рыбку', 'rprtr258');

CREATE TABLE feed (
    emote_id INT,
    username TEXT,
    count INT,
    at TIMESTAMP,
    PRIMARY KEY(emote_id, username)
    -- FOREIGN KEY emote_id REFERENCES emote(id)
);

CREATE TABLE IF NOT EXISTS balaboba (
    id INT PRIMARY KEY,
    pasta TEXT
);
INSERT INTO balaboba (id, pasta) VALUES
(0, 'Секретная паста OOOO');
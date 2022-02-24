## How to launch (supposing you have installed [task](https://taskfile.dev), git, sqlite3 and golang compiler)

```bash
git clone https://github.com/rprtr258/twitch-bot-api/
cd twitch-bot-api
task init
```
Put nickname and oauth token in `.env` file like so:

```bash
NICK="trooba_bot"
PASSWORD="oauth:OLOLOLOLOLOLOLOLOLOLOLOLOLOLOL"
```

Then run bot with `task run`

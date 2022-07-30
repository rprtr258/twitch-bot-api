build_receiver: # Build IRC receiver
	rm receiver || echo 'no receiver'
	gcc -Wall -Wconversion receiver.c -o receiver
	strip receiver

run_receiver: # Run IRC receiver
	task: build_receiver
	./receiver $(cat ./oauth.txt) rprtr258

build_sender: # Build IRC sender
	rm sender || echo 'no sender'
	gcc -Wall -Wconversion sender.c -o sender
	strip sender

run_sender: # Run IRC sender
	task: build_sender
	./sender $(cat ./oauth.txt) rprtr258

  build_balancer:
    desc: Build balancer
    dir: balancer
    cmds:
      - rm ../run_balancer || echo 'no balancer'
      - cargo build
      - strip target/debug/balancer
      - ln target/debug/balancer ../run_balancer

run_balancer: # Run balancer
	task: build_balancer
	./run_balancer balancer_config.txt

run: # Launch bot
	winpty python3 src/main.py

send: # Send message from bot
	python3 src/cmd.py '{{.CLI_ARGS}}'

init: # Checkout submodules, init database and build balaboba binary
	git submodule update --init --recursive
	sudo apt install sqlite3
	sqlite3 db.db < sql/init.sql
	(cd src/balaboba; go build -o ../../balaboba cmd/cmd.go)

  linter:
    desc: Run linter
    dir: src
    cmds:
      - flake8 --filename src/**/*.py --max-line-length 120 --ignore=E302,E305 --exclude __init__.py,src/pyth/*,src/quotes_generator/*

  get_chat_logs:
    desc: Collect data for training markov chain
    generates: [dataset]
    cmds:
      - tcd -q --channel screamlark --first 10
      - cat *.txt | rg -v 'StreamElements' | cut -d' ' -f3- > logs
      - cat dataset logs | rg -v '(!(g|say)|Продолжай кормить рыбку!|ставки|^\d{1,3} \d{1,3}$)' | sort -u > tmp_dataset
      - mv tmp_dataset dataset
      - rm logs

  train:
    desc: Train markov chain on collected data
    deps: [get_chat_logs]
    sources: [dataset]
    generates: [model.json]
    cmds:
      - python quotes_generator/main.py train --dataset dataset --model model.json

deploy-fix: # Commit, push and deploy
	task: commit
	task: push

deploy: # Prepare new data, commit, push and deploy
	task: train
	task: deploy-fix

dump-db: # Dump balaboba table into blab.sql and feed table into feed.sql
	sqlite3 db.db < sql/dump.sql

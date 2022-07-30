.PHONY: help
help: # Show list of all commands
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: build-receiver
build-receiver: # Build IRC receiver
	rm receiver || echo 'no receiver'
	gcc -Wall -Wconversion receiver.c -o receiver
	strip receiver

.PHONY: run-receiver
run-receiver: # Run IRC receiver
	$(MAKE) build_receiver
	./receiver $(cat ./oauth.txt) rprtr258

.PHONY: build-sender
build-sender: # Build IRC sender
	rm sender || echo 'no sender'
	gcc -Wall -Wconversion sender.c -o sender
	strip sender

.PHONY: run-sender
run-sender: # Run IRC sender
	$(MAKE) build-sender
	./sender $(cat ./oauth.txt) rprtr258

.PHONY: build-balancer
build-balancer: # Build balancer
	rm run_balancer || echo 'no balancer'
	cd balancer
	cargo build
	strip target/debug/balancer
	ln target/debug/balancer ../run_balancer

.PHONY: run_balancer
run_balancer: # Run balancer
	$(MAKE) build_balancer
	./run_balancer balancer_config.txt

.PHONY: run
run: # Launch bot
	python3 src/main.py

.PHONY: send
send: # Send message from bot
	python3 src/cmd.py $(MSG)

.PHONY: init
init: # Checkout submodules, init database and build balaboba binary
	git submodule update --init --recursive
	sudo apt install sqlite3
	sqlite3 db.db < sql/init.sql
	cd src/balaboba
	go build -o ../../balaboba cmd/cmd.go

.PHONY: linter
linter: # Run linter
	cd src
	flake8 --filename src/**/*.py --max-line-length 120 --ignore=E302,E305 --exclude __init__.py,src/pyth/*,src/quotes_generator/*

.PHONY: get_chat_logs
get_chat_logs: # Collect data for training markov chain
	tcd -q --channel screamlark --first 10
	cat *.txt | rg -v 'StreamElements' | cut -d' ' -f3- > logs
	cat dataset logs | rg -v '(!(g|say)|Продолжай кормить рыбку!|ставки|^\d{1,3} \d{1,3}$)' | sort -u > tmp_dataset
	mv tmp_dataset dataset
	rm logs

.PHONY: train
train: get_chat_logs # Train markov chain on collected data
	python quotes_generator/main.py train --dataset dataset --model model.json

.PHONY: deploy-fix
deploy-fix: # Commit, push and deploy
	task: commit
	task: push

.PHONY: deploy
deploy: # Prepare new data, commit, push and deploy
	task: train
	task: deploy-fix

.PHONY: dump-db
dump-db: # Dump balaboba table into blab.sql and feed table into feed.sql
	sqlite3 db.db < sql/dump.sql

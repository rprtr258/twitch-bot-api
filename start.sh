set -e
set -o pipefail
./receiver $(cat ../rprtr259.oauth) | ./src/main.py | ./sender $(cat ../rprtr259.oauth)

all: server

test:
	python ./src/server/worker_test.py

server:
	python ./src/server/server.py

proto: go python

go:
	./generate-go.sh

python:
	./generate-py.sh
all: server

server:
	python ./src/server/server.py

proto: go python

go:
	./generate-go.sh

python:
	./generate-py.sh
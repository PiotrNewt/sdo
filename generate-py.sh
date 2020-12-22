#!/bin/bash

cd src/proto
python3 -m grpc_tools.protoc -I. --python_out=../server --grpc_python_out=../server *.proto
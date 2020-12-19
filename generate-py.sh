#!/bin/bash

cd src/proto
python3 -m grpc_tools.protoc -I. --python_out=../py-mlpb --grpc_python_out=../py-mlpb *.proto
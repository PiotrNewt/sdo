#!/bin/bash

GOGO_ROOT=${GOPATH}/src/github.com/gogo/protobuf

cd ./src/proto
echo "generate go pb code"
protoc -I.:${GOGO_ROOT}:${GOGO_ROOT}/protobuf --gofast_out=plugins=grpc:../go-mlpb *.proto
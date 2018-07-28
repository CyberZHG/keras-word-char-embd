#!/usr/bin/env bash
mkdir -p dataset
cd dataset
if [ ! -f /dataset/aclImdb_v1.tar.gz ]; then
    wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    tar -xvzf aclImdb_v1.tar.gz
fi

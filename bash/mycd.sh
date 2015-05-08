#! /usr/bin/env bash

cd() {
    if [ $# -eq 0 ]; then
        builtin cd ~
    elif [ -d ~/learn/cs$1 ]; then
        builtin cd ~/learn/cs$1
    elif [ -d ~/learn/ee$1 ]; then
        builtin cd ~/learn/ee$1
    elif [ -d ~/learn/er$1 ]; then
        builtin cd ~/learn/er$1
    else
        builtin cd $1
    fi
}

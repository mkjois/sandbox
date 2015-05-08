#! /usr/bin/env bash

cd() {
    if [ $# -eq 0 ]; then
        builtin cd ~
    elif [ -d $UNIVDIR/cs$1 ]; then
        builtin cd $UNIVDIR/cs$1
    elif [ -d $UNIVDIR/ee$1 ]; then
        builtin cd $UNIVDIR/ee$1
    elif [ -d $UNIVDIR/er$1 ]; then
        builtin cd $UNIVDIR/er$1
    else
        builtin cd $1
    fi
}

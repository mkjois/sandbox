#!/usr/bin/env python

import re
import string
import time


def answer(chunk, word):
    matches = re.findall(r'(?=%s)' % word, chunk)
    if len(matches) < 2:
        return string.replace(chunk, word, '')
    other = string.replace(chunk, word, '', 1)
    for _ in xrange(20):
        chunk = string.replace(chunk, word, '')


def powerset(thing):
    if not hasattr(thing, '__len__') or not hasattr(thing, '__getslice__'):
        raise ValueError('argument to powerset() must have __len__ and __getslice__')
    if len(thing) == 0:
        yield type(thing)()
    else:
        for rest in powerset(thing[:-1]):
            yield rest + thing[-1:]
            yield rest


if __name__ == '__main__':
    S, T = 'hpqiuhqpsidhafsdpqifuubpaiwuefneaipq', 0
    start = time.time()
    for _ in xrange(T):
        string.replace(S, 'pq', '')
    end = time.time()
    regex = re.compile(r'pq')
    start = time.time()
    for _ in xrange(T):
        re.sub(regex, '', S)
    end = time.time()
    for s in powerset(list(xrange(4))):
        print(s)
    for s in powerset('abcd'):
        print(s)

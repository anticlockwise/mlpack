import numpy
import collections
import functools
from itertools import ifilterfalse
from heapq import nsmallest
from operator import itemgetter

LINEAR     = 0
POLYNOMIAL = 1
RBF        = 2
SIGMOID    = 3
CUSTOM     = 4

class Counter(dict):
    def __missing__(self, key):
        return 0

def lru_cache(max_size=100):
    max_queue = max_size * 10
    def decorating_function(user_function, len=len, iter=iter, tuple=tuple,
            sorted=sorted, KeyError=KeyError):
        cache = {}
        queue = collections.deque()
        refcount = Counter()
        sentinel = object()
        kwd_mark = object()

        queue_append, queue_popleft = queue.append, queue.popleft
        queue_appendleft, queue_pop = queue.appendleft, queue.pop

        @functools.wraps(user_function)
        def wrapper(*args, **kwds):
            key = args
            if kwds:
                key += (kwd_mark,) + tuple(sorted(kwds.items()))

            queue_append(key)
            refcount[key] += 1

            try:
                result = cache[key]
                wrapper.hits += 1
            except KeyError:
                result = user_function(*args, **kwds)
                cache[key] = result
                wrapper.misses += 1

                if len(cache) > max_size:
                    key = queue_popleft()
                    refcount[key] -= 1
                    while refcount[key]:
                        key = queue_popleft()
                        refcount[key] -= 1
                    del cache[key], refcount[key]

            if len(queue) > max_queue:
                refcount.clear()
                queue_appendleft(sentinel)
                for key in ifilterfalse(refcount.__contains__,
                        iter(queue_pop, sentinel)):
                    queue_appendleft(key)
                    refcount[key] = 1

            return result

        def clear():
            cache.clear()
            queue.clear()
            refcount.clear()
            wrapper.hits = wrapper.misses = 0

        wrapper.hits = wrapper.misses = 0
        wrapper.clear = clear
        return wrapper
    return decorating_function

class Kernel(object):
    def __init__(self, config):
        self.config = config

    def compute(self, xi, xj):
        return 0.0

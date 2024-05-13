from collections import deque


class FixedQueue(deque):
    def __init__(self, maxlen):
        super().__init__(maxlen=maxlen)

import numpy as np
import torch


def add(a, b):
    return a + b


class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, num):
        self.result += num
        return self.result


cal1 = Calculator()

test = cal1.add(3)
print(test)
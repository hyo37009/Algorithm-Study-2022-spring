from math import *

class Heap:
    def __init__(self, heap):
        self.heap = [0] + heap
        self.heapsize = len(heap)
        self.height = floor(log2(self.heapsize))

    def parent(self, i):
        return i//2
    def left(self, i):
        return 2 * i
    def right(self, i):
        return 2 * i + 1

    def max_heapify(self, i):
        l = self.left(i)
        r = self.right(i)

        if l <= self.heapsize and self.heap[r] > self.heap[i]:
            largest = l
        else: largest = i
        if r <= self.heapsize and self.heap[r] > self.heap[largest]:
            largest = r

        if largest != i:
            self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
            self.max_heapify(largest)

    def buildMaxHeap(self):
        self.


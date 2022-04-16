"""
A_star 2D
"""

import os
import sys
import math
import heapq

import plotting, env

class AStar:
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        self.Env = env.Env()

        self.u_set = self.Env.mations
        self.obs = self.Env.obs

        self.OPEN = []
        self.CLOSED  = []
        self.PARENT = dict()
        self.g = dict()

    def searching(self):
        '''
        A스타 탐색.
        :return: path, 방문 순서
        '''

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN, (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(self.s_start), self.s_start))

        return self.extract_path(self.PARENT), self.CLOSED

    def searching_repeated_astar(self, e):
        pass

    def repeated_searching(self, s_start, s_goal, e):
        pass

    def get_neighbor(self, s):
        """
        s 주변에 장애물 상태가 아닌 neighbors를 찾는다
        :param s: state
        :return: neighbors
        """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]
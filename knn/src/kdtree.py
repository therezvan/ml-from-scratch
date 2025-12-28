# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 14:22:29 2025

@author: rezi
"""
import numpy as np
import heapq


class KDtreeNode:
    def __init__(self, point, label, left=None, right=None, axis=0):
        self.label = label
        self.point = point
        self.left = left
        self.right = right
        self.axis = axis


class KDtree:
    def __init__(self, X, y):

        self.root = self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):
        if len(X) == 0:
            return None
        axis = depth % X.shape[1]
        stored_idx = X[:, axis].argsort()
        X = X[stored_idx]
        y = y[stored_idx]

        median = len(X) // 2

        return KDtreeNode(point=X[median], label=y[median],
                          left=self.build_tree(
                              X[:median], y[:median], depth+1),
                          right=self. build_tree(
                              X[median+1:], y[median+1:], depth + 1),
                          axis=axis)

    def query(self, x, k=1):

        heap = []

        def search(node):
            if node is None:
                return
            dist = np.linalg.norm(node.point - x)
            if len(heap) < k:
                heapq.heappush(heap, (-dist, node.label))

            else:
                if -dist > heap[0][0]:
                    heapq.heappushpop(heap, (-dist, node.label))

            axis = node.axis
            diff = x[axis] - node.point[axis]

            if diff < 0:
                close = node.left
                away = node.right
            else:
                close = node.right
                away = node.left

            search(close)

            if len(heap) < k or abs(diff) < -heap[0][0]:
                search(away)
        search(self.root)
        result = []
        for neg_dist, label in sorted(heap, reverse=True):
            real_dist = -neg_dist
            result.append((label, real_dist))
        return result

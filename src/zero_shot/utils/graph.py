import numpy as np
from collections import defaultdict
from functools import  partial
from queue import PriorityQueue, Queue
import warnings
import heapq
from heapq import heappush, heappop
import sys
sys.setrecursionlimit(10000)

directions = np.array([(-1, 0),(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]) 
def unilateral_dfs_tree(g_mask, inp_mask, start, b_mask = None, alpha=0.1, thresh=0.95, context_size=3, dis_map=None, tolerance=-0.5):
    h, w = inp_mask.shape
    global directions
    rows, cols = g_mask.shape[:2]
    stack = PriorityQueue()
    # Dijikstra
    status = np.zeros(g_mask.shape, dtype=int)

    # This to indicate strictly incremental path
    g_mask = np.abs(g_mask - 0.001)
    cost = np.full_like(g_mask, 1e4)
    px, py = start
    if dis_map is not None:
        stack.put((0, 0, (dis_map[px, py][None, :], start)))
    else: 
        stack.put((0, 0, (np.zeros(1, 2), start)))
    cost[px, py] = 0
    leaves = set()
    border = set()
    begin = start
    parent = {}
    dead = set()
    dfs_tree = defaultdict(list)
    i = 0
    while not stack.empty():
        state, _, (moves, (x, y)) = stack.get()
        # print(moves, x, y)
        prior = np.mean(moves, axis=0)
        dis_map[x, y] = (prior + dis_map[x, y]) / 2
        # print(dis_map[x, y])
        valid = (directions * dis_map[x, y]).sum(axis=1)
        score = np.abs(valid).tolist()
        indexes = [index for index, s in sorted(enumerate(score), key= lambda x: x[1]) if valid[index] >= tolerance]
        accepted_dir = [directions[i] for i in indexes]
        scores = [score[i] for i in indexes]
        # print(accepted_dir)
        for di in accepted_dir:
            (dx, dy) = di.tolist()
            if moves.shape[0] >= context_size: 
                moves = moves[1:].copy()
            di_state = np.concatenate([moves, np.array([(dx, dy)])], axis=0)
            
            nx, ny = x + dx, y + dy
            if ((nx - h) * nx > 0) or ((ny - w) * ny > 0):
                continue
            
            if 0 <= nx < rows and 0 <= ny < cols and g_mask[nx, ny] > alpha and status[nx, ny] <= status[x, y]:
                # Update on inverse confidence
                if cost[nx, ny] > state + g_mask[nx, ny]:
                    # Set cost as uncertainty gain 
                    i += 1
                    cost[nx, ny] = state + g_mask[nx, ny]
                    # print(cost[nx, ny])
                    # Start by largest margin
                    # Erase entry from other branch
                    if (nx, ny) in parent.keys():
                        # print(dfs_tree[parent[(nx, ny)]])
                        (prx, pry) = parent[(nx, ny)]
                        if cost[prx, pry] < cost[x, y]:
                            dead.add(parent[(nx, ny)])
                            dfs_tree[parent[(nx, ny)]].remove((nx, ny))
                            stack.put((float(cost[nx, ny]), i, (np.zeros([1, 2]), (nx, ny))))
                        else:
                            stack.put((float(cost[nx, ny]), i, (di_state, (nx, ny))))
                    else:
                        stack.put((float(cost[nx, ny]), i, (di_state, (nx, ny))))
                    parent[(nx, ny)] = (x, y)
                    # Determine that they have gone out of beta mask
                    # Odd for out-going, Even for in-going.
                    if b_mask is None:
                        if (g_mask[nx, ny] - thresh) * (g_mask[x, y] - thresh) < 0:
                            if status[nx, ny] == 0:
                                border.add((nx, ny))
                            status[nx, ny] = status[x, y] + 1
                        else:
                            status[nx, ny] = status[x, y]
                    else: 
                        if b_mask[nx, ny] != b_mask[x, y]:
                            if status[nx, ny] == 0:
                                border.add((nx, ny))
                            status[nx, ny] = status[x, y] + 1
                        else:
                            status[nx, ny] = status[x, y]
                        
                    dfs_tree[(x, y)].append((nx, ny))
            
        if i == 0:
            # Force leaves to be sink
            status[x, y] += inp_mask[x, y] % 2
            leaves.add((x, y))
            # Continual of flow
            if inp_mask[x, y] == 1:
                begin = (x, y)
    return {'dfs_tree': dfs_tree, 
            'parent': parent, 
            'cost': cost, 
            'border': border, 
            'leaves': leaves, 
            'status': status, 
            'begin': begin,
            'dis_map': dis_map,
            'dead': dead}

def dfs_tree(g_mask, inp_mask, start, weight = 1, alpha=0.1, thresh=0.95):
    rows, cols = g_mask.shape[:2]
    stack = PriorityQueue()
    stack.put((0, start))

    # Dijikstra
    status = np.zeros(g_mask.shape, dtype=int)

    # This to indicate strictly incremental path
    g_mask = np.abs(g_mask - 0.001)
    cost = np.full_like(g_mask, 1e4)
    px, py = start
    cost[px, py] = 0
    leaves = set()
    border = set()
    begin = start
    parent = {}
    directions = [(-1, 0),(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)] 
    dfs_tree = defaultdict(list)
    
    while not stack.empty():
        
        state, (x, y) = stack.get()
        i = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and g_mask[nx, ny] > alpha and status[nx, ny] <= status[x, y]:
                # Update on inverse confidence
                if cost[nx, ny] > state + (1 - g_mask[nx, ny]) * weight:
                    # Set cost as uncertainty gain 
                    i += 1
                    cost[nx, ny] = state + (1 - g_mask[nx, ny]) * weight
                    # Start by largest margin
                    stack.put((float(cost[nx, ny]), (nx, ny)))
                    # Erase entry from other branch
                    if (nx, ny) in parent.keys():
                        # print(dfs_tree[parent[(nx, ny)]])
                        dfs_tree[parent[(nx, ny)]].remove((nx, ny))

                    parent[(nx, ny)] = (x, y)
                    # Determine that they have gone out of mask
                    # Odd for out-going, Even for in-going.
                    if (g_mask[nx, ny] - thresh) * (g_mask[x, y] - thresh) < 0 :
                        if status[nx, ny] == 0:
                            border.add((nx, ny))
                        status[nx, ny] = status[x, y] + 1
                    else:
                        status[nx, ny] = status[x, y] 
                    dfs_tree[(x, y)].append((nx, ny))
            
        if i == 0:
            # Force leaves to be sink
            status[x, y] += inp_mask[x, y] % 2
            leaves.add((x, y))
            # Continual of flow
            if inp_mask[x, y] == 1:
                begin = (x, y)
    return {'dfs_tree': dfs_tree, 
            'parent': parent, 
            'cost': cost, 
            'border': border, 
            'leaves': leaves, 
            'status': status, 
            'begin': begin}

# We do post 
def longest_path_branching(tree, start, dead = []):
    visited = set()
    branches = []
    def dfs(node, path, visited, branches, state=True):
        valid = False
        if len(tree[node]) == 0:
            if node in dead:
                return [], False
            return [node], True
        if node in visited:
            return path, False
        paths = []
        visited.add(node)
        for neighbor in tree[node]:
            if neighbor in visited:
                continue
            path, ret = dfs(neighbor, path[:], visited, branches)
            if ret is True:
                valid = True
                paths.append(path + [node])

        if len(paths) == 0:
            max_path = [node]
        else:
            paths = sorted(paths, key= lambda x: -len(x))
            max_path = paths[0]
            branches += paths[1:]
        return max_path, valid
    output, _ = dfs(start, [], visited, branches)
    return  branches + [output] 

def longest_path(tree, start):
    def dfs(node, path):
        path.append(node)
        max_path = path[:]
        
        for neighbor in tree[node]:
            new_path = dfs(neighbor, path[:])
            if len(new_path) > len(max_path):
                max_path = new_path
        
        return max_path
    
    return dfs(start, [])
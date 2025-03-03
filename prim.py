from itertools import combinations
from heapq import *

def prim_real(vertexs, node_pos, net_info, ratio):
    vertexs = list(vertexs)
    if len(vertexs)<=1:
        return 0
    
    adjacent_dict :dict[str, list[tuple[int, str, str]]] = {}
    for node in vertexs:
        adjacent_dict[node] = []

    node_info = net_info['nodes']
    port_info = net_info['ports']

    for node1, node2 in list(combinations(vertexs, 2)):
        if node1 in node_pos:
            pin_x_1 = node_pos[node1][0] * ratio + node_info[node1]["pin_offset"][0]
            pin_y_1 = node_pos[node1][1] * ratio + node_info[node1]["pin_offset"][1]
        else:
            pin_x_1 = port_info[node1]["pin_offset"][0]
            pin_y_1 = port_info[node1]["pin_offset"][1]
        if node2 in node_pos:
            pin_x_2 = node_pos[node2][0] * ratio + node_info[node2]["pin_offset"][0]
            pin_y_2 = node_pos[node2][1] * ratio + node_info[node2]["pin_offset"][1]
        else:
            pin_x_2 = port_info[node2]["pin_offset"][0]
            pin_y_2 = port_info[node2]["pin_offset"][1]

        weight = abs(pin_x_1-pin_x_2) + abs(pin_y_1-pin_y_2)
        adjacent_dict[node1].append((weight, node1, node2))
        adjacent_dict[node2].append((weight, node2, node1))

    
    start = vertexs[0]
    minu_tree = []
    visited = set()
    visited.add(start)
    adjacent_vertexs_edges = adjacent_dict[start]
    heapify(adjacent_vertexs_edges)
    cost = 0
    cnt = 0
    while cnt < len(vertexs)-1:
        weight, v1, v2 = heappop(adjacent_vertexs_edges)
        if v2 not in visited:
            visited.add(v2)
            minu_tree.append((weight, v1, v2))
            cost += weight
            cnt += 1
            for next_edge in adjacent_dict[v2]:
                if next_edge[2] not in visited:
                    heappush(adjacent_vertexs_edges, next_edge)
    return cost

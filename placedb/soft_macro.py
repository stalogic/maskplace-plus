from itertools import combinations
import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.sparse import csr_matrix
import time
import math
import hashlib
import os, pickle, gzip
import pathlib

from placedb import PlaceDB, LefDefReader, LefDefPlaceDB
from copy import deepcopy
def graph_hexdigest(G:nx.Graph) -> str:
    t0 = time.time()
    G_pickled = pickle.dumps(G)
    md5_digest = hashlib.md5(G_pickled).hexdigest()
    print(f"Time used for graph hexdigest: {time.time() - t0}s")
    return md5_digest

def placedb_to_graph(placedb: PlaceDB) -> nx.Graph:
    G = nx.Graph()

    for net_name in placedb.net_info:
        source_info = placedb.net_info[net_name]['source']
        if 'node_type' not in source_info:
            print(f"{net_name=} is empty, skip")
            continue
        if source_info['node_type'] == 'PIN':
            print(f"{net_name=} Input is PIN, skip")
            continue

        net_nodes = placedb.net_info[net_name]['nodes'].values()
        for node1, node2 in combinations(net_nodes, 2):
            node_name1 = node1['key']
            node_name2 = node2['key']

            if node_name1 in placedb.macro_info or node_name2 in placedb.macro_info:
                continue

            if node_name1 in placedb.port_info or node_name2 in placedb.port_info:
                assert False

            if node_name1 not in G:
                node_info = placedb.cell_info[node_name1]
                width = node_info['width']
                height= node_info['height']
                G.add_node(node_name1, area = width * height)

            if node_name2 not in G:
                node_info = placedb.cell_info[node_name2]
                width = node_info['width']
                height= node_info['height']
                G.add_node(node_name2, area = width * height)

            pin1 = node1['pin_name']
            pin2 = node2['pin_name']

            G.add_edge(node_name1, node_name2, net_name=net_name, pin1=pin1, pin2=pin2)

    return G


def spectral_graph_partition(G:nx.Graph, base_area:float, min_ratio=0.1, max_ratio=1.5, cache_root:str='./cache') -> list[nx.Graph]:
    
    graph_digest = graph_hexdigest(G)
    cache_path = pathlib.Path(cache_root)
    graph_partition_result = cache_path / f"{graph_digest}_gp.pkl.gz"
    print(f"graph partition cache: {graph_partition_result.name}")

    graph_partition_result = pathlib.Path('cache/078d528292661958d4463ba785f5534c_gp.pkl.gz')
    
    if graph_partition_result.exists():
        with gzip.open(graph_partition_result, 'rb') as f:
            subgraphs, discards = pickle.load(f)
        print(f"read graph partition result from {graph_partition_result}")
    else:
        # 使用谱聚类算法
        sc = SpectralClustering(2, affinity='precomputed', random_state=0)
        print(f"{' start graph partition ':#^80}")
        subgraphs, discards = graph_partition(sc, G, base_area, min_ratio, max_ratio)
        print(f"{' finish graph partition ':#^80}")
        with gzip.open(graph_partition_result, 'wb') as f:
            pickle.dump((subgraphs, discards), f)
        print(f"save graph partition result into {graph_partition_result}")
    
    subgraph_node, subgraph_edge, subgraph_area = 0, 0, 0
    for graph in subgraphs:
        subgraph_node += len(graph.nodes)
        subgraph_edge += len(graph.edges)
        subgraph_area += sum([node['area'] for _, node in graph.nodes(data=True)])

    print(f"Subgraph# node: {subgraph_node:.3e}, area:{subgraph_area:.3e}, edge: {subgraph_edge:.3e}")

    discard_node, discard_edge, discard_area = 0, 0, 0
    for graph in discards:
        discard_node += len(graph.nodes)
        discard_edge += len(graph.edges)
        discard_area += sum([node['area'] for _, node in graph.nodes(data=True)])
    print(f"Discards# node: {discard_node:.3e}, area:{discard_area:.3e}, edge: {discard_edge:.3e}")

    original_area = sum([node['area'] for _, node in G.nodes(data=True)])
    print(f"Original# node: {len(G.nodes):.3e}, area:{original_area:.3e}, edge: {len(G.edges):.3e}")
    return subgraphs


def graph_partition(sc:SpectralClustering, G:nx.Graph, base_area:float, min_ratio=0.1, max_ratio=1.5) -> tuple[list[nx.Graph], list[nx.Graph]]:
    
    # 获取图的邻接矩阵
    adj_matrix: np.ndarray = nx.to_numpy_array(G, dtype='bool')
    sprase_matrix = csr_matrix(adj_matrix)
    labels = sc.fit_predict(sprase_matrix)
    subgraphs = []
    discards = []
    num_clusters = 2
    for i in range(num_clusters):
        subgraph_nodes = [node for node, label in zip(G.nodes(), labels) if label == i]
        subgraph :nx.Graph = G.subgraph(subgraph_nodes)
        subgraph_area = sum([node['area'] for _, node in subgraph.nodes(data=True)])
        print(f"sub graph# node:{len(subgraph.nodes)}, edge: {len(subgraph.edges)}, area: {subgraph_area:.3e}")

        if subgraph_area > max_ratio * base_area:
            graphs, discard_graphs = graph_partition(sc, subgraph, base_area, min_ratio, max_ratio)
            subgraphs.extend(graphs)
            discards.extend(discard_graphs)
        elif min_ratio is not None and subgraph_area < min_ratio * base_area:
            discards.append(subgraph)
        else:
            subgraphs.append(subgraph)
    return subgraphs, discards


def stoer_wagner_graph_partition(G:nx.Graph, base_area:float, min_ratio=0.1, max_ratio=1.5, cache_root:str='./cache'):
    # graph_digest = graph_hexdigest(G)
    # cache_path = pathlib.Path(cache_root)
    # graph_partition_result = cache_path / f"{graph_digest}_gp.pkl.gz"
    
    # if graph_partition_result.exists():
    #     with gzip.open(graph_partition_result, 'rb') as f:
    #         subgraphs, discards = pickle.load(f)
    #     print(f"read graph partition result from {graph_partition_result}")
    # else:
    #     # 使用谱聚类算法
    #     sc = SpectralClustering(2, affinity='precomputed', random_state=0)
    #     print(f"{' start graph partition ':#^80}")
    #     subgraphs, discards = graph_partition(sc, G, base_area, min_ratio, max_ratio)
    #     print(f"{' finish graph partition ':#^80}")
    #     with gzip.open(graph_partition_result, 'wb') as f:
    #         pickle.dump((subgraphs, discards), f)
    #     print(f"save graph partition result into {graph_partition_result}")
    t0 = time.time()
    cut_value, partition = nx.stoer_wagner(G)
    print(f"{partition=}")
    print(f"{cut_value=}")
    print(f"{time.time() - t0}s")
    exit()
    
    subgraph_node, subgraph_edge, subgraph_area = 0, 0, 0
    for graph in subgraphs:
        subgraph_node += len(graph.nodes)
        subgraph_edge += len(graph.edges)
        subgraph_area += sum([node['area'] for _, node in graph.nodes(data=True)])

    print(f"Subgraph# node: {subgraph_node:.3e}, area:{subgraph_area:.3e}, edge: {subgraph_edge:.3e}")

    discard_node, discard_edge, discard_area = 0, 0, 0
    for graph in discards:
        discard_node += len(graph.nodes)
        discard_edge += len(graph.edges)
        discard_area += sum([node['area'] for _, node in graph.nodes(data=True)])
    print(f"Discards# node: {discard_node:.3e}, area:{discard_area:.3e}, edge: {discard_edge:.3e}")

    original_area = sum([node['area'] for _, node in G.nodes(data=True)])
    print(f"Original# node: {len(G.nodes):.3e}, area:{original_area:.3e}, edge: {len(G.edges):.3e}")
    return subgraphs

def convert_to_soft_macro_placedb(placedb:PlaceDB, parser:LefDefReader, gamma:float=1.1) -> PlaceDB:
    """
    将placedb中的std cell，使用谱聚类算法，将其划分为soft macro。
    并使用soft macro替代std cell，生成新的placedb。
    gamma 表示创建soft macro时，正方形边长缩放比例。
    """
    
    # 标准单元聚类
    start_time = time.time()
    base_area = sum([macro['width'] * macro['height'] for macro in placedb.hard_macro_info.values()]) / len(placedb.hard_macro_info)
    G = placedb_to_graph(placedb)
    print(f"{len(G.nodes)=:*^100}")
    print(f"{len(G.edges)=:*^100}")
    # stoer_wagner_graph_partition(G, base_area)
    subgraph_list = spectral_graph_partition(G, base_area)
    total_nodes, total_edges = 0, 0
    for i, subgraph in enumerate(subgraph_list):
        area = sum([attr['area'] for _, attr in subgraph.nodes(data=True)])
        print(f"Subgraph {i}: nodes = {len(subgraph.nodes())} edges = {len(subgraph.edges())} area = {area:.3e} area ratio: {area/base_area:.3f}")
        total_nodes += len(subgraph.nodes())
        total_edges += len(subgraph.edges())
    print(f"Time used for graph cluster: {time.time() - start_time:.1f}")

    # 生成新的place_instance_dict
    place_instance_dict = {}
    node2soft_macro = {}
    soft_macro_list = []
    for i, subgraph in enumerate(subgraph_list):
        area = sum([attr['area'] for _, attr in subgraph.nodes(data=True)])
        macro_name = f'virtual/soft_macro_{i}'
        macro = {
            'type': f'soft_macro_{i}',
            'coordinate': (0, 0),
            'size': [round(math.sqrt(area) * gamma), round(math.sqrt(area) * gamma)],
            'orientation': 'N',
            'attributes': 'VIRTUAL',
        }
        place_instance_dict[macro_name] = macro
        soft_macro_list.append(macro_name)
        for node in subgraph.nodes():
            node2soft_macro[node] = macro_name

    for key in parser.place_instance_dict:
        node_attribute = parser.place_instance_dict[key]['attributes']

        if node_attribute == 'PLACED':
            # remove all std cell
            continue
        elif node_attribute == 'FIXED':
            # keep all macro
            place_instance_dict[key] = parser.place_instance_dict[key]
        else:
            raise ValueError(f"Unknown node attribute: {node_attribute}")

    # 生成新的place_net_dict
    place_net_dict = {}
    for net in parser.place_net_dict:
        node_set = set()
        nodes = []
        for node, pin in parser.place_net_dict[net]['nodes']:
            if node == 'PIN':
                nodes.append((node, pin))
            elif node in node2soft_macro:
                # net中，属于同一个soft_macro的node，只保留一次，并使用soft_macro代替
                soft_macro = node2soft_macro[node]
                if soft_macro not in node_set:
                    nodes.append((soft_macro, 'VPIN'))
                    node_set.add(soft_macro)
            elif node in place_instance_dict:
                nodes.append((node, pin))
            else:
                # discard node
                pass

        if len(nodes) >= 2:
            place_net_dict[net] = {
                'id': len(place_net_dict),
                'key': net,
                'nodes': nodes
            }

    # 生成新的place_pin_dict
    lef_dict = {}
    for key in parser.lef_dict:
        lef_dict[key] = parser.lef_dict[key]

    # 添加soft_macro的信息，主要是VPIN
    for soft_macro in soft_macro_list:
        macro_info = place_instance_dict[soft_macro]
        soft_macro_type = macro_info['type']
        lef_dict[soft_macro_type] = {
            'key': soft_macro_type,
            'size': macro_info['size'],
            'pin': {
                'VPIN': {
                    'key': 'VPIN',
                    'direction': 'VIRTUAL',
                    'rect_left': 0,
                    'rect_lower': 0,
                    'rect_right': macro_info['size'][0],
                    'rect_upper': macro_info['size'][1]
                }
            }
        }

    # 生成新的PlaceDB
    new_place_db = LefDefPlaceDB(place_net_dict=place_net_dict,
                                    place_instance_dict=place_instance_dict,
                                    place_pin_dict=parser.place_pin_dict,
                                    lef_dict=lef_dict,
                                    die_area=parser.die_area,)
    return new_place_db
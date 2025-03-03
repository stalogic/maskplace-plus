import os
from placedb import PlaceDB
from prim import prim_real

def comp_res(placedb :PlaceDB, node_pos, ratio):

    hpwl = 0.0
    cost = 0.0
    for net_name in placedb.net_info:
        max_height = max(placedb.canvas_size)
        max_x = 0.0
        min_x = max_height * 1.1
        max_y = 0.0
        min_y = max_height * 1.1

        for node_name in placedb.net_info[net_name]["nodes"]:
            if node_name not in node_pos:
                continue
            pin_x = node_pos[node_name][0] * ratio + placedb.net_info[net_name]["nodes"][node_name]["pin_offset"][0]
            pin_y = node_pos[node_name][1] * ratio + placedb.net_info[net_name]["nodes"][node_name]["pin_offset"][1]
            max_x = max(pin_x, max_x)
            min_x = min(pin_x, min_x)
            max_y = max(pin_y, max_y)
            min_y = min(pin_y, min_y)

        if os.getenv('PLACEENV_IGNORE_PORT', '0') != '1':
            for port_info in placedb.port_info[net_name]["ports"].values():
                pin_x, pin_y = port_info["pin_offset"]

                max_x = max(pin_x, max_x)
                min_x = min(pin_x, min_x)
                max_y = max(pin_y, max_y)
                min_y = min(pin_y, min_y)

        if min_x <= max_height:
            hpwl_tmp = (max_x - min_x) + (max_y - min_y)
        else:
            hpwl_tmp = 0
        if "weight" in placedb.net_info[net_name]:
            hpwl_tmp *= placedb.net_info[net_name]["weight"]
        hpwl += hpwl_tmp

        net_node_set = set(placedb.net_info[net_name]["nodes"])
        if os.getenv('PLACEENV_IGNORE_PORT', '0') != '1':
            net_node_set |= set(placedb.net_info[net_name]["ports"])
            
        for net_node in list(net_node_set):
            if net_node not in node_pos and net_node not in placedb.port_info:
                net_node_set.discard(net_node)

        prim_cost = prim_real(net_node_set, node_pos, placedb.net_info[net_name], ratio)
        if "weight" in placedb.net_info[net_name]:
            prim_cost *= placedb.net_info[net_name]["weight"]
        assert hpwl_tmp <= prim_cost +1e-5
        cost += prim_cost
    return hpwl, cost

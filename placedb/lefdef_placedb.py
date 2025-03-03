import os

from functools import cached_property
from placedb import PlaceDB

class LefDefPlaceDB(PlaceDB):

    def __init__(self, place_net_dict, place_instance_dict, place_pin_dict, lef_dict, die_area):
        self.place_net_dict = place_net_dict
        self.place_instance_dict = place_instance_dict
        self.place_pin_dict = place_pin_dict
        self.lef_dict = lef_dict
        self.die_area = die_area

    @cached_property
    def macro_info(self) -> dict[str, dict[str, object]]:
        macro_info = {}
        macro_info.update(self.hard_macro_info)
        macro_info.update(self.soft_macro_info)
        return macro_info
    
    @cached_property
    def hard_macro_info(self) -> dict[str, dict[str, object]]:
        hard_macro_info = {}
        for key in self.place_instance_dict.keys():
            if self.place_instance_dict[key]['attributes'] == 'FIXED':
                hard_macro_info[key] = {
                    'type': self.place_instance_dict[key]['type'],
                    'width': self.place_instance_dict[key]['size'][0],
                    'height': self.place_instance_dict[key]['size'][1]
                }
        return hard_macro_info
    
    @cached_property
    def soft_macro_info(self) -> dict[str, dict[str, object]]:
        soft_macro_info = {}
        for key in self.place_instance_dict.keys():
            if self.place_instance_dict[key]['attributes'] == 'VIRTUAL':
                soft_macro_info[key] = {
                    'type': self.place_instance_dict[key]['type'],
                    'width': self.place_instance_dict[key]['size'][0],
                    'height': self.place_instance_dict[key]['size'][1]
                }
        return soft_macro_info

    @cached_property
    def cell_info(self) -> dict[str, dict[str, object]]:
        cell_info = {}
        for key in self.place_instance_dict.keys():
            if self.place_instance_dict[key]['attributes'] == 'PLACED':
                cell_info[key] = {
                    'type': self.place_instance_dict[key]['type'],
                    'width': self.place_instance_dict[key]['size'][0],
                    'height': self.place_instance_dict[key]['size'][1]
                }
        return cell_info
    
    @cached_property
    def node_info(self) -> dict[str, dict[str, object]]:
        node_info = {}
        for key in self.place_instance_dict.keys():
            node_info[key] = {
                'type': self.place_instance_dict[key]['type'],
                'width': self.place_instance_dict[key]['size'][0],
                'height': self.place_instance_dict[key]['size'][1]
            }
        return node_info
    
    
    @cached_property
    def net_info(self) -> dict[str, dict[str, object]]:
        net_info :dict[str, dict[str, object]] = {}
        for net_name in self.place_net_dict:
            net_info[net_name] = {
                    'id': self.place_net_dict[net_name]['id'],
                    'key': net_name,
                    'source': {},
                    'nodes': {},
                    'ports': {}
                    }
            for node_name, pin_name in self.place_net_dict[net_name]['nodes']:
                if node_name == 'PIN':
                    pin_offset = self.place_pin_dict[pin_name]['coordinate']
                    pin_direction = self.place_pin_dict[pin_name]['direction']
                    if pin_direction == 'INPUT':
                        if os.getenv('NET_SOURCE_CHECK', '1') != '0':
                            assert len(net_info[net_name]['source']) == 0, "net has more then one input pin, set Environment Variable `NET_SOURCE_CHECK=0` to disable this check"
                        net_info[net_name]['source']['node_name'] = pin_name
                        net_info[net_name]['source']['node_type'] = 'PIN'
                        net_info[net_name]['source']['pin_name'] = pin_name
                    node_info = {
                        'key': pin_name,
                        'pin_name': pin_name,
                        'node_type': 'PIN',
                        'pin_offset': pin_offset
                    }
                    net_info[net_name]['ports'][pin_name] = node_info
                else:
                    node_type = self.place_instance_dict[node_name]['type']
                    pin_info = self.lef_dict[node_type]['pin'][pin_name]
                    pin_offset_x = (pin_info['rect_left'] + pin_info['rect_right']) / 2
                    pin_offset_y = (pin_info['rect_lower'] + pin_info['rect_upper']) / 2
                    pin_offset = (pin_offset_x, pin_offset_y)

                    pin_direction = pin_info['direction']
                    if pin_direction == 'OUTPUT':
                        if os.getenv('NET_SOURCE_CHECK', '1') != '0':
                            assert len(net_info[net_name]['source']) == 0, "net has more then one input pin, set Environment Variable `NET_SOURCE_CHECK=0` to disable this check"
                        net_info[net_name]['source']['node_name'] = node_name
                        net_info[net_name]['source']['node_type'] = node_type
                        net_info[net_name]['source']['pin_name'] = pin_name

                    node_info = {
                        'key': node_name,
                        'pin_name': pin_name,
                        'node_type': node_type,
                        'pin_offset': pin_offset
                    }
                    net_info[net_name]['nodes'][node_name] = node_info

        return net_info
    
    @cached_property
    def port_info(self) -> dict[str, dict[str, object]]:
        port_info = {}
        for key in self.place_pin_dict:
            port_info[key] = {
                'direction': self.place_pin_dict[key]['direction'],
                'orientation': self.place_pin_dict[key]['orientation'],
                'coordinate': self.place_pin_dict[key]['coordinate']
            }
        return port_info

    @cached_property
    def node2net_dict(self) -> dict[str, set[str]]:
        node2net_dict :dict[str, set[str]] = {}
        for net_name in self.net_info:
            for node in self.net_info[net_name]['nodes'].values():
                node_name = node['key']
                if node_name not in node2net_dict:
                    node2net_dict[node_name] = set()
                node2net_dict[node_name].add(net_name)
        return node2net_dict

    @cached_property
    def port2net_dict(self) -> dict[str, set[str]]:
        port2net_dict :dict[str, set[str]] = {}
        for net_name in self.net_info:
            for node in self.net_info[net_name]['ports'].values():
                port_name = node['key']
                if port_name not in port2net_dict:
                    port2net_dict[port_name] = set()
                port2net_dict[port_name].add(net_name)
        return port2net_dict


    @property
    def canvas_size(self):
        return self.die_area
    
    @cached_property
    def macro_place_queue(self) -> list[str]:
        macro_net_num = {}
        for macro_name in self.macro_info:
            macro_net_num[macro_name] = len(self.node2net_dict[macro_name])
        max_net_num = max(macro_net_num.values())
        for macro_name in self.macro_info:
            macro_net_num[macro_name] /= max_net_num

        macro_area = {}
        for macro_name in self.macro_info:
            macro_area[macro_name] = self.macro_info[macro_name]['width'] * self.macro_info[macro_name]['height']
        max_macro_area = max(macro_area.values())
        for macro_name in self.macro_info:
            macro_area[macro_name] /= max_macro_area

        macro_place_priority = {}
        for macro_name in self.macro_info:
            macro_place_priority[macro_name] = macro_net_num[macro_name] * macro_area[macro_name] + int(hash(macro_name)%10000)*1e-6

        macro_place_queue = sorted(macro_place_priority, key = lambda x: macro_place_priority[x], reverse = True)

        return macro_place_queue


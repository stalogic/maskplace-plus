from abc import ABC, abstractmethod

class PlaceDB(ABC):

    @property
    @abstractmethod
    def macro_info(self) -> dict[str, dict[str, object]]:
        """
        Return the macro information in the form of a dictionary.
        Return:
            {
                macro_name: 
                    {
                        type: str,
                        width: float|int,
                        height: float|int,
                    },
            }
        """
        pass

    @property
    @abstractmethod
    def hard_macro_info(self) -> dict[str, dict[str, object]]:
        """
        Return the hard macro information in the form of a dictionary.
        Return:
            {
                macro_name: 
                    {
                        type: str,
                        width: float|int,
                        height: float|int,
                    },
            }
        """
        pass

    @property
    @abstractmethod
    def soft_macro_info(self) -> dict[str, dict[str, object]]:
        """
        Return the soft macro information in the form of a dictionary.
        Return:
            {
                macro_name: 
                    {
                        type: str,
                        width: float|int,
                        height: float|int,
                    },
            }
        """
        pass

    @property
    @abstractmethod
    def cell_info(self) -> dict[str, dict[str, object]]:
        """
        Return the cell information in the form of a dictionary.
        Return:
            {
                cell_name: 
                    {
                        type: str,
                        width: float|int,
                        height: float|int,
                    },
            }
        """
        pass

    @property
    @abstractmethod
    def node_info(self) -> dict[str, dict[str, object]]:
        """
        Return the node information in the form of a dictionary.
        Return:
            {
                node_name: 
                    {
                        type: str,
                        width: float|int,
                        height: float|int,
                    },
            }
        """
        pass

    @property
    @abstractmethod
    def net_info(self) -> dict[str, dict[str, object]]:
        """
        Return the net information in the form of a dictionary.
        Return:
            {
                net_name: {
                    id: int,
                    key: net_name,
                    source: {
                        node_name: str,
                        node_type: str
                    },
                    nodes: {
                        node_name: {
                            key: node_name,
                            pin_name: str, 
                            node_type: str, 
                            pin_offset: tuple[float, float]
                        },
                    },
                    ports: {
                        port_name: {
                            key: port_name,
                            pin_name: str, 
                            node_type: str, 
                            pin_offset: tuple[float, float]
                        }
                    }
                }
            }
        """
        pass

    @property
    @abstractmethod
    def port_info(self) -> dict[str, dict[str, object]]:
        """
        Return the port information in the form of a dictionary.
        Return:
            {
                port_name: 
                    {
                        direction: str, 
                        orientation: str, 
                        coordinate: tuple[float, float]
                    },
            }
        """


    @property
    @abstractmethod
    def node2net_dict(self) -> dict[str, set[str]]:
        """
        Return the node2net dictionary in the form of a dictionary.
        Return:
            {
                node_name: [net_name1, net_name2, ...],
            }
        """
        pass

    @property
    @abstractmethod
    def port2net_dict(self) -> dict[str, set[str]]:
        """
        Return the port2net dictionary in the form of a dictionary.
        Return:
            {
                port_name: [net_name1, net_name2, ...],
            }
        """
        pass

    @property
    @abstractmethod
    def canvas_size(self) -> tuple[int, int]:
        """
        Return the canvas size in the form of a tuple.
        Return:
            (width, height)
        """
        pass

    @property
    @abstractmethod
    def macro_place_queue(self) -> list[str]:
        """
        Return the macro placement queue in the form of a list.
        place env should place the macro in the order of the list.
        """
        pass

if __name__ == '__main__':
    placedb = PlaceDB()

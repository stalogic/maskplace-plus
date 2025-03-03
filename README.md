

可用的环境变量

- NET_SOURCE_CHECK
    创建PlaceDB实例时是否校验每个net只有一个输入引脚，通过设置`NET_SOURCE_CHECK=0`关闭校验
    具体逻辑见`lefdef_placedb.py:net_info`

- PLACEENV_IGNORE_PORT
    PlaceEnv在reset时，是否导入芯片的引脚位置，如果导入芯片引脚位置，后续计算HPWL时，会计算相应的线长。通过设置`PLACEENV_IGNORE_PORT=1`关闭
    具体见`place_env.py:reset`
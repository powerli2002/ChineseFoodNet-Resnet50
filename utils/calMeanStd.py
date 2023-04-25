import re
import pandas as pd
import numpy as np


def getData(line):
    p = r".+\[(.+)], std=\[(.+)]"
    m = re.match(p, line)
    return m.group(1).replace(" ", "").split(","), m.group(2).replace(" ", "").split(",")


if __name__ == '__main__':
    # line = '[0/283]: mean=[0.557927668094635, 0.43669748306274414, 0.2793944478034973], std=[0.24065925180912018, 0.24462337791919708, 0.2507200837135315]'
    # data = getData(line)
    # for i in data[0]:
    #     print(i)
    # a = float(data[0][1]) + float(data[0][1])
    # print(a)
    mean_list = []
    std_list = []
    f = open('std.txt')
    for line in f:
        m_list = []
        s_list = []
        data = getData(line)
        for m in data[0]:
            m_list.append(float(m))
        for s in data[1]:
            s_list.append(float(s))
        mean_list.append(m_list)
        std_list.append(s_list)
        print(line)
    mean_array = np.array(mean_list)
    std_array = np.array(std_list)
    print(mean_array.mean(axis=0))
    print(std_array.mean(axis=0))
import matplotlib.pyplot as plt
import numpy as np

x = [0, 1, 2, 3, 4, 5]
y = [0, 2, 9, 10, 6, 3]
x.append(10)

import csv
f = open('test.csv','w')
writer = csv.writer(f)
writer.writerow(x)
f.close()
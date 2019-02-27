import csv
import numpy as np
import pandas as pd

def stat(a):
    sum = 0
    count = 0
    for da in a:
        sum = sum + da
        count = count + 1
    aver = sum / count
    sqsum = 0
    for da in a:
        sqsum = sqsum + (da - aver) ** 2
    sdev = (sqsum / count) ** 0.5
    return aver,sdev

filename = 'housing.csv'
df = pd.DataFrame(columns = ["area", "room", "price"])
with open(filename) as f:
    reader = csv.reader(f)
    for row in reader:
        d = np.array(row,dtype=np.float)
        df.loc[df.shape[0] + 1] = d
#print(df)
area = df['area']
room = df['room']
av,sd = stat(area)
print()

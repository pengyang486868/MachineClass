import csv
import numpy as np
import pandas as pd

global columnNames

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

def read():    
    filename = 'housing.csv'
    df = pd.DataFrame(columns=columnNames)
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            d = np.array(row,dtype=np.float)
            df.loc[df.shape[0] + 1] = d
    return df

def write(df):
    filename = 'housing.csv'
    df = pd.DataFrame(columns=columnNames)
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            d = np.array(row,dtype=np.float)
            df.loc[df.shape[0] + 1] = d
    return df

columnNames = ["area", "room", "price"]
rawdata = read()
#print(df)
area = rawdata['area']
room = rawdata['room']
av,sd = stat(area)
print()

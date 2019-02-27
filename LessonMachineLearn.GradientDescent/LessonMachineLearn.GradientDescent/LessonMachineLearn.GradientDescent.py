import csv
import numpy as np
import pandas as pd

filename = 'housing.csv'
df = pd.DataFrame(columns = ["area", "room", "price"])
with open(filename) as f:
    reader = csv.reader(f)
    for row in reader:
        d = np.array(row,dtype=np.float)
        df.loc[df.shape[0] + 1] = d
#print(df)
col=df['area']
sum=0
for da in col:
    sum=sum+da
print()

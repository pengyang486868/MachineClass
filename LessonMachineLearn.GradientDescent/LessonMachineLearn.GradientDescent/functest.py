import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

global columnNames

def read():    
    #filename = 'housing.txt'
    df = pd.DataFrame(columns=columnNames)
    
    d = np.array([1600,1770,3,330])
    df.loc[1] = d
    d = np.array([2400,2740,3,369])
    df.loc[2] = d
    d = np.array([1416,1634,2,232])
    df.loc[3] = d
    d = np.array([3000,3412,4,540])
    df.loc[4] = d
    return df

def write(df,filename):
    df.to_csv(filename)

def model(w0,w1,w2,w3,x1,x2,x3):
    return w0 + w1 * x1 + w2 * x2

def lossfunc(w0,w1,w2,w3,x1,x2,x3,y):
    list = (model(w0,w1,w2,w3,x1,x2,x3) - y) ** 2
    sum = 0
    count = 0
    for da in list:
        sum = sum + da
        count = count + 1
    return sum / count / 2

def gradientcoef(w0,w1,w2,w3,x1,x2,x3,y,xcurrent):
    list = (model(w0,w1,w2,w3,x1,x2,x3) - y) * xcurrent
    sum = 0
    count = 0
    for da in list:
        sum = sum + da
        count = count + 1
    return sum / count


#-------------------------------------------
# main
#-------------------------------------------
columnNames = ["area","size", "room", "price"]
rawdata = read()
trainsize = rawdata.shape[0]

area = rawdata['area']
size = rawdata['size']
room = rawdata['room']
price = rawdata['price']

w0,w1,w2,w3 = 0,0,0,0
rate = 0.1#0.00000001
iters = 80

start_time = time.time()

for i in range(iters):
    jw = lossfunc(w0,w1,w2,w3,area,size,room,price) # comment when timing
    w0 = w0 - rate * gradientcoef(w0,w1,w2,w3,area,size,room,price,1)
    w1 = w1 - rate * gradientcoef(w0,w1,w2,w3,area,size,room,price,area)
    w2 = w2 - rate * gradientcoef(w0,w1,w2,w3,area,size,room,price,size)
    w3 = w3 - rate * gradientcoef(w0,w1,w2,w3,area,size,room,price,room)
    print(jw) # comment when timing
end_time = time.time()

answer = w0 + w1 * area + w2 * size + w3 * room
plt.plot(range(answer.shape[0]),answer.sort_values())
plt.plot(range(price.shape[0]),price.sort_values())
#plt.scatter(room,answer)
#plt.scatter(room,price)
plt.show()

predict_area = 2650
predict_size = 3000
predict_room = 4
predict_price = model(w0,w1,w2,w3,predict_area,predict_size,predict_room)

#Problem 2C output:423554
print(predict_price)
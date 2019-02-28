import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    filename = 'housing.txt'
    df = pd.DataFrame(columns=columnNames)
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            d = np.array(row,dtype=np.float)
            df.loc[df.shape[0] + 1] = d
    return df

def write(df,filename):
    df.to_csv(filename)

def model(w0,w1,w2,x1,x2):
    return w0 + w1 * x1 + w2 * x2

def lossfunc(w0,w1,w2,x1,x2,y):
    list = (model(w0,w1,w2,x1,x2) - y) ** 2
    sum = 0
    count = 0
    for da in list:
        sum = sum + da
        count = count + 1
    return sum / count / 2

def gradientcoef(w0,w1,w2,x1,x2,y,xcurrent):
    list = (model(w0,w1,w2,x1,x2) - y) * xcurrent
    sum = 0
    count = 0
    for da in list:
        sum = sum + da
        count = count + 1
    return sum / count


#-------------------------------------------
# main
#-------------------------------------------
columnNames = ["area", "room", "price"]
rawdata = read()
trainsize = rawdata.shape[0]

area = rawdata['area']
room = rawdata['room']
price = rawdata['price']
avarea,sdarea = stat(area)
avroom,sdroom = stat(room)
avprice,sdprice = stat(price)

#plt.plot(range(area.shape[0]),area)
#plt.show()
transdata = pd.DataFrame(columns=columnNames)
transdata['area'] = (area - avarea) / sdarea
transdata['room'] = (room - avroom) / sdroom
transdata['price'] = (price - avprice) / sdprice
#print(transdata)
write(transdata,'normalized.txt')

normalized = pd.read_csv('normalized.txt',usecols=columnNames)
#print(normalized)
area = normalized['area']
room = normalized['room']
price = normalized['price']

w0,w1,w2 = 0,0,0
rate = 1
iters = 100

for i in range(iters):
    jw = lossfunc(w0,w1,w2,area,room,price)
    w0 = w0 - rate * gradientcoef(w0,w1,w2,area,room,price,1)
    w1 = w1 - rate * gradientcoef(w0,w1,w2,area,room,price,area)
    w2 = w2 - rate * gradientcoef(w0,w1,w2,area,room,price,room)
    print(jw)

print(w0 + w1 * area + w2 * room - price)
answer = w0 + w1 * area + w2 * room
plt.plot(range(answer.shape[0]),answer.sort_values())
plt.plot(range(price.shape[0]),price.sort_values())
#plt.scatter(room,answer)
#plt.scatter(room,price)
plt.show()

predict_area = 2650
predict_room = 4
predict_area_norm = (predict_area - avarea) / sdarea
predict_room_norm = (predict_room - avroom) / sdroom
predict_price_norm = model(w0,w1,w2,predict_area_norm,predict_room_norm)
predict_price = predict_price_norm * sdprice + avprice

#Problem 2C output:423554
print(predict_price)
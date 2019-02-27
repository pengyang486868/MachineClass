import csv

filename = 'housing.csv'
with open(filename) as f:
    reader = csv.reader(f)
    for row in reader:
        for num in row:
            print(int(num))

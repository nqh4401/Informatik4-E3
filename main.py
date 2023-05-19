import pandas as pd
import matplotlib.pyplot as plt
import csv

data = pd.read_csv(r'C:\Users\Huy Nguyen\Downloads\archive\train.csv')


# y = B0 + x * B1
def regression_coefficients(B0_now, B1_now, points ,L ):
    B0_gradient = 0
    B1_gradient  = 0

    n = len(points)
    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y

        #print(x,y)

        B1_gradient += ( 1 / n ) * x * ( (B1_now * x + B0_now) - y )
        B0_gradient += ( 1 / n ) * ( (B1_now * x + B0_now) - y )

    B1 = B1_now - L * B1_gradient
    B0 = B0_now - L * B0_gradient

    return B1, B0

B1 = 1
B0 = 1
L = 0.001
n = 301


for i in range(n):
    B1, B0 = regression_coefficients(B0, B1, data, L)

print(B0, B1)

plt.scatter(data.x, data.y, color = "yellow",s = 1 )
plt.plot(data.x, data.x * B1 + B0, color = "red")
plt.show()
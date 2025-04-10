#Import thư viện
import numpy as np
from matplotlib import pyplot as plt
from GPSymbolicRegression import *

#Sinh data 
n = 100
x = np.linspace(0.1, 10, n)
y_smooth = x**3 + x**2 - 2*x + 1
random.seed(0)
s = np.random.normal(0, 50, n)
y = y_smooth + s

ax = plt.subplot()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.plot(x, y, 'o')

model = GeneticSymbolicRegressor(generations=2000)
model.fit(x, y)
y_predict = model.predict(x)

print(model.visualize_best_solution())
print(model.best_loss)
plt.plot(x, y, color='b', label='target', marker='o', linestyle='none')
plt.plot(x, y_predict, color='r', label='predict')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#Enter Value of Table
x = np.array([10.858,10.617,10.183,9.7003,9.652,10.086,9.459,8.3972,7.6251,7.1907,7.046,6.9494,6.7081,6.3221,6.0325,5.7429,5.5016,5.2603,5.1638,5.0673,4.9708,4.8743,4.7777,4.7295,4.633,4.4882,4.3917,4.2951,4.2469,4.0056,3.716,3.523,3.4265,3.3782,3.4265,3.3782,3.3299,3.3299,3.4265])
y = np.array([31.002,31.021,31.058,31.095,31.133,31.188,31.226,31.263,31.319,31.356,31.412,31.468,31.524,31.581,31.618,31.674,31.712,31.768,31.825,31.862,31.919,31.975,32.013,32.07,31.126,31.164,32.221,32.259,32.296,32.334,32.391,32.448,32.505,32.543,32.6,32.657,32.696,32.753,32.791])
#Linear Regration Process
linreg = LinearRegression()
x = x.reshape(-1,1)
linreg.fit(x,y)
y_predict = linreg.predict(x)
#Drive Graph
plt.scatter(x,y)
plt.plot(x, y_predict,color="red")
plt.show()

print(linreg.coef_[0])
print(linreg.intercept_)
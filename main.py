import numpy as np
import matplotlib.pyplot as plt
import math
import random as rd
import pandas as pd
import func

plt.style.use('Solarize_Light2')

x = [102, 80, 70, 40, 25, 15, 10]
y = [2.067, 1.84, 1.698, 1.298, 0.983, 0.82, 0.676]
x2 = [10.09950494, 8.94427191, 8.366600265, 6.32455532, 5, 3.872983346, 3.16227766]
x2d = pd.array(x2)
print(x2d)
xd = pd.array(x)
yd = pd.array(y)
print(yd)

coeff_2, coeff_1, y_int = func.regression_eqn(xd, yd, linear=False)
res, mres, resstd = func.residuals(xd, yd)
# print(y_int)
func.scatter_plot_er_2(xd, yd, coeff_2, coeff_1, y_int, resstd, xt='Length', yt='Period', title='Pendulum Lab Data')


coeff_21, y_int1 = func.regression_eqn(x2d, yd, linear=True)  # when linear only returns 2
res1, mres1, resstd1 = func.residuals(x2d, yd)

func.scatter_plot(x2d, yd, coeff_21[0], y_int1[0])  # xt='SQRT Length', yt='Period', title='Pendulum Lab Data Linearized')


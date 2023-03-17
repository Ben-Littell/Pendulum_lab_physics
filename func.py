import numpy as np
import matplotlib.pyplot as plt
import math
import random as rd
import pandas as pd

plt.style.use('seaborn')


def mean(data):
    total = sum(data)
    m = total / len(data)
    return m


def median(data):
    data.sort()
    if len(data) % 2 == 0:
        m = (data[len(data) // 2] + data[len(data) // 2 - 1]) / 2
    else:
        m = data[len(data) // 2]
    return m


def variance(data):
    new_list = [(val - mean(data)) ** 2 for val in data]
    v = mean(new_list)
    return v


def stand_dev(data):
    v = variance(data)
    s = math.sqrt(v)
    return s


def get_random(df, per):
    num_per = int(per/100 * len(df))
    training = df.sample(num_per)
    training = training.sort_index()
    test = df.drop(training.index)
    return training, test


# BI LINEAR REGRESSION EQUATION
def regression_eqn(ind_array, dep_array, linear=True):
    # input as two arrays or 2 columns of a DF
    x_4 = (ind_array**4).sum()
    x_3 = (ind_array**3).sum()
    x_2 = (ind_array**2).sum()
    x_1 = (ind_array).sum()
    n = len(ind_array)
    xy_2 = ((ind_array**2 * dep_array)).sum()
    xy = (ind_array * dep_array).sum()
    if linear:
        matrix1 = [[x_2, ind_array.sum()], [ind_array.sum(), n]]
        matrix2 = [[xy], [dep_array.sum()]]
        invarray1 = np.linalg.inv(matrix1)
        solution = np.dot(invarray1, matrix2)
        return solution
    else:
        matrix1 = [[x_4, x_3, x_2], [x_3, x_2, x_1], [x_2, x_1, n]]
        matrix2 = [[xy_2], [xy], [dep_array.sum()]]
        invarray1 = np.linalg.inv(matrix1)
        solution = np.dot(invarray1, matrix2)
        return solution[0][0], solution[1][0], solution[2][0]


def sigma_xy(xd, yd):
    nlist = []
    for i in range(len(xd)):
        nlist.append((xd[i] * yd[i]))
    return sum(nlist)


def least_sqrs(xd, yd):
    matrix1 = [[sum(val ** 2 for val in xd), sum(xd)], [sum(xd), len(xd)]]
    matrix2 = [sigma_xy(xd, yd), sum(yd)]
    array1 = np.array(matrix1)
    array2 = np.array(matrix2)
    invarray1 = np.linalg.inv(array1)
    solution = np.dot(invarray1, array2)
    return solution


def residuals(xd, yd, n=2):
    xdl = xd.tolist()
    ydl = yd.tolist()
    coeff2, coeff1, y_int = regression_eqn(xd, yd, linear=False)
    ys = [(coeff2 * (val**2)) + (coeff1*val) + y_int for val in xdl]
    r = [yd[n]-ys[n] for n in range(len(ydl))]
    mr = mean(r)
    stdr = stand_dev(r)
    return r, mr, stdr


def scatter_plot_er_2(data1, data2, coeff2, coeff1, y_int, std, title='Graph', xt='X', yt='Y', n=2):
    data1 = data1.tolist()
    data2 = data2.tolist()
    y_vals = []
    e1 = []
    e2 = []
    x_data = [min(data1), max(data1)]
    for val in range(len(data1)):
        ans = (coeff2 * (data1[val]**2)) + (coeff1*data1[val]) + y_int
        y_vals.append(ans)
    for val in range(len(data1)):
        ans = (coeff2 * (data1[val]**2)) + (coeff1*data1[val]) + y_int +(n*std)
        e1.append(ans)
    for val in range(len(data1)):
        ans = (coeff2 * (data1[val]**2)) + (coeff1*data1[val]) + y_int -(n*std)
        e2.append(ans)
    plt.plot(data1, y_vals, '-r')
    plt.plot(data1, e1, '--r')
    plt.plot(data1, e2, '--r')
    plt.scatter(data1, data2)
    plt.title(title)
    plt.xlabel(xt)
    plt.ylabel(yt)
    plt.text(x_data[1], y_vals[1], f'Y={round(coeff2, 5)}*X^2+{round(coeff1, 2)}X+{round(y_int, 2)}', color='g')
    plt.show()


def scatter_plot(data1, data2, slope, y_int):
    y_vals = []
    x_data = [min(data1), max(data1)]
    for val in range(2):
        ans = (slope * x_data[val]) + y_int
        y_vals.append(ans)
    plt.plot(x_data, y_vals, '-r')
    plt.scatter(data1, data2)
    plt.title('SCATTER PLOT')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.text(x_data[1], y_vals[1], f'Y={round(slope, 4)}*X+{round(y_int, 4)}', color='g')
    plt.show()

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def f1(t, x):
    return -x

def f2(t, x):
    return x

def real_x_finder1(x):
    return np.exp(-x)

def real_x_finder2(x):
    return np.exp(x)

def f3_1(t, x, y):
    return y

def f3_2(t, x, y):
    return -x

def euler_iteration(f, x, t, h):
    return x + h * f(t, x)

def euler_iteration_2d(t, x, y, h):
    return x + h * f3_1(t, x, y), y + h * f3_2(t, x, y)

def rk4_iteration(f, x, t, h):
    k1 = f(t, x)
    k2 = f(t + h/2, x + h/2 * k1)
    k3 = f(t + h/2, x + h/2 * k2)
    k4 = f(t + h, x + h * k3)
    k = 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x + h * k 
    
def rk4_iteration_2d(t, x, y, h):
    k1 = f3_1(t, x, y)
    m1 = f3_2(t, x, y)
    k2 = f3_1(t + h / 2, x + h * k1 / 2, y + h * m1 / 2)
    m2 = f3_2(t + h / 2, x + h * k1 / 2, y + h * m1 / 2)
    k3 = f3_1(t + h / 2, x + h * k2 / 2, y + h * m2 / 2)
    m3 = f3_2(t + h / 2, x + h * k2 / 2, y + h * m2 / 2)
    k4 = f3_1(t + h, x + h * k3, y + h * m3)
    m4 = f3_2(t + h, x + h * k3, y + h * m3)
    k = 1/6 * (k1 + 2 * k2 +2 * k3 + k4)
    m = 1/6 * (m1 + 2 * m2 + 2 *m3 + m4)
    return x + h * k, y + h * m

def euler(x0,t_array,h,f):
    x = np.zeros(len(t_array))
    x[0] = x0
    for i in range(1, len(t_array)):
        x[i] = euler_iteration(f, x[i-1], t_array[i-1], h)
    return x

def rk4(x0,t_array,h,f):
    x = np.zeros(len(t_array))
    x[0] = x0
    for i in range(1, len(t_array)):
        x[i] = rk4_iteration(f, x[i - 1], t_array[i - 1], h)
    return x

def err(x0, x, t, f_):
    x_real = np.zeros(len(t))
    for i in range(len(t)):
        x_real[i] = x0 * f_(t[i])
    delta = np.abs(x - x_real)
    return delta

def euler_2d(x0, y0, t, h):
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    x[0] = x0
    y[0] = y0
    for i in range(1, len(t)):
        x[i], y[i] = euler_iteration_2d(t[i-1], x[i-1], y[i-1], h)
    return x, y

def rk4_2d(x0, y0, t, h):
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    x[0] = x0
    y[0] = y0
    for i in range(1, len(t)):
        x[i], y[i] = rk4_iteration_2d(t[i-1], x[i-1], y[i-1], h)
    return x, y

def err_2d(x0, y0, x, y, t):
    real_x = np.zeros(len(t))
    real_y = np.zeros(len(t))
    for i in range(len(t)):
        real_x[i] = x0 * np.cos(t[i])
        real_y[i] = y0 * np.sin(t[i])
    delta_x = (real_x - x) ** 2
    delta_y = (real_y - y) ** 2
    delta = np.zeros(len(t))
    for i in range(len(t)):
        delta[i] = np.sqrt(delta_x[i] + delta_y[i])
    return delta

def painter(t, eu_err, rk4_err):
    
    plt.figure(num = "Эйлер Погрешности")
    plt.title ("Эйлер Погрешности")
    plt.xlabel("t")
    plt.ylabel("ошибка")
    plt.plot(t[0], eu_err[0], color='red')
    plt.plot(t[1], eu_err[1], color='green')
    plt.plot(t[2], eu_err[2], color='black')
    plt.plot(t[3], eu_err[3], color = 'blue')
    plt.legend(["h = 0.1", "h = 0.5", "h = 0.025", "h = 0.01"])
    plt.grid()

    plt.figure(num ="РГ погрешности")
    plt.title("РГ погрешности")
    plt.xlabel("t")
    plt.ylabel("ошибка")
    plt.plot(t[0], rk4_err[0], color='red')
    plt.plot(t[1], rk4_err[1], color='green')
    plt.plot(t[2], rk4_err[2], color = 'black')
    plt.plot(t[3], rk4_err[3], color = 'blue')
    plt.legend(["h = 0.1", "h = 0.5", "h = 0.025", "h = 0.01" ])
    plt.grid()
    
    i1 = np.arange(0, len(t[0]), 1)
    i2 = np.arange(0, len(t[1]), 1)
    i3 = np.arange(0, len(t[2]), 1)
    i4 = np.arange(0, len(t[3]), 1)

    plt.figure(num="Эйлер погрешности iter")
    plt.title ("Эйлер Погрешности в зависимости от итераций")
    plt.xlabel("t")
    plt.ylabel("ошибка")
    plt.plot(i1, eu_err[0], color='red')
    plt.plot(i2, eu_err[1], color='green')
    plt.plot(i3, eu_err[2], color='black')
    plt.plot(i4, eu_err[3], color = 'blue')
    plt.legend(["h = 0.1", "h = 0.5", "h = 0.025", "h = 0.01"])
    plt.xlim(0, 1000)
    plt.grid()
    
    plt.figure(num="РГ погрешности iter")
    plt.title("РГ погрешности в зависимости от итераций")
    plt.xlabel("Номер итерации")
    plt.ylabel("ошибка")
    plt.plot(i1, rk4_err[0], color='red')
    plt.plot(i2, rk4_err[1], color='green')
    plt.plot(i3, rk4_err[2], color = 'black')
    plt.plot(i4, rk4_err[3], color = 'blue')
    plt.legend(["h = 0.1", "h = 0.5", "h = 0.025", "h = 0.01"])
    plt.xlim(0, 1000)
    plt.grid()
   
    plt.show()
    
def main():
    #x(0) = 1
    x0 = 1
    y0 = 1
    #Выбор функций в зависимости от задания (1 или 2)
    funct = f2
    real_x_finder = real_x_finder2
    
    h=[0.1, 0.05, 0.025, 0.01] 
    t=[]
    euler_res=[]
    rk4_res=[]
    eu_err=[]
    rk4_err=[]
    
    for i in range (len(h)):
        t.append(np.arange(0, 10, h[i]))
        euler_res.append(euler(x0, t[i], h[i], funct))
        rk4_res.append(rk4(x0, t[i], h[i], funct))
        eu_err.append (err(x0, euler_res[i], t[i], real_x_finder))
        rk4_err.append(err(x0, rk4_res[i], t[i], real_x_finder))

    #Для задания #3
    
    eu_err_2d=[]
    rk4_err_2d=[]
        
    euler_x_res1, euler_y_res1 = euler_2d(x0, y0, t[0] , h[0])
    euler_x_res2, euler_y_res2 = euler_2d(x0, y0, t[1], h[1])
    euler_x_res3, euler_y_res3 = euler_2d(x0, y0, t[2], h[2])
    euler_x_res4, euler_y_res4 = euler_2d(x0, y0, t[3], h[3])

    rk4_x_res1, rk4_y_res1 = rk4_2d(x0, y0, t[0] , h[0])
    rk4_x_res2, rk4_y_res2 = rk4_2d(x0, y0, t[1], h[1])
    rk4_x_res3, rk4_y_res3 = rk4_2d(x0, y0, t[2], h[2])
    rk4_x_res4, rk4_y_res4 = rk4_2d(x0, y0, t[3], h[3])
        
    euler_x_res=[euler_x_res1,euler_x_res2,euler_x_res3,euler_x_res4]
    euler_y_res=[euler_y_res1,euler_y_res2,euler_y_res3,euler_y_res4]
    rk4_x_res=[rk4_x_res1,rk4_x_res2,rk4_x_res3,rk4_x_res4]
    rk4_y_res=[rk4_y_res1,rk4_y_res2,rk4_y_res3,rk4_y_res4]

    for i in range (len(h)):
        eu_err_2d.append (err_2d(x0, y0, euler_x_res[i], euler_y_res[i], t[i]))
        rk4_err_2d.append (err_2d(x0, y0, rk4_x_res[i], rk4_y_res[i], t[i]))

    #painter(t, eu_err, rk4_err)
    painter(t, eu_err_2d, rk4_err_2d)
    
if __name__ == "__main__":
    main()
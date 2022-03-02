# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt

def f(x, n, a):
    return a/(1+x**n) - x

def mu(x, n, a):
    return n * (1 - x / a)

def s2(x, n, a):
    return math.sqrt(mu(x,n,a)**2 - 1)

def tau(x, n, a, s2):
    return (1 / s2) * np.arccos(-1 / mu(x,n,a))

# #Методбисекции/дихотомии
def bisection_method(a, b, eps, n, alpha):
    res= []
    if f(a,n,alpha)*f(b,n,alpha) >0:
        print("Нет корней")
    else:
        iteration= 0
        midpoint = (a + b)/2.0
        while (iteration<20) and (abs(f(midpoint,n, alpha))>eps):
            midpoint = (a + b)/2.0
            if f(a,n,alpha)*f(midpoint,n,alpha) <0:
                b = midpoint
            else:
                a = midpoint

            iteration+= 1
            res.append(abs(f(midpoint,n, alpha)))
            x_res = midpoint
    return(midpoint, iteration, res, x_res)


def main():
    eps = 0.00001
    n=[2,4,6]
    alpha = np.arange(2.2,5.2,0.2)
    s2_list=[[],[],[]]
    tau_list=[[],[],[]]
    counter=0
    for i_n in n:
        for i_a in alpha:
            bi_res=bisection_method(0, 10, eps, i_n, i_a)
            s2_temp=s2(bi_res[3],i_n,i_a)
            s2_list[counter].append(s2_temp)
            tau_list[counter].append(tau(bi_res[3],i_n,i_a,s2_temp))
        counter+=1
  
    plt.plot(alpha, tau_list[0], '-', label = 'n=2' )
    plt.plot(alpha, tau_list[1], '-', label = 'n=4' )
    plt.plot(alpha, tau_list[2], '-', label = 'n=6' )
    plt.xlabel('alpha', fontsize=14, color='red')
    plt.ylabel('tau', fontsize=14, color='red')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
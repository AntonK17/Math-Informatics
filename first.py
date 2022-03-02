# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def f(x,n):
    return 10/(1+x**n) - x
def dfdx(x,n):
    return -10*n*x**(n-1)/(1+x**n)**2 - 1

#МетодНьютона
def Newton(x, eps, n):
    res = []
    differ = []
    f_= f(x,n)
    iteration = 0
    while (iteration<20) and (abs(f_)>eps):
        print('x=',x)   
        try:
            x = x - f_/dfdx(x,n)
        except ZeroDivisionError as err:
            print("Ошибка: {}".format(err))
            exit(1)
        f_ = f(x,n)
        res.append(x)
        iteration+= 1
        differ.append(abs(f_))
    return x, iteration, res, differ

#Методбисекции/дихотомии
def bisection_method(a, b, eps, n):
    res= []
    if f(a,n)*f(b,n) >0:
        print("Нет корней")
    else:
        iteration= 0
        midpoint = (a + b)/2.0
        while (iteration<20) and (abs(f(midpoint,n))>eps):
            midpoint = (a + b)/2.0
            if f(a,n)*f(midpoint,n) <0:
                b = midpoint
            else:
                a = midpoint

            iteration+= 1
            res.append(abs(f(midpoint,n)))
            x_res = midpoint
    return(midpoint, iteration, res, x_res)


def main():
    eps = 0.00001
    n=6
    # for i in range(-10,10):
    #     newton_res = Newton(i, eps,n)
    #     if newton_res[1]<20:
    #         plt.plot(range(0,newton_res[1]), newton_res[3], '-', label = 'Метод Ньютона для x0 =' + str(i) + ', x=' + str(newton_res[0]))

    bis_res= bisection_method(-10,10,eps,n)
    print(bis_res[1])
    plt.plot(range(0, bis_res[1]), bis_res[2], '-', label = 'Метод дихотомии x=' + str(bis_res[3]) )
    plt.title('Сходимость при n=' + str(n), fontsize=14)
    
    plt.xlabel('Число итераций', fontsize=14, color='red')
    plt.ylabel('Разность', fontsize=14, color='red')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()

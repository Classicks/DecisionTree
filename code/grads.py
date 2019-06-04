import numpy as np

def example(x):
    return np.sum(x**2)


def example_grad(x):
    return 2*x

def foo(x):
    result = 1
    λ = 4 # this is here to make sure you're using Python 3
    for x_i in x:
        result += x_i**λ
    return result

def foo_grad(x):
    x = 4 * np.power(x,3)
    return x

def bar(x):
    return np.prod(x)
    
def bar_grad(x):
    prod = np.prod(x)
    temp = x[0]
    x0 = prod / temp
    temp = x[1]
    x1 = prod / temp
    temp = x[2]
    x2 = prod / temp
    temp = x[3]
    x3 = prod / temp
    temp = x[4]
    x4 = prod / temp
    x = np.array([x0,x1,x2,x3,x4])
    return x

import numpy as np
import math

def ackley(inp : np.ndarray): # f(0,0) = 0  [-5, 5]*2
    x = inp[0]
    y = inp[1]

    return -20*math.exp(-0.2*math.sqrt(0.5*(x**2 + y**2))) - math.exp(0.5*(math.cos(2*math.pi*x) + math.cos(2*math.pi*y))) + math.e + 20


def rosenbrock(inp : np.ndarray): # f(1, .., 1) = 0,  [-inf, inf]*n
    s = 0
    for i in range(len(inp)-1):
        s += 100 * (inp[i+1] - inp[i]**2)**2 + (1-inp[i])**2
    return s

def beale(inp : np.ndarray): # f(3, 0.5) = 0, [-4.5, 4.5]*2
    x = inp[0]
    y = inp[1]

    return (1.5-x+x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 -x + x*y**3)**2

def goldstein(inp : np.ndarray): # f(0, -1) = 3, [-10, 10]
    x = inp[0]
    y = inp[1]

    return (1 + ((x+y+1)**2) * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * (30 + ((2*x -3*y)**2) * (18 - 32*x + 12 * x**2 + 48 * y - 36 * x*y + 27*y**2))

def bukin(inp : np.ndarray): # f(-10, 1) = 0,  [-15, -5]x[-3, 3]
    x = inp[0]
    y = inp[1]

    return 100 * math.sqrt(math.fabs(y-0.01*x**2)) + 0.01 * math.fabs(x+10)

def matyas(inp : np.ndarray): # f(0, 0) = 0, [-10, 10]
    x = inp[0]
    y = inp[1]

    return 0.26*(x**2 + y**2) - 0.48 * x * y


def levy(inp : np.ndarray): # f(1, 1) = 0, [-10, 10]
    x = inp[0]
    y = inp[1]

    return math.sin(3*math.pi*x)**2 + ((x-1)**2) * (1+math.sin(3*math.pi*y)**2) + ((y-1)**2) * (1 + math.sin(2*math.pi*y)**2)

def himmelblau(inp : np.ndarray): # f(3, 2) = f(-2.8051, 3.1314) = f(-3.7783, -3.2831) = f(3.5844, -1.8481) 0, [-5, 5]
    x = inp[0]
    y = inp[1]

    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def crosstray(inp : np.ndarray): # f(+-1.3494, +-1.3494) = -2.0626, [-10, 10]
    x = inp[0]
    y = inp[1]

    return -0.0001 * (math.fabs( math.sin(x)*math.sin(y)*math.exp(math.fabs(100-math.sqrt(x**2+y**2)/math.pi))) + 1 )**0.1

def eggholder(inp : np.ndarray): # f(512, 404.2319) = -959.6407
    x = inp[0]
    y = inp[1]

    return -(y+47) * math.sin(math.sqrt(math.fabs(x/2 + y + 47))) - x * math.sin(math.sqrt(math.fabs(x-(y+47))))

def schaffer(inp): # f(0, +-1.2531) = f(+-1.2531, 0) = 0.292579
    x = inp[0]
    y = inp[1]

    return 0.5 + (math.cos(math.sin(math.fabs(x**2 - y**2)))**2 - 0.5)/(1 + 0.001 * (x**2 + y**2))**2
 
def styblinkski(inp): # f(-2.9035, ..., -2.9035) = -39.16617   [-5, 5]
    s = 0
    for i in range(len(inp)):
        s += inp[i] ** 4 - 16 * inp[i]**2 + 5*inp[i]
    return (s/2)/len(inp)

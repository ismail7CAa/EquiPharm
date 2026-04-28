import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution
from scipy.interpolate import CubicSpline
# import matplotlib.pyplot as plt

def assym_sigmoid(x, b, c, d, e, f):
    return (c + ( (d - c) / (1 + np.exp(b * (np.log(x) - np.log(e))))**f ))


def bi_sigmoid(
    x,
    bottom,
    span,
    frac,
    logEC50_1,
    nH_1,
    logEC50_2,
    nH_2,
):
    """
    Bi-sigmoidal (double Hill) dose–response function.

    Parameters
    ----------
    x : array-like
        Independent variable (typically log10 concentration).
    bottom : float
        Minimum response.
    span : float
        Total dynamic range (top - bottom).
    frac : float
        Fraction of span assigned to the first sigmoid (0–1).
    logEC50_1 : float
        log10(EC50) of the first component.
    nH_1 : float
        Hill coefficient of the first component.
    logEC50_2 : float
        log10(EC50) of the second component.
    nH_2 : float
        Hill coefficient of the second component.
    """

    section_1 = (span * frac) / (1.0 + 10.0 ** (nH_1 * (logEC50_1 - x)))
    section_2 = (span * (1.0 - frac)) / (1.0 + 10.0 ** (nH_2 * (logEC50_2 - x)))

    return bottom + section_1 + section_2


def midpointTm(b, e, f, minmax=None, eps=1e-6):
    if minmax != None:
        min, max = minmax
    tm = e / np.exp(-np.log(2**(1/f) - 1) / b)
    return tm if minmax==None else (tm + eps) * (max - min) + min


def curve_fitting(x, y):

    eps = 1e-6
    def objective_function(params, x, y):
        return np.sum((y - assym_sigmoid(x, *params))**2)

    # Normalise time
    t_min = x.min()
    t_max = x.max()
    x = (x - t_min) / (t_max - t_min) + eps
    x = list(x)
    y = list(y)

    # Perform curve fitting
    for max_f in [100, 1000]:                       # max_f may need to vary depending on the input data
        bounds = [(eps, 10), (eps, 10), (eps, 10), (eps, 10), (1, max_f)]

        result = differential_evolution(objective_function, bounds, args=(x, y), strategy='best1bin', maxiter=1000)
        popt = result.x
        b, c, d, e, f = tuple(popt) 
    
        # get y_hat and midpointTm
        y_hat = assym_sigmoid(x, b, c, d, e, f)
        midTm = midpointTm(b, e, f, (t_min, t_max))
    
        # Calculate R^2
        ss_res = np.sum((y - y_hat) ** 2)       # Residual sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
        r_squared = 1 - (ss_res / ss_tot)
        if r_squared > 0.5:
            break

    return y_hat, midTm, r_squared


def curve_fitting_bi_sigmoid(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    eps = 1e-8
    y_range = y.max() - y.min()

    def objective_function(params, x, y):
        return np.sum((y - bi_sigmoid(x, *params)) ** 2)

    # Parameter bounds
    bounds = [
        (y.min() - 0.5 * y_range, y.min() + 0.5 * y_range),  # bottom
        (eps, 20.0 * y_range),                               # span
        (eps, 20),                                           # frac
        (x.min(), x.max()),                                  # logEC50_1
        (eps, 20),                                           # nH_1
        (x.min(), x.max()),                                  # logEC50_2
        (eps, 20),                                           # nH_2
    ]

    # Try once; repeating blindly does not help identifiability
    result = differential_evolution(
        objective_function,
        bounds,
        args=(x, y),
        strategy="best1bin",
        maxiter=1500,
        popsize=20,
        tol=1e-6,
        polish=True,
        seed=0,
    )

    popt = result.x
    # print(popt)
    y_hat = bi_sigmoid(x, *popt)

    # R²
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    midTm = 0
    return y_hat, midTm, r_squared


def cut_curve(y, x = None, x_min = None, rtn_drv = False):
    # find maximum point from right, and turning point from left
    def D(x):
        x = np.array(x)
        return x[1:len(x)] - x[0:len(x)-1]
    
    d_y = D(y)
    d2_y = D(d_y)
    d3_y = D(d2_y)

    # 1. Find the steepest slope
    steepest_slope = np.argmax(d_y)

    # 2. Move to right to reach the maximum
    for i in np.arange(steepest_slope, len(y)-1, 1):
        if y[i+1] < y[i]:
            break
    cut_max = i

    # 3. Move to left to get to the min of y'
    for i in np.arange(steepest_slope, 1, -1):
        if (d_y[i-1] > d_y[i] - 0) or (d_y[i-1] * d_y[i] < 0):
            if x_min is None or x[i] < x_min:
                break
    cut_min = i
    # print(cut_min)

    # plt.plot(x[1:], d_y, linewidth=2)
    # plt.savefig(f'pic/{cut_min}.jpeg')
    # plt.close()

    if rtn_drv:
        return cut_min, cut_max, steepest_slope, d_y, d2_y
    else:
        return cut_min, cut_max, steepest_slope


def get_melting_tempreture(x, y, fit_curve = True, method = "cut", window = 50):

    """
    This is the updated version of the old function
    curve_fitting_with_cut_curve()
    """
    
    cut_min, cut_max, steepest_slope = cut_curve(y, x, x_min=40)
    Tm_no_cf = (x[int(steepest_slope)] + x[int(steepest_slope) + 1]) / 2

    if method == "pad":
        cut_min_new = cut_min - window
        cut_max_new = cut_max + window
        if cut_min_new <= 0:
            cut_min_new = 0
        if cut_max_new >= len(y):
            cut_max_new = len(y)
    
        y_max = y[cut_max]
        y_min = y[cut_min]
    
        y[cut_max + 1 : len(y)] = y[cut_max]
        y[0 : cut_min - 1] = y[cut_min]

    # Cut curve
    elif method == "cut":
        x = x[cut_min : cut_max]
        y = y[cut_min : cut_max]

    # Do the main curve fitting
    if fit_curve:
        y_hat, midTm_cf, r_squared = curve_fitting_bi_sigmoid(x, y)
    else:
        y_hat, midTm_cf, r_squared = (0, 0, 0)

    out = {
        'x' : x,
        't' : x,
        'y' : y,
        'y_hat_sigmoid' : y_hat,
        'Tm_sigmoid' : midTm_cf,
        'r_squared_sigmoid' : r_squared,
        'Tm_steepest_slope' : Tm_no_cf,
    }
    return out

def cubic_spline(x, y, n):
    cs = CubicSpline(x, y)
    x_new = np.linspace(min(x), max(x), n)
    y_new = cs(x_new)
    return pd.Series(x_new), pd.Series(y_new)


import matplotlib.pyplot as plt
import numpy as np
from uncertainties.unumpy import uarray
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit

import os
os.makedirs("build", exist_ok=True)

def rel_Abweichung(exp, theo):
    return (np.abs(exp-theo)/(theo)*100) #ist schon in Prozent

def contrast(U_max, U_min):
    return (U_max - U_min)/(U_max + U_min)

def contrast_theo(phi):
    return np.abs(2*unp.cos(phi)*unp.sin(phi))

def refraction_glass(d, theta, theta_0, M, wavelength):
    return (2 * d * theta * theta_0)/(2 * d * theta * theta_0 - wavelength * M)

def refraction_gas(M, L, wavelength):
    return 1 + (M * wavelength / L)


# Contrast Analysis
phi, U_max1_raw, U_max2_raw, U_max3_raw, U_min1_raw, U_min2_raw, U_min3_raw= np.genfromtxt("Messdaten/contrast.txt", unpack=True)

phi = np.array([ufloat(p, 2.0) for p in phi])

U_max1 = np.array([ufloat(u, 0.01) for u in U_max1_raw])
U_max2 = np.array([ufloat(u, 0.01) for u in U_max2_raw])
U_max3 = np.array([ufloat(u, 0.01) for u in U_max3_raw])
U_min1 = np.array([ufloat(u, 0.01) for u in U_min1_raw])
U_min2 = np.array([ufloat(u, 0.01) for u in U_min2_raw])
U_min3 = np.array([ufloat(u, 0.01) for u in U_min3_raw])


U_max_all = np.array([U_max1, U_max2, U_max3])
U_min_all = np.array([U_min1, U_min2, U_min3])

U_max_vals = unp.nominal_values(U_max_all)
U_min_vals = unp.nominal_values(U_min_all)


U_max_mean = np.mean(U_max_vals, axis=0)
U_max_std = np.std(U_max_vals, axis=0, ddof=1)

U_min_mean = np.mean(U_min_vals, axis=0)
U_min_std = np.std(U_min_vals, axis=0, ddof=1)

U_max = np.array([ufloat(m, s) for m, s in zip(U_max_mean, U_max_std)])
U_min = np.array([ufloat(m, s) for m, s in zip(U_min_mean, U_min_std)])


# print(f"U_max {U_max}")
# print(f"U_min {U_min}")
x = np.linspace(0, np.pi, 1000)

fig, ax = plt.subplots(1, 1, layout="constrained")
# ax.plot(unp.nominal_values(phi), unp.nominal_values(contrast(U_max, U_min)), "o", label="Calculated values")
ax.errorbar(
    unp.nominal_values(phi), 
    unp.nominal_values(contrast(U_max, U_min)),
    xerr=unp.std_devs(phi), 
    yerr=unp.std_devs(contrast(U_max, U_min)),
    fmt="o", 
    markersize=2.5,
    label="Calculated values",
    capsize=0,
    color="indigo"
)
ax.plot(x * 180 / np.pi, contrast_theo(x), "-", label="Theoretical curve", color="darkturquoise")
ax.axvline(x=135, color='red', linestyle='--', label=r"$\phi = 135^\circ$")
ax.set_xlabel(r"$\phi\,[\unit{\degree}$]")
ax.set_ylabel(r"Contrast $\nu$")
ax.set_xlim([-5, 185])
ax.set_ylim([0, 1.05])
ax.legend(loc="upper center")
plt.grid()
fig.savefig("build/contrast.pdf")

# refractive index of glass
M_glass = np.genfromtxt("Messdaten/glass.txt", unpack=True)

M_glass = unp.uarray(np.mean(M_glass), np.std(M_glass))
# M_glass = np.mean(M_glass)
lambda_HeNe = 632.990 * 10e-9

d = 1 * 10e-3
theta = uarray(10, 2) * np.pi / 180
theta_0 = uarray(10, 2) * np.pi / 180 
n_glass = refraction_glass(d,theta, theta_0, M_glass, lambda_HeNe)
n_glass_mean = np.mean(n_glass)

# print(f"M_glass mean: {M_glass}")
print(f"n_glass: {n_glass}")
print(f"n_glass mean: {n_glass_mean}") 

# refractive index of gas
p, M1, M2, M3 = np.genfromtxt("Messdaten/gas.txt", unpack=True)
p = unp.uarray(p, 1)

M_all = np.vstack([M1, M2, M3])

M_mean = np.mean(M_all, axis=0)
M_std = np.std(M_all, axis=0)

M_gas = uarray(M_mean, M_std)
# print(f"M_gas mean: {M_gas}")

L = ufloat(100.0, 0.1) * 10e-3 # length of the gas cell

n_gas = refraction_gas(M_gas, L, lambda_HeNe)
# print(f"n_gas: {n_gas}")
# print(f"n_gas mean: {np.mean(n_gas)}")


# refractive index of gas with Lorentz-Lorenz Law

def n_taylor(x, a, b):
    return a * x + b

T_C_vals = unp.uarray([21.5, 21.6, 21.7], [1, 1, 1])
T_vals = T_C_vals + 273.15

print(f"T_vals: {T_vals}")
T_C = uarray(np.mean(unp.nominal_values(T_C_vals)), np.std(unp.nominal_values(T_C_vals)))
T = uarray(np.mean(unp.nominal_values(T_vals)), np.std(unp.nominal_values(T_vals)))

print(f"T_C: {T_C}")
print(f"T: {T}")

x_plot = p / T
x = np.linspace(-1, 3.5, 1000)
params = curve_fit(n_taylor, unp.nominal_values(x_plot), unp.nominal_values(M_gas)) 

[err_a, err_b] = params[1]

cov = params[1]
err_a = np.sqrt(cov[0, 0])
err_b = np.sqrt(cov[1, 1])

a = ufloat(params[0][0], err_a)
b = ufloat(params[0][1], err_b)

print(f"a: {a}")
print(f"b: {b}")

p_0 = 1013
T_0 = 15 + 273.15

# n_theo = n_taylor(p_0/T_0, lambda_HeNe/L , 1)
n_theo = lambda_HeNe/L * n_taylor(p_0/T_0,a, b) + 1
print(f"n_theo: {n_theo}")

fig2, ax2 = plt.subplots(1, 1, layout="constrained")
# ax2.plot(unp.nominal_values(x_plot), unp.nominal_values(M_gas), "x", label="Measured Values")
ax2.errorbar(
    unp.nominal_values(x_plot),
    unp.nominal_values(M_gas),
    xerr=unp.std_devs(x_plot),
    yerr=unp.std_devs(M_gas),
    fmt="o",
    markersize=2.5,
    label=r"Measured values of $\bar{M}$",
    capsize=0,
    color="indigo"
)
ax2.plot(x, unp.nominal_values(n_taylor(x,a, b)), "-", label="Linear fit", color="darkturquoise")
ax2.set_xlabel(r"$\frac{p}{T}\,[\unit{\milli\bar \kelvin^{-1}}$]")
ax2.set_ylabel(r"${M}$")
ax2.set_xlim([-0.1, 3.5])
ax2.set_ylim([-1, 45])
ax2.legend(loc="best")
plt.grid()
fig2.savefig("build/fit.pdf")

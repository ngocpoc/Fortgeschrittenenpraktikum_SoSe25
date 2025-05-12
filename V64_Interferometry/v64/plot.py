import matplotlib.pyplot as plt
import numpy as np
from uncertainties.unumpy import uarray
from uncertainties import ufloat
import uncertainties.unumpy as unp

np.set_printoptions(precision=4)
def rel_Abweichung(exp, theo):
    return (np.abs(exp-theo)/(theo)*100) #ist schon in Prozent

def contrast(U_max, U_min):
    return (U_max - U_min)/(U_max + U_min)

def contrast_theo(phi):
    return np.abs(2*unp.cos(phi)*unp.sin(phi))

def refraction_glass(d, theta, theta_0, M, wavelength):
    return (2 * d * theta * theta_0)/(2 * d * theta * theta_0 - wavelength * M)

def refraction_gas(M, L, wavelength):
    return 1 - (M * wavelength / L)


# Contrast Analysis
phi, U_max1, U_max2, U_max3, U_min1, U_min2, U_min3= np.genfromtxt("Messdaten/contrast.txt", unpack=True)

phi = uarray(phi, 2.0)
U_max1, U_max2, U_max3 = uarray(U_max1, 0.02), uarray(U_max2, 0.02), uarray(U_max3, 0.01)
U_min1, U_min2, U_min3 = uarray(U_min1, 0.02), uarray(U_min2, 0.02), uarray(U_min3, 0.01)

U_max = (U_max1 + U_max2 + U_max3)/ 3
U_min = (U_min1 + U_min2 + U_min3)/ 3

print(f"U_max {U_max}")
print(f"U_min {U_min}")
x = np.linspace(0, np.pi, 1000)

fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(unp.nominal_values(phi), unp.nominal_values(contrast(U_max, U_min)), "o", label="Calculated values")
ax.plot(x * 180 / np.pi, contrast_theo(x), "-", label="Theoretical curve")
ax.set_xlabel(r"$\phi \mathbin{/} \unit{\degree}$")
ax.set_ylabel(r"Contrast $\nu$")
ax.legend(loc="upper center")
plt.grid()
fig.savefig("build/constrast.pdf")

# refractive index of glass
M_glass = np.genfromtxt("Messdaten/glass.txt", unpack=True)

M_glass = unp.uarray(np.mean(M_glass), np.std(M_glass))
# M_glass = np.mean(M_glass)
lambda_HeNe = 632.990 * 10e-9

d = 1 * 10e-3
theta = uarray(10, 2) * np.pi / 180
theta_0 = uarray(10, 2) * np.pi / 180 
n_glass = refraction_glass(d,theta, theta_0, M_glass, lambda_HeNe)
# n_glass_mean = np.mean(n_glass)

print(f"M_glass mean: {M_glass}")
print(f"n_glass: {n_glass}")
# print(f"n_glass mean: {n_glass_mean}")/

# refractive index of gas
p, M1, M2, M3 = np.genfromtxt("Messdaten/gas.txt", unpack=True)
p = unp.uarray(p, 1)

M_all = np.vstack([M1, M2, M3])

M_mean = np.mean(M_all, axis=0)
M_std = np.std(M_all, axis=0)

M_gas = uarray(M_mean, M_std)
print(f"M_gas mean: {M_gas}")

L = ufloat(100.0, 0.1) * 10e-3 # length of the gas cell

n_gas = refraction_gas(M_gas, L, lambda_HeNe)
print(f"n_gas: {n_gas}")

# x = np.linspace(0, 10, 1000)
# y = x ** np.sin(x)

# fig, (ax1, ax2) = plt.subplots(1, 2, layout="constrained")
# ax1.plot(x, y, label="Kurve")
# ax1.set_xlabel(r"$\alpha \mathbin{/} \unit{\ohm}$")
# ax1.set_ylabel(r"$y \mathbin{/} \unit{\micro\joule}$")
# ax1.legend(loc="best")

# ax2.plot(x, y, label="Kurve")
# ax2.set_xlabel(r"$\alpha \mathbin{/} \unit{\ohm}$")
# ax2.set_ylabel(r"$y \mathbin{/} \unit{\micro\joule}$")
# ax2.legend(loc="best")

# fig.savefig("build/plot.pdf")
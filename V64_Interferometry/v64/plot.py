import matplotlib.pyplot as plt
import numpy as np
from uncertainties.unumpy import uarray

def rel_Abweichung(exp, theo):
    return (np.abs(exp-theo)/(theo)*100) #ist schon in Prozent

def contrast(U_max, U_min):
    return (U_max - U_min)/(U_max + U_min)

def contrast_theo(phi):
    return np.abs(2*np.cos(phi)*np.sin(phi))

phi, U_max1, U_max2, U_max3, U_min1, U_min2, U_min3= np.genfromtxt("Messdaten/contrast.txt", unpack=True)

# phi = uarray(phi, 1.0)
# U_max1, U_max2, U_max3 = uarray(U_max1, 0.02), uarray(U_max2, 0.02), uarray(U_max3, 0.02)
# U_min1, U_min2, U_min3 = uarray(U_min1, 0.02), uarray(U_min2, 0.02), uarray(U_min3, 0.02)

U_max = (U_max1 + U_max2 + U_max3)/ 3
U_min = (U_min1 + U_min2 + U_min3)/ 3

x = np.linspace(0, np.pi, 1000)

fig, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(phi, contrast(U_max, U_min), "o", label="Calculated values")
ax.plot(x * 180 / np.pi, contrast_theo(x), "-", label="Theoretical curve")
ax.set_xlabel(r"$\phi \mathbin{/} \unit{\degree}$")
ax.set_ylabel(r"Contrast $\nu$")
ax.legend(loc="best")
plt.grid()
fig.savefig("build/constrast.pdf")


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
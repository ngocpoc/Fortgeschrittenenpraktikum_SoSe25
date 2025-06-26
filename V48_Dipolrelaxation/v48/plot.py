import matplotlib.pyplot as plt
import numpy as np
from uncertainties.unumpy import uarray
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
import scipy.constants as const
# import matplotlib.ticker as ticker

k_B = const.k
eV = 6.241509074460763e18 

def rel_Abweichung(exp, theo):
    return (np.abs(exp-theo)/(theo)*100) #ist schon in Prozent

def lin_fit(x,m,b):
    return m*x + b

def exp_Untergrund(t, m, a, b):
    return m * np.exp(a * t) + b


def plot_strom(ax, T_K, I, mask_exclude, mask_untergrund, untergrund_fit=None, label="Messwerte", fit_color="darkgreen"):
    """Hilfsfunktion zum Plotten der Stromdaten mit Untergrund"""
    ax.plot(T_K, I, "x", color="tab:blue", label=label)
    ax.plot(T_K[mask_exclude], I[mask_exclude], "x", color="tab:red", label="Exkludierte Messwerte")
    ax.plot(T_K[mask_untergrund], I[mask_untergrund], "x", color="tab:green", label="Untergrund")

    if untergrund_fit is not None:
        T_fit = np.linspace(195, 300, 2000)
        I_fit = exp_Untergrund(T_fit, *untergrund_fit)
        ax.plot(T_fit, I_fit, "-", color=fit_color, label="Exp. Fit (Untergrund)")

    ax.set_xlabel(r"$T\,[\si{\kelvin}]$")
    ax.set_ylabel(r"$I\,[\si{\pico\ampere}]$")
    ax.set_xlim([195, 295])
    ax.grid()
    ax.legend()

t_1, I_1, T_1 = np.genfromtxt("Messdaten/ersteReihe.txt", unpack=True)
t_2, I_2, T_2 = np.genfromtxt("Messdaten/zweiteReiheVor.txt", unpack=True)
t_2_f, I_2_f, T_2_f = np.genfromtxt("Messdaten/zweiteReihe.txt", unpack=True)\

T_1_K = T_1 + 273.15
T_2_K = T_2 + 273.15
T_2_K_f = T_2_f + 273.15

#####################################################
# Heizraten
#####################################################
cut = 30

# Fit für erste Messreihe (zwei lineare Bereiche)
params1, cov1 = curve_fit(lin_fit, t_1, T_1_K)
# params1, cov1 = curve_fit(lin_fit, t_1[:cut], T_1_K[:cut])
# params12, cov12 = curve_fit(lin_fit, t_1[cut:], T_1_K[cut:])

# Fit für zweite Messreihe
params2, cov2 = curve_fit(lin_fit, t_2, T_2_K) 

params2_f, cov2_f = curve_fit(lin_fit, t_2_f, T_2_K_f) 

# Fit-Parameter mit Unsicherheiten
def get_fit_params(params, cov):
    m, b = params
    err_m, err_b = np.sqrt(np.diag(cov))
    return ufloat(m, err_m), ufloat(b, err_b)

m1, b1 = get_fit_params(params1, cov1)
# m12, b12 = get_fit_params(params12, cov12)
m2, b2 = get_fit_params(params2, cov2)

m2_f, b2_f = get_fit_params(params2_f, cov2_f)

m1_s = m1/60
m2_s = m2/60

m2_s_f = m2_f/60

# Nominalwerte extrahieren für Plot
m1_nom, b1_nom = unp.nominal_values([m1, b1])
# m12_nom, b12_nom = unp.nominal_values([m12, b12])
m2_nom, b2_nom = unp.nominal_values([m2, b2])

m2_nom_f, b2_nom_f = unp.nominal_values([m2_f, b2_f]) 
tlist1 = np.linspace(0,105, 1000)
# tlist12 = np.linspace(cut, 105, 1000)
tlist2 = np.linspace(0,40, 1000)


fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)

ax1.plot(t_1, T_1_K, "x", label="Messwerte")
ax1.plot(tlist1, lin_fit(tlist1, m1_nom, b1_nom), "-", color="darkviolet", label="Lin. Regression")
# ax1.plot(tlist12, lin_fit(tlist12, m12_nom, b12_nom), "-", label="Lin. Regression")
ax1.set_xlabel(r"$t\,[\si{\minute}$]")
ax1.set_title("Erste Messreihe")
ax1.set_ylabel(r"$T\,[\si{\kelvin}]$")
ax1.legend(loc="upper left")
ax1.grid()

ax2.plot(t_2, T_2_K, "x", label="Messwerte")
ax2.plot(tlist2, lin_fit(tlist2, m2_nom, b2_nom), "-", color="darkviolet",  label="Lin. Regression")
ax2.set_xlabel(r"$t\,[\si{\minute}]$")
ax2.set_title("Zweite Messreihe")
ax2.legend(loc="upper left")
ax2.grid()

fig.savefig("build/Heizraten.pdf")

#####################################################
# Depolarisationsstrom 
#####################################################
faktor1 = (1/20) * 0.3 * 1e-11
faktor2 = (1/20) * 0.3 * 1e-10

# Erste Messung #####################################
I_1 = -faktor1 * I_1
I_1 = I_1 * 1e12    # in pA 

# Indizes für Untergrund
# indices_untergrund = np.concatenate([
#     np.array([22, 23]),
#     np.arange(66, 87)
# ])

# # Indizes für exkludierte Werte (alles außerhalb des interessanten Bereichs)
# indices_exclude = np.concatenate([
#     np.arange(0, 22),
#     np.arange(87, len(T_1_K))
# ])

indices_untergrund = np.concatenate([
    np.arange(0,17),
    # np.array([22, 23]),
    np.arange(66, 87)
])

# Indizes für exkludierte Werte (alles außerhalb des interessanten Bereichs)
indices_exclude = np.concatenate([
    np.arange(17, 22),
    np.arange(87, len(T_1_K))
])

# Bool-Masken erzeugen
mask_untergrund = np.zeros_like(I_1, dtype=bool)
mask_untergrund[indices_untergrund] = True

mask_exclude = np.zeros_like(I_1, dtype=bool)
mask_exclude[indices_exclude] = True

params_untergrund, cov_untergrund = curve_fit(
    exp_Untergrund,
    T_1_K[mask_untergrund],
    I_1[mask_untergrund],
    p0=(0.01, 0.1, -1.1)
)

fig2, ax2 = plt.subplots()
plot_strom(
    ax2,
    T_1_K,
    I_1,
    mask_exclude=mask_exclude,
    mask_untergrund=mask_untergrund,
    untergrund_fit=params_untergrund
)
ax2.set_ylim([-1.3, 8])
fig2.savefig("build/ersterStrom.pdf")

# Zweite Messung
I_2[0:20] = -faktor1 * I_2[0:20]
I_2[20:28] = -faktor2 * I_2[20:28]
I_2[28:] = -faktor1 * I_2[28:]
I_2 = I_2 * 1e12    # in pA

# Indizes für Untergrund
indices_untergrund2 = np.concatenate([
    np.arange(0, 11), 
    # np.array([12, 13]),
    np.arange(30, 34)
])

# Indizes für exkludierte Werte (alles außerhalb des interessanten Bereichs)
indices_exclude2 = np.concatenate([
    np.array([11]),
    np.arange(20,23),
    np.array([27]),
    np.arange(34, len(T_2_K))
])

# Bool-Masken erzeugen
mask_untergrund2 = np.zeros_like(I_2, dtype=bool)
mask_untergrund2[indices_untergrund2] = True

mask_exclude2 = np.zeros_like(I_2, dtype=bool)
mask_exclude2[indices_exclude2] = True

params_untergrund2, cov_untergrund2 = curve_fit(
    exp_Untergrund,
    T_2_K[mask_untergrund2],
    I_2[mask_untergrund2],
    p0=(2, 0.01, -1.1)
)

fig3, ax3 = plt.subplots()
plot_strom(
    ax3,
    T_2_K,
    I_2,
    mask_exclude=mask_exclude2,
    mask_untergrund=mask_untergrund2,
    untergrund_fit=params_untergrund2
)
ax3.set_ylim([-1.3, 14])
fig3.savefig("build/zweiterStromVor.pdf")

I_2_f[0:20] = -faktor1 * I_2_f[0:20]
I_2_f[20:28] = -faktor2 * I_2_f[20:28]
I_2_f[28:] = -faktor1 * I_2_f[28:]
I_2_f = I_2_f * 1e12    # in pA

fig3, ax3 = plt.subplots()
ax3.plot(T_2_K_f, I_2_f, "x", color="tab:blue", label="Messwerte")
ax3.set_xlabel(r"$T\,[\si{\kelvin}]$")
ax3.set_ylabel(r"$I\,[\si{\pico\ampere}]$")
ax3.set_xlim([195, 295])
ax3.grid()
ax3.legend()
# ax3.set_ylim([-1.3, 14])
fig3.savefig("build/zweiterStrom.pdf")


#####################################################
# Messungen ohne Untergrund
#####################################################
m1_exp, a1_exp, b1_exp = params_untergrund
m2_exp, a2_exp, b2_exp = params_untergrund2

T_clean1 = T_1_K[~mask_exclude]
I_clean1 = I_1[~mask_exclude]-exp_Untergrund(T_clean1, m1_exp, a1_exp, b1_exp)

T_clean2 = T_2_K[~mask_exclude2]
I_clean2 = I_2[~mask_exclude2]-exp_Untergrund(T_clean2, m2_exp, a2_exp, b2_exp)

# Indizes für Polarisationsansatz
indices_polarisation1 = np.concatenate([
    np.arange(17, 33), 
])

# Indizes für Stromdichtenansatz
indices_stromdichte1 = np.concatenate([
    np.arange(17,58),
])

# Bool-Masken erzeugen
mask_polarisation1 = np.zeros_like(I_clean1, dtype=bool)
mask_polarisation1[indices_polarisation1] = True

mask_stromdichte1 = np.zeros_like(I_clean1, dtype=bool)
mask_stromdichte1[indices_stromdichte1] = True

fig4, ax4 = plt.subplots()
ax4.plot(T_clean1, I_clean1, "x", color="tab:blue", label="Korrigierte Messwerte")
ax4.plot(T_clean1[mask_polarisation1], I_clean1[mask_polarisation1], "o", fillstyle="none", color="tab:green", label="Polarisationsansatz")
ax4.plot(T_clean1[mask_stromdichte1], I_clean1[mask_stromdichte1], "x", fillstyle="full", color="tab:orange", label="Stromdichtenansatz")
ax4.set_xlabel(r"$T\,[\si{\kelvin}]$")
ax4.set_ylabel(r"$I_D\,[\si{\pico\ampere}]$")
ax4.grid()
ax4.set_xlim([195,295])
ax4.set_ylim([-1.3,8])
ax4.legend(loc="upper left")
fig4.savefig("build/ersterUntergrundfrei.pdf")


# Indizes für Polarisationsansatz
indices_polarisation2 = np.concatenate([
    np.arange(11, 18), 
])

# Indizes für Stromdichtenansatz
indices_stromdichte2 = np.concatenate([
    np.arange(11,26),
])

# Bool-Masken erzeugen
mask_polarisation2 = np.zeros_like(I_clean2, dtype=bool)
mask_polarisation2[indices_polarisation2] = True

mask_stromdichte2 = np.zeros_like(I_clean2, dtype=bool)
mask_stromdichte2[indices_stromdichte2] = True


fig5, ax5 = plt.subplots()
ax5.plot(T_clean2, I_clean2,"x", color="tab:blue", label="Korrigierte Messwerte")
ax5.plot(T_clean2[mask_polarisation2], I_clean2[mask_polarisation2], "o", fillstyle="none", color="tab:green", label="Polarisationsansatz")
ax5.plot(T_clean2[mask_stromdichte2], I_clean2[mask_stromdichte2], "x",fillstyle="full", color="tab:orange", label="Stromdichtenansatz")
ax5.set_xlabel(r"$T\,[\si{\kelvin}]$")
ax5.set_ylabel(r"$I_D\,[\si{\pico\ampere}]$")
ax5.grid()
ax5.set_xlim([195,295])
ax5.set_ylim([-1.3,14])
ax5.legend(loc="upper left")
fig5.savefig("build/zweiterUntergrundfrei.pdf")


#####################################################
# Polarisationsansatz
#####################################################

I_0 = 1 # 1pA oder 1A, je nachdem

lnI_1 = np.log((I_clean1[mask_polarisation1] )/ I_0) # hier ist I_0 jetzt 1pA
T_1_inv = 1 / T_clean1[mask_polarisation1]

params_pol1, cov_pol1 = curve_fit(lin_fit, T_1_inv, lnI_1)

m1_pol, b1_pol = params_pol1
Tlist_inv = np.linspace(4e-3, 4.35e-3, 1000)
fig6, ax6 = plt.subplots()
ax6.plot(T_1_inv, lnI_1, "x", color="tab:green", label="Messwerte")
ax6.plot(Tlist_inv, lin_fit(Tlist_inv, m1_pol, b1_pol), "-", color="darkgreen", label="Lin. Fit")
ax6.set_xlabel(r"$T^{-1}\,[\si{\per\kelvin}]$")
ax6.set_ylabel(r"$\ln\left( \frac{I_D}{I_0} \right)$")
ax6.set_xlim([4e-3, 4.35e-3])
ax6.set_ylim([-1.25,2])
ax6.legend()
ax6.grid()
fig6.savefig("build/polarisation1.pdf")


lnI_2 = np.log((I_clean2[mask_polarisation2] )/ I_0) # hier ist I_0 jetzt 1pA
T_2_inv = 1 / T_clean2[mask_polarisation2]

params_pol2, cov_pol2 = curve_fit(lin_fit, T_2_inv, lnI_2)

m2_pol, b2_pol = params_pol2
fig7, ax7 = plt.subplots()
ax7.plot(T_2_inv, lnI_2, "x", color="tab:green", label="Messwerte")
ax7.plot(Tlist_inv, lin_fit(Tlist_inv, m2_pol, b2_pol), "-", color="darkgreen", label="Lin. Fit")
ax7.set_xlabel(r"$T^{-1}\,[\si{\per\kelvin}]$")
ax7.set_ylabel(r"$\ln\left( \frac{I_D}{I_0} \right)$")
ax7.set_xlim([4e-3, 4.35e-3])
ax7.set_ylim([-1.25,2])
ax7.legend()
ax7.grid()
fig7.savefig("build/polarisation2.pdf")

#####################################################
# Stromdichtenansatz
#####################################################

# Strom- und Temperaturdaten im Bereich des Stromdichtenansatzes
I_strom1 = I_clean1[mask_stromdichte1]
T_strom1 = T_clean1[mask_stromdichte1]

def IntTrapez(T, Strom, T_end):
    # T_end = T[-1]
    integral = np.array([trapezoid(Strom[(T > t) & (T <= T_end)], T[(T > t) & (T <= T_end)])for t in T])
    return np.log(integral/Strom)

ln_integral1 = IntTrapez(T_strom1, I_strom1, np.max(T_strom1))
T_strom1 = T_strom1[np.isfinite(ln_integral1)]
I_strom1 = I_strom1[np.isfinite(ln_integral1)]
ln_integral1 = ln_integral1[np.isfinite(ln_integral1)]
T_inv1 = 1/T_strom1

params_strom1, cov_strom1 = curve_fit(lin_fit, T_inv1, ln_integral1)

m1_str, b1_str = get_fit_params(params_strom1, cov_strom1)
W1_str = m1_str*k_B

Tlist_strom = np.linspace(3.65e-3, 7e-3)
fig8, ax8 = plt.subplots()

ax8.plot(T_inv1, ln_integral1, "x", color="tab:orange", label=r"Messwerte")
ax8.plot(Tlist_strom, lin_fit(Tlist_strom, *params_strom1), "-", color="orangered", label="Lin. Fit")
ax8.set_xlabel(r"$T^{-1}\,[\si{\per\kelvin}]$")
ax8.set_ylabel(r"$\ln\left( \frac{1}{I(T)}\int_T^{\infty} I(T')\,\symup{d}T' \right)$")
ax8.set_xlim([3.65e-3, 4.4e-3])
ax8.set_ylim([-1.25, 8])
ax8.grid()
ax8.legend(loc="upper left")
fig8.savefig("build/stromdichte1.pdf")

# zweite Messung
I_strom2 = I_clean2[mask_stromdichte2]
T_strom2 = T_clean2[mask_stromdichte2]

ln_integral2 = IntTrapez(T_strom2, I_strom2, np.max(T_strom2))
T_strom2 = T_strom2[np.isfinite(ln_integral2)]
I_strom2 = I_strom2[np.isfinite(ln_integral2)]
ln_integral2 = ln_integral2[np.isfinite(ln_integral2)]
T_inv2 = 1/T_strom2

params_strom2, cov_strom2 = curve_fit(lin_fit, T_inv2 , ln_integral2)

m2_str, b2_str = get_fit_params(params_strom2, cov_strom2)
W2_str = m2_str*k_B

fig9, ax9 = plt.subplots()

ax9.plot(T_inv2, ln_integral2, "x", color="tab:orange", label=r"Messwerte")
ax9.plot(Tlist_strom, lin_fit(Tlist_strom, *params_strom2), "-", color="orangered", label="Lin. Fit")
ax9.set_xlabel(r"$T^{-1}\,[\si{\per\kelvin}]$")
ax9.set_ylabel(r"$\ln\left( \frac{1}{I(T)} \int_T^{\infty} I(T')\,\symup{dT}'\right)$")
ax9.set_xlim([3.65e-3, 4.4e-3])
ax9.set_ylim([-1.25, 8])
ax9.grid()
ax9.legend(loc="upper left")
fig9.savefig("build/stromdichte2.pdf")

#####################################################
# Relaxationszeit
#####################################################
def tau_max(T, rate, W):
    return (T**2 * k_B)/(rate * W)

def tau0(tau, T, W):
    return tau * unp.exp(-W / (k_B * T))

def tau(tau0, T, W):
    return tau0 * unp.exp(W / (k_B * T))
# erste Messreihe

idx_max1 = np.argmax(I_strom1)
I_max1 = I_strom1[idx_max1]
T_max1 = T_strom1[idx_max1]

tau_max1 = tau_max(T_max1, m1_s, W1_str)
tau0_1 = tau0(tau_max1, T_max1, W1_str)

# zweite Messreihe
idx_max2 = np.argmax(I_strom2)
I_max2 = I_strom2[idx_max2]
T_max2 = T_strom2[idx_max2]

tau_max2 = tau_max(T_max2, m2_s, W2_str)
tau0_2 = tau0(tau_max2, T_max2, W2_str)

Tlist_relax = np.linspace(200,300, 3000)

fig10, ax10 = plt.subplots()
ax10.plot(Tlist_relax, unp.nominal_values(tau(tau0_1, Tlist_relax, W1_str)), "-", color="tab:blue", label="erste Messreihe")
ax10.plot(Tlist_relax, unp.nominal_values(tau(tau0_2, Tlist_relax, W2_str)), "-", color="tab:orange", label="zweite Messreihe")
ax10.set_xlabel(r"$T\,[\si{\kelvin}]$")
ax10.set_ylabel(r"$\tau\,[\si{\second}]$")
ax10.legend(loc="upper right")
ax10.set_ylim([0,400])
ax10.set_xlim([250,290])
ax10.grid()
fig10.savefig("build/relaxation.pdf")
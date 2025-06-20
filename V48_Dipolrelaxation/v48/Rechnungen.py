import matplotlib.pyplot as plt
import numpy as np
from uncertainties.unumpy import uarray
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
import scipy.constants as const
from scipy.integrate import trapezoid

def rel_Abweichung(exp, theo):
    return (np.abs(exp-theo)/(theo)*100) #ist schon in Prozent

def lin_fit(x,m,b):
    return m*x + b

def exp_Untergrund(t, m, a, b):
    return m * np.exp(a * t) + b
    
t_1, I_1, T_1 = np.genfromtxt("Messdaten/ersteReihe.txt", unpack=True)
t_2, I_2, T_2 = np.genfromtxt("Messdaten/zweiteReiheVor.txt", unpack=True)
T_1_K = T_1 + 273.15
T_2_K = T_2 + 273.15

#####################################################
# Heizraten
#####################################################
print("Heizraten:")
cut = 30

# Fit für erste Messreihe (zwei lineare Bereiche)
params1, cov1 = curve_fit(lin_fit, t_1, T_1_K)
# params1, cov1 = curve_fit(lin_fit, t_1[:cut], T_1_K[:cut])
# params12, cov12 = curve_fit(lin_fit, t_1[cut:], T_1_K[cut:])

# Fit für zweite Messreihe
params2, cov2 = curve_fit(lin_fit, t_2, T_2_K) 

# Fit-Parameter mit Unsicherheiten
def get_fit_params(params, cov):
    m, b = params
    err_m, err_b = np.sqrt(np.diag(cov))
    return ufloat(m, err_m), ufloat(b, err_b)

m1, b1 = get_fit_params(params1, cov1)
# m12, b12 = get_fit_params(params12, cov12)
m2, b2 = get_fit_params(params2, cov2)

m1_s = m1/60
m2_s= m2/60

print("Erste Messung")
print(rf"(Heizrate) m1: {m1} K/min bzw. {m1_s} K/sec")
print(rf"b1: {b1} K" )
# print(rf"b12: {b12}")
# print(rf"(Heizrate) m12: {m12}")

print("Zweite Messung")
print(rf"(Heizrate) m2: {m2} K/min bzw. {m2_s} K/sec")
print(rf"b2: {b2} K")

faktor1 = (1/20) * 0.3 * 1e-11
faktor2 = (1/20) * 0.3 * 1e-10

# Erste Messung #####################################
I_1 = -faktor1 * I_1
I_1 = I_1 * 1e12

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


# Zweite Messung
I_2[0:20] = -faktor1 * I_2[0:20]
I_2[20:28] = -faktor2 * I_2[20:28]
I_2[28:] = -faktor1 * I_2[28:]
I_2 = I_2 * 1e12

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


#####################################################
# Messungen ohne Untergrund
#####################################################
m1_exp, a1_exp, b1_exp = params_untergrund
m2_exp, a2_exp, b2_exp = params_untergrund2

T_clean1 = T_1_K[~mask_exclude]
I_clean1 = I_1[~mask_exclude]-exp_Untergrund(T_clean1, m1_exp, a1_exp, b1_exp)

T_clean2 = T_2_K[~mask_exclude2]
I_clean2 = I_2[~mask_exclude2]-exp_Untergrund(T_clean2, m2_exp, a2_exp, b2_exp)

#####################################################
# Polarisationsansatz
#####################################################
I_0 = 1 # 1pA oder 1A, je nachdem
k_B = const.k
print("Polarisationsansatz")
# Messung 1
print("Messung 1")
indices_polarisation1 = np.concatenate([
    np.arange(17, 33), 
])

mask_polarisation1 = np.zeros_like(I_clean1, dtype=bool)
mask_polarisation1[indices_polarisation1] = True

lnI_1 = np.log((I_clean1[mask_polarisation1] )/ I_0) # hier ist I_0 jetzt 1pA
T_1_inv = 1 / T_clean1[mask_polarisation1]

params_pol1, cov_pol1 = curve_fit(lin_fit, T_1_inv, lnI_1)

m1_pol, b1_pol = get_fit_params(params_pol1, cov_pol1)

print(rf"m1_pol: {m1_pol} K")
print(rf"b1_pol: {b1_pol} ohne Einheit")
print(rf"W1_pol = {-m1_pol*k_B} eV")

# Messung 2
print("Messung 2")
indices_polarisation2 = np.concatenate([
    np.arange(11, 18), 
])

mask_polarisation2 = np.zeros_like(I_clean2, dtype=bool)
mask_polarisation2[indices_polarisation2] = True

lnI_2 = np.log((I_clean2[mask_polarisation2] )/ I_0) # hier ist I_0 jetzt 1pA
T_2_inv = 1 / T_clean2[mask_polarisation2]

params_pol2, cov_pol2 = curve_fit(lin_fit, T_2_inv, lnI_2)

m2_pol, b2_pol = get_fit_params(params_pol2, cov_pol2)
print(rf"m2_pol: {m2_pol} K")
print(rf"b2_pol: {b2_pol} ohne Einheit")
print(rf"W2_pol = {-m2_pol*k_B} eV")

#####################################################
# Stromdichtenansatz
#####################################################
print("Stromdichtenansatz:")
def IntTrapez(T, Strom):
    T_end = T[-1]
    integral = np.array([trapezoid(Strom[(T > t) & (T <= T_end)], T[(T > t) & (T <= T_end)])for t in T])
    return np.log(integral/Strom)

# Messung 1
print("erste Messung")
indices_stromdichte1 = np.concatenate([
    np.arange(33,58),
])

mask_stromdichte1 = np.zeros_like(I_clean1, dtype=bool)
mask_stromdichte1[indices_stromdichte1] = True

# Strom- und Temperaturdaten im Bereich des Stromdichtenansatzes
I_strom1 = I_clean1[mask_stromdichte1]
T_strom1 = T_clean1[mask_stromdichte1]


ln_integral1 = IntTrapez(T_strom1, I_strom1)
T_strom1 = T_strom1[np.isfinite(ln_integral1)]
I_strom1 = I_strom1[np.isfinite(ln_integral1)]
ln_integral1 = ln_integral1[np.isfinite(ln_integral1)]
T_inv1 = 1/T_strom1

params_strom1, cov_strom1 = curve_fit(lin_fit, T_inv1, ln_integral1)

m1_str, b1_str = get_fit_params(params_strom1, cov_strom1)

W1_str = m1_str*k_B

print(rf"m1_str: {m1_str} K")
print(rf"b1_str: {b1_str} keine Einheit")
print(rf"W1_str = {W1_str} eV")

# Messung 2
print("zweite Messung")
indices_stromdichte2 = np.concatenate([
    np.arange(18,26),
])

mask_stromdichte2 = np.zeros_like(I_clean2, dtype=bool)
mask_stromdichte2[indices_stromdichte2] = True


# zweite Messung
I_strom2 = I_clean2[mask_stromdichte2]
T_strom2 = T_clean2[mask_stromdichte2]

ln_integral2 = IntTrapez(T_strom2, I_strom2)
# print(ln_integral2)
T_strom2 = T_strom2[np.isfinite(ln_integral2)]
I_strom2 = I_strom2[np.isfinite(ln_integral2)]
ln_integral2 = ln_integral2[np.isfinite(ln_integral2)]
T_inv2 = 1/T_strom2

params_strom2, cov_strom2 = curve_fit(lin_fit, T_inv2, ln_integral2)

m2_str, b2_str = get_fit_params(params_strom2, cov_strom2)
W2_str = m2_str*k_B

print(rf"m2_str: {m2_str} K")
print(rf"b2_str: {b2_str} keine Einheit")
print(rf"W2_str = {W2_str} eV")


#####################################################
# Relaxationszeit
#####################################################
def tau_max(T, rate, W):
    return (T**2 * k_B)/(rate * W)

def tau0(tau, T, W):
    return tau * unp.exp(-W / (k_B * T))

print("Relaxationszeit:")

# erste Messreihe
print("erste Messreihe")
idx_max1 = np.argmax(I_strom1)
I_max1 = I_strom1[idx_max1]
T_max1 = T_strom1[idx_max1]

tau_max1 = tau_max(T_max1, m1_s, W1_str)
tau0_1 = tau0(tau_max1, T_max1, W1_str)

print(f"I_max = {I_max1} pA bei T_max = {T_max1} K")
print(rf"tau_max1: {tau_max1} s")
print(rf"tau0_1: {tau0_1} s")

# zweite Messreihe
print("zweite Messreihe")
idx_max2 = np.argmax(I_strom2)
I_max2 = I_strom2[idx_max2]
T_max2 = T_strom2[idx_max2]

tau_max2 = tau_max(T_max2, m2_s, W2_str)
tau0_2 = tau0(tau_max2, T_max2, W2_str)

print(f"I_max = {I_max2} pA bei T_max = {T_max2} K")
print(rf"tau_max2: {tau_max2} s")
print(rf"tau0_2: {tau0_2} s")
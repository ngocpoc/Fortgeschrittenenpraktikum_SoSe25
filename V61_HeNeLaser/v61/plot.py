import matplotlib.pyplot as plt
import numpy as np
from uncertainties.unumpy import uarray
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
import scipy.constants as const

#Frequenzmessung; Abstand zwischen den Frequenzen 
Frequenzen_1 = np.genfromtxt("Messwerte/Multimoden_1.txt", unpack=True)
Fehler_1 = np.ones(len(Frequenzen_1))
Frequenzen_mit_Fehler_1 = unp.uarray(Frequenzen_1,Fehler_1)
Laenge_1 = ufloat(76.1, 1) # in cm
Differenz_Frequenzen_1 = np.diff(Frequenzen_mit_Fehler_1) 
Differenz_Frequenzen_1_Mittel = np.sum(Differenz_Frequenzen_1)/len(Differenz_Frequenzen_1)  
Diff_F_fuer_plot_1 = 1/Differenz_Frequenzen_1_Mittel
#print("1.Messung:" )
#print(Differenz_Frequenzen_1)
#print(Differenz_Frequenzen_1_Mittel)

Frequenzen_2 = np.genfromtxt("Messwerte/Multimoden_2.txt", unpack=True)
Fehler_2 = np.ones(len(Frequenzen_2))
Frequenzen_mit_Fehler_2 = unp.uarray(Frequenzen_2,Fehler_2)
Laenge_2 = ufloat(92.6, 1) # in cm
Differenz_Frequenzen_2 = np.diff(Frequenzen_mit_Fehler_2) 
Differenz_Frequenzen_2_Mittel = np.sum(Differenz_Frequenzen_2)/len(Differenz_Frequenzen_2)  
Diff_F_fuer_plot_2 = 1/Differenz_Frequenzen_2_Mittel
#print("2.Messung:" )
#print(Differenz_Frequenzen_2)
#print(Differenz_Frequenzen_2_Mittel)

Frequenzen_3 = np.genfromtxt("Messwerte/Multimoden_3.txt", unpack=True)
Fehler_3 = np.ones(len(Frequenzen_3))
Frequenzen_mit_Fehler_3 = unp.uarray(Frequenzen_3,Fehler_3)
Laenge_3 = ufloat(109, 1) # in cm
Differenz_Frequenzen_3 = np.diff(Frequenzen_mit_Fehler_3) 
Differenz_Frequenzen_3_Mittel = np.sum(Differenz_Frequenzen_3)/len(Differenz_Frequenzen_3)  
Diff_F_fuer_plot_3 = 1/Differenz_Frequenzen_3_Mittel
#print("3.Messung:" )
#print(Differenz_Frequenzen_3)
#print(Differenz_Frequenzen_3_Mittel)

Frequenzen_4 = np.genfromtxt("Messwerte/Multimoden_4.txt", unpack=True)
Fehler_4 = np.ones(len(Frequenzen_4))
Frequenzen_mit_Fehler_4 = unp.uarray(Frequenzen_4,Fehler_4)
Laenge_4 = ufloat(160, 1) # in cm
Differenz_Frequenzen_4 = np.diff(Frequenzen_mit_Fehler_4) 
Differenz_Frequenzen_4_Mittel = np.sum(Differenz_Frequenzen_4)/len(Differenz_Frequenzen_4)  
Diff_F_fuer_plot_4 = 1/Differenz_Frequenzen_4_Mittel
#print("4.Messung:" )
#print(Differenz_Frequenzen_4)
#print(Differenz_Frequenzen_4_Mittel)

Frequenzen_5 = np.genfromtxt("Messwerte/Multimoden_5.txt", unpack=True)
Fehler_5 = np.ones(len(Frequenzen_5))
Frequenzen_mit_Fehler_5 = unp.uarray(Frequenzen_5,Fehler_5)
Laenge_5 = ufloat(205.3, 1) # in cm
Differenz_Frequenzen_5 = np.diff(Frequenzen_mit_Fehler_5) 
Differenz_Frequenzen_5_Mittel = np.sum(Differenz_Frequenzen_5)/len(Differenz_Frequenzen_5)  
Diff_F_fuer_plot_5 = 1/Differenz_Frequenzen_5_Mittel
#print("5.Messung:" )
#print(Differenz_Frequenzen_5)
#print(Differenz_Frequenzen_5_Mittel)

#Plot davon: 
Laengen = np.array([unp.nominal_values(Laenge_1), unp.nominal_values(Laenge_2), unp.nominal_values(Laenge_3), unp.nominal_values(Laenge_4), unp.nominal_values(Laenge_5)])
Mittelwerte_plot = np.array([unp.nominal_values(Diff_F_fuer_plot_1), unp.nominal_values(Diff_F_fuer_plot_2), unp.nominal_values(Diff_F_fuer_plot_3), unp.nominal_values(Diff_F_fuer_plot_4), unp.nominal_values(Diff_F_fuer_plot_5)])

Laengen_Fehler = np.array([unp.std_devs(Laenge_1), unp.std_devs(Laenge_2), unp.std_devs(Laenge_3), unp.std_devs(Laenge_4), unp.std_devs(Laenge_5)])
Mittelwerte_plot_Fehler = np.array([unp.std_devs(Diff_F_fuer_plot_1), unp.std_devs(Diff_F_fuer_plot_2), unp.std_devs(Diff_F_fuer_plot_3), unp.std_devs(Diff_F_fuer_plot_4), unp.std_devs(Diff_F_fuer_plot_5)])

x = np.linspace(75,210)
params, covariance_matrix = np.polyfit(Laengen, Mittelwerte_plot, deg=1, cov=True)
errors_Ausgleichsgerade = np.sqrt(np.diag(covariance_matrix))
#print("Ausgleichsgerade: ", params[0], " * x + ", params[1]) # Ausgleichsgerade:  6.711742498286715e-05  * x +  -4.955055243754489e-05


fig, (ax1) = plt.subplots(1, 1, layout="constrained")
plt.errorbar(Laengen, Mittelwerte_plot, yerr = Mittelwerte_plot_Fehler, xerr = Laengen_Fehler, fmt='o', alpha = 0.4, capsize = 3.3, color = "hotpink", label="Messwerte")
ax1.plot(
    x,
    params[0] * x + params[1],
    label="Lineare Regression",
    linewidth=1,
    color="blueviolet",
)
ax1.set_xlabel("Laenge")
ax1.set_ylabel("1/Frequenzen")
ax1.legend(loc="best")

fig.savefig("build/plot.pdf")

#Ausgabe: 
#1.Messung:
#[199.0+/-1.4142135623730951 195.0+/-1.4142135623730951
# 108.0+/-1.4142135623730951 289.0+/-1.4142135623730951
# 199.0+/-1.4142135623730951 195.0+/-1.4142135623730951]
#1.Mittelwert:197.50+/-0.24

#2.Messung:
#[161.0+/-1.4142135623730951 162.0+/-1.4142135623730951
# 147.0+/-1.4142135623730951 179.0+/-1.4142135623730951
# 165.0+/-1.4142135623730951 161.0+/-1.4142135623730951
# 161.0+/-1.4142135623730951 165.0+/-1.4142135623730951]
#2.Mittelwert: 162.62+/-0.18

#3.Messung:
#[139.0+/-1.4142135623730951 135.0+/-1.4142135623730951
# 138.0+/-1.4142135623730951 139.0+/-1.4142135623730951
# 135.0+/-1.4142135623730951 139.0+/-1.4142135623730951
# 141.0+/-1.4142135623730951 136.0+/-1.4142135623730951
# 135.0+/-1.4142135623730951]
#3.Mittelwert: 137.44+/-0.16

#4.Messung:
#[94.0+/-1.4142135623730951 93.0+/-1.4142135623730951
# 94.0+/-1.4142135623730951 94.0+/-1.4142135623730951
# 94.0+/-1.4142135623730951 93.0+/-1.4142135623730951
# 94.0+/-1.4142135623730951 94.0+/-1.4142135623730951
# 94.0+/-1.4142135623730951 93.0+/-1.4142135623730951
# 94.0+/-1.4142135623730951 94.0+/-1.4142135623730951
# 94.0+/-1.4142135623730951 90.0+/-1.4142135623730951]
#4.Mittelwert: 93.50+/-0.10

#5.Messung:
#[71.0+/-1.4142135623730951 75.0+/-1.4142135623730951
# 72.0+/-1.4142135623730951 75.0+/-1.4142135623730951
# 71.0+/-1.4142135623730951 75.0+/-1.4142135623730951
# 71.0+/-1.4142135623730951 71.0+/-1.4142135623730951
# 75.0+/-1.4142135623730951 83.0+/-1.4142135623730951
# 64.0+/-1.4142135623730951 75.0+/-1.4142135623730951
# 71.0+/-1.4142135623730951 71.0+/-1.4142135623730951]
#5.Mittelwert: 72.86+/-0.10











def rel_Abweichung(exp, theo):
    return (np.abs(exp-theo)/(theo)*100) #ist schon in Prozent

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
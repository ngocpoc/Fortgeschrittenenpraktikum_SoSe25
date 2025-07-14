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

fig.savefig("build/Frequenzspektrum.pdf")

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

#Moden des Lasers (00 und 01) x = r !!!!
#Anfang mit 00er Mode 

x_mm_00, I_muA_00 = np.genfromtxt("Messwerte/Mode_00.txt", unpack=True)

def Gaus(x,I_0,x_0,w):
    return I_0*np.exp(-2 *((x+x_0)**2)/(w**2))

params2, covariance_matrix2 = curve_fit(Gaus,  x_mm_00,  I_muA_00) #p0 = (1000,2,4)

uncertainties2 = np.sqrt(np.diag(covariance_matrix2))

for name, value, uncertainty in zip("Ixw", params2, uncertainties2):
    print(f"{name} = {value:8.4f} ± {uncertainty:.4f}")  

#I_0 = 13.3068 ± 0.1892  
#x_0 = -3.9828 ± 0.0791  
#w =    9.6445 ± 0.1584  

x_plot_2 = np.linspace(-12,20,1000)

fig, (ax1) = plt.subplots(1, 1, layout="constrained")
#plt.errorbar(Laengen, Mittelwerte_plot, yerr = Mittelwerte_plot_Fehler, xerr = Laengen_Fehler, fmt='o', alpha = 0.4, capsize = 3.3, color = "hotpink", label="Messwerte")
ax1.plot(
    x_mm_00,
    I_muA_00,
    "x",
    label="Messwerte",
    color="blueviolet",
)
ax1.plot(
    x_plot_2,
    Gaus(x_plot_2,params2[0],params2[1],params2[2]),
    label="Ausgelichskurve",
    color="hotpink",
)
ax1.set_xlabel("x/mm")
ax1.set_ylabel("I /muA")
ax1.legend(loc="best")

fig.savefig("build/Mode_00.pdf")

#Jetzt 01 Mode 

x_mm_01, I_muA_01 = np.genfromtxt("Messwerte/Mode_01.txt", unpack=True)

def Gaus01(x,I_0,x_0,w):
    return I_0*(((x+x_0)/w)**2)*np.exp(-2 *((x+x_0)**2)/(w**2))

params3, covariance_matrix3 = curve_fit(Gaus01,  x_mm_01,  I_muA_01, p0 = (3.5,2.5,15)) #

uncertainties3 = np.sqrt(np.diag(covariance_matrix3))

for name, value, uncertainty in zip("Ixw", params3, uncertainties3):
    print(f"{name} = {value:8.4f} ± {uncertainty:.4f}")  

#I_01 = 17.3682 ± 0.4487 
#x_01 = -2.9582 ± 0.1126 
#w_01 =  9.7969 ± 0.1565 

x_plot_3 = np.linspace(-14,20,1000)

fig, (ax1) = plt.subplots(1, 1, layout="constrained")
#plt.errorbar(Laengen, Mittelwerte_plot, yerr = Mittelwerte_plot_Fehler, xerr = Laengen_Fehler, fmt='o', alpha = 0.4, capsize = 3.3, color = "hotpink", label="Messwerte")
ax1.plot(
    x_mm_01,
    I_muA_01,
    "x",
    label="Messwerte",
    color="blueviolet",
)
ax1.plot(
    x_plot_3,
    Gaus01(x_plot_3,params3[0],params3[1],params3[2]),
    label="Ausgleichskurve",
    color="hotpink",
)
ax1.set_xlabel("x/mm")
ax1.set_ylabel("I /muA")
ax1.legend(loc="best")

fig.savefig("build/Mode_01.pdf")


#Polarisation des Lasers RIP ich hasse Polarplots 

theta_Grad, I_muA_Pol = np.genfromtxt("Messwerte/Polarisation.txt", unpack=True)

theta_Rad = np.radians(theta_Grad)

def Malus(theta, theta0, I_0):
    return I_0 * np.cos((theta - theta0))** 2 

params4, covariance_matrix4 = curve_fit(Malus,  theta_Rad,  I_muA_Pol, p0 = (np.pi/2,7)) #

uncertainties4 = np.sqrt(np.diag(covariance_matrix4))

for name, value, uncertainty in zip("tI", params4, uncertainties4):
    print(f"{name} = {value:8.4f} ± {uncertainty:.4f}")  

#theta_0_Rad =   1.5043 ± 0.0106
#I_0 =   5.7701 ± 0.0705

x_Pol_plot = np.linspace(0,2*np.pi, 1000)

fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.plot(
    theta_Rad, 
    I_muA_Pol, 
    "x", 
    label = "Messwerte")
ax.plot(
    x_Pol_plot,
    Malus(x_Pol_plot,params4[0],params4[1]),
    label = "Ausgleichskurve"
)
ax.legend(loc="best")

fig.savefig("build/Polarisation.pdf")

#Wellenlänge des Lasers 
x_mm_Wellen, I_muA_Wellen  = np.genfromtxt("Messwerte/Wellenlaenge.txt", unpack=True)

fig, (ax1) = plt.subplots(1, 1, layout="constrained")
ax1.plot(
    x_mm_Wellen,
    I_muA_Wellen,
    "x",
    label="Messwerte",
    color="blueviolet",
)
ax1.set_xlabel("x/mm")
ax1.set_ylabel("I /muA")
ax1.legend(loc="best")
fig.savefig("build/Wellenlaenge.pdf")

d = 0.001 #in cm 10 mumeter (aus anderem Protokoll geklaut)
L = 20 #in cmeter
a_1 = 1.05 #in cm 10.5 mmeter
a_2 = 1.05 #in cm 10.5 mmeter
n = 1
#-2              7.56
#8.5             105.34
#19              34.41
lamda_1 = (a_1*d)/(n * L)
lamda_2 = (a_2*d)/(n * L)
print("lamda", lamda_1) # lamda = 5.2500000000000006e-05 cm 10^-2 nanometer 10^-9 --> 525 nm 

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
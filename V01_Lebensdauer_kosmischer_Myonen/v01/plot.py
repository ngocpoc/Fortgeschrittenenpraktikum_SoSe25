import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit

delay, Anzahl_Delay_pro_30_s = np.genfromtxt("Messdaten/Variation_Verzoegerungsleitung.txt", unpack=True)

Anzahl_Delay_pro_1_s = Anzahl_Delay_pro_30_s/30

def Fehler_bei_N(N):
    return np.sqrt(N)

Anzahl_Delay_pro_1_s_mit_Fehler = unp.uarray(Anzahl_Delay_pro_1_s,Fehler_bei_N(Anzahl_Delay_pro_1_s)) 

#print(max(Anzahl_Delay_pro_1_s)/2)


# Slice from index 1 to index 5
Anzahl_Delay_pro_1_s_Ausgleichskonstante = Anzahl_Delay_pro_1_s[19:24]
delay_Ausgleichskonstante = delay[19:24]
#print(Anzahl_Delay_pro_1_s)

Steigung_1 = (Anzahl_Delay_pro_1_s[16] - Anzahl_Delay_pro_1_s[15])/(delay[16] - delay[15])
yAchsenabschnitt_1 = Anzahl_Delay_pro_1_s[16] - Steigung_1 * delay[16]

Steigung_2 = (Anzahl_Delay_pro_1_s[27] - Anzahl_Delay_pro_1_s[26])/(delay[27] - delay[26])
yAchsenabschnitt_2 = Anzahl_Delay_pro_1_s[27] - Steigung_2 * delay[27]

halbwerts_links = ((max(Anzahl_Delay_pro_1_s)/2) - yAchsenabschnitt_1)/Steigung_1
halbwerts_rechts = ((max(Anzahl_Delay_pro_1_s)/2) - yAchsenabschnitt_2)/Steigung_2

params, covariance_matrix = np.polyfit(delay_Ausgleichskonstante, Anzahl_Delay_pro_1_s_Ausgleichskonstante, deg=0, cov=True)

Halbwertsbreite_gesamt = abs(halbwerts_links) + abs(halbwerts_rechts)
aufloesezeit = abs(20 - Halbwertsbreite_gesamt)
print("halbwerts_links: ", halbwerts_links)
print("halbwerts_rechts: ", halbwerts_rechts)
print("Halbwertsbreite_gesamt: ", Halbwertsbreite_gesamt)
print("aufloesezeit: ", aufloesezeit)

fig,ax = plt.subplots(figsize = (6,5))
ax.errorbar(delay,Anzahl_Delay_pro_1_s, yerr=Fehler_bei_N(Anzahl_Delay_pro_1_s), fmt="o", alpha = 0.4, capsize = 3.3, label = "Messwerte")
ax.plot(delay,Anzahl_Delay_pro_1_s, "o", mfc='none', color='steelblue')
#plt.plot(
#    x_plot,
#    params[0] * x_plot + params[1],
#    label ="Lineare Regression",
#    linewidth=1,
#)
#plt.grid()
#plt.plot (-5, Anzahl_Delay_pro_1_s, "x", label = "Messwerte")
ax.hlines(y=max(Anzahl_Delay_pro_1_s)/2, xmin=-5, xmax=7, linewidth=2, color='r', linestyle = "dashed", label = "Halbwertsbreite")
ax.hlines(y=params[0], xmin=-1.5, xmax=3.5, linewidth=2, color='r', linestyle = "solid", label = "Ausgleichskonstante")
ax.axvline(x=halbwerts_links,ymin = 0, ymax = 0.8, linestyle = "dashed", color="violet")
ax.axvline(x=halbwerts_rechts,ymin = 0, ymax = 0.8, linestyle = "dashed", color="violet")
#plt.axhline(y=, color='r', linestyle='-')
ax.set_xlabel("Verzögerung in ns")
ax.set_ylabel(r"Anzahl in $1 / \mathrm{s}$")
ax.legend(loc = "best")
#plt.margins(0.075)
fig.savefig("Verzoegerung.pdf")
#Auflösungszeit ist apparently Betrag(2 * Breite der Pulse (10 ns bei uns) - Halbwertsbreite (die ist 11.443766937669375)) = 8.556233062330625


#Kalibrierung des Multichannel-Analyzers

Kanalnummer, Zeitdifferenz_in_mikrosek = np.genfromtxt("Messdaten/Kalibrierung_Multichannel_Auswertung.txt", unpack=True)

params1, covariance_matrix1 = np.polyfit(Kanalnummer, Zeitdifferenz_in_mikrosek, deg=1, cov=True)

print("n ist: ", params1[1])
print("m ist: ", params1[0])

#n ist:  0.09696833488281219
#m ist:  0.02171211931450518

x_plot = np.linspace(0,400) 

fig1,ax1 = plt.subplots(figsize = (6,5))
ax1.plot(Kanalnummer,Zeitdifferenz_in_mikrosek, "x", mfc='none', label = "Messwerte")
plt.plot(
    x_plot,
    params1[0] * x_plot + params1[1],
    label ="Lineare Regression",
    linewidth=1,
)
ax1.set_xlabel("Kanalnummer")
ax1.set_ylabel(r"Zeitdifferenz in $\mu\mathrm{s}$")
ax1.legend(loc = "best")
#plt.margins(0.075)
fig1.savefig("Kalibrierung_MUltichannel.pdf")

#ax.set_xlabel(r"$t / \mathrm{s}$")
#ax.set_ylabel(r"$U / \mathrm{V}$");

#Berechnung der Lebensdauer der Myonen 

Counts_pro_Channel = np.genfromtxt("Messdaten/Lebendauer_Berechnung.txt", unpack=True)
channelnumber = np.array(range(1,413))

#print(channelnumber)
Lebensdauer_Messwerte = params1[0] * channelnumber + params1[1]


#print(Lebensdauer_Messwerte)
Counts_pro_Channel_fit = Counts_pro_Channel[16:]
Lebensdauer_Messwerte_fit = Lebensdauer_Messwerte[16:]
print(len(Counts_pro_Channel_fit), len(Lebensdauer_Messwerte_fit))
def exp(x,a,b,U):
    return a*np.exp(-b*x) + U

#scipy.optimize.curve_fit lambda x,a,b,U: a*np.exp(-b*x) + U
params2, covariance_matrix2 = curve_fit(exp,  Lebensdauer_Messwerte_fit,  Counts_pro_Channel_fit, p0 = (2,2,2))

uncertainties = np.sqrt(np.diag(covariance_matrix2))

for name, value, uncertainty in zip("abc", params2, uncertainties):
    print(f"{name} = {value:8.3f} ± {uncertainty:.3f}")

#a =  306.815 ± 43.820
#b =    2.762 ± 0.221
#U =    3.770 ± 0.531

x_plot2 = np.linspace(0.1,9)
a = params2[0]
b = params2[1]
U = params2[2]

fig2,ax2 = plt.subplots(figsize = (6,5))
ax2.plot(Lebensdauer_Messwerte,Counts_pro_Channel, ".", mfc='none', label = "Messwerte")
ax2.plot(x_plot2, a*np.exp(-b*x_plot2) + U, label = "Ausgleichskurve" )
ax2.set_xlabel(r"t in $\mu\mathrm{s}$")
ax2.set_ylabel("Anzahl")
ax2.legend(loc = "best")
#plt.margins(0.075)
fig2.savefig("Lebensdauer_der_Myonen.pdf")

#params, covariance_matrix = np.polyfit(x, y, deg=0, cov=True)

#params, covariance_matrix = np.polyfit(x, y, deg=1, cov=True)

#d_kl = ufloat(15.57,0.01) * 10**-1 # in cm (gemessen)
#t_kl_r0, t_kl_h0 = np.genfromtxt("kl_Kugel.txt", unpack=True)

# Plots
#x_err = 1/temp
#y_err = unp.log(eta_T)
#x = 1/unp.nominal_values(temp)
#y = np.log(unp.nominal_values(eta_T))

#print (f"temp: {temp}")
#print (f"y: {y}")
#params, covariance_matrix = np.polyfit(x, y, deg=1, cov=True)
##params, covariance_matrix = np.polyfit(x, y, deg=1, cov=True)
#errors = np.sqrt(np.diag(covariance_matrix))
#plt.plot (x, y, "x", label = "Messwerte")
#plt.errorbar(x,y, xerr=unp.std_devs(x_err), yerr=unp.std_devs(y_err), fmt="x", label = "Messwerte")
#plt.plot(
#    x_plot,
#    params[0] * x_plot + params[1],
#    label ="Lineare Regression",
#    linewidth=1,
#)
#plt.grid()
#plt.xlabel(r'$T^{-1}$ [K $^{-1}$]')
#plt.ylabel(r"$\ln{ \left ( \eta \right )}$")
#plt.legend(loc = "best")
#plt.margins(0.075)
#plt.savefig("plot.pdf")

#def rel_Abweichung(exp, theo):
#    return (np.abs(exp-theo)/(theo)*100) #ist schon in Prozent

#x = np.linspace(0, 10, 1000)
#y = x ** np.sin(x)

#fig, (ax1, ax2) = plt.subplots(1, 2, layout="constrained")
#ax1.plot(x, y, label="Kurve")
#ax1.set_xlabel(r"$\alpha \mathbin{/} \unit{\ohm}$")
#ax1.set_ylabel(r"$y \mathbin{/} \unit{\micro\joule}$")
#ax1.legend(loc="best")

#ax2.plot(x, y, label="Kurve")
#ax2.set_xlabel(r"$\alpha \mathbin{/} \unit{\ohm}$")
#ax2.set_ylabel(r"$y \mathbin{/} \unit{\micro\joule}$")
#ax2.legend(loc="best")

#fig.savefig("build/plot.pdf")
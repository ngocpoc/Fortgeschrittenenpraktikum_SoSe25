import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit

delay, Anzahl_Delay_pro_30_s = np.genfromtxt("Messdaten/Variation_Verzoegerungsleitung.txt", unpack=True)

Anzahl_Delay_pro_1_s = Anzahl_Delay_pro_30_s/30

print("Anzahl_Delay_pro_1_s: " , Anzahl_Delay_pro_1_s)
def Fehler_bei_N(N):
    return np.sqrt(N)

Anzahl_Delay_pro_1_s_mit_Fehler = unp.uarray(Anzahl_Delay_pro_1_s,Fehler_bei_N(Anzahl_Delay_pro_1_s)) 

#print(max(Anzahl_Delay_pro_1_s)/2)
print("Fehler bei N: ",Fehler_bei_N(Anzahl_Delay_pro_1_s))

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

uncertainties = np.sqrt(np.diag(covariance_matrix))
print("uncertenty 1. Plot", uncertainties)


Halbwertsbreite_gesamt = abs(halbwerts_links) + abs(halbwerts_rechts)
aufloesezeit = abs(20 - Halbwertsbreite_gesamt)
print("halbwerts_links: ", halbwerts_links)
print("halbwerts_rechts: ", halbwerts_rechts)
print("Halbwertsbreite_gesamt: ", Halbwertsbreite_gesamt)
print("aufloesezeit: ", aufloesezeit)
print("Höhe Ausgleichskonstante: ", params[0]) #11.266666666666666

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
ax.set_xlabel(r"Verzögerung $\symup{\Delta} t$ in $\unit{\nano\second}$")
ax.set_ylabel(r"Counts $N$ in $1 / \mathrm{s}$")
ax.legend(loc = "best")
plt.grid()
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

uncertainties1 = np.sqrt(np.diag(covariance_matrix1))

for name, value, uncertainty in zip("mn", params1, uncertainties1):
    print(f"{name} = {value:8.4f} ± {uncertainty:.4f}")

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
plt.grid()
fig1.savefig("Kalibrierung_MUltichannel.pdf")

#ax.set_xlabel(r"$t / \mathrm{s}$")
#ax.set_ylabel(r"$U / \mathrm{V}$");

#Berechnung der Lebensdauer der Myonen 

Counts_pro_Channel = np.genfromtxt("Messdaten/Lebendauer_Berechnung.txt", unpack=True)
channelnumber = np.array(range(1,513)) #beeinhaltet die Zahlen 1 bis inklusive 512

#print(channelnumber)
Lebensdauer_Messwerte = params1[0] * channelnumber + params1[1]


#print(Lebensdauer_Messwerte)

#Sachen mit falschem Peak drin
#Counts_pro_Channel_fit = Counts_pro_Channel[17:412]
#Lebensdauer_Messwerte_fit = Lebensdauer_Messwerte[17:412]
#Counts_pro_Channel_ex_begin = Counts_pro_Channel[:17]
#Lebensdauer_Messwerte_ex_begin = Lebensdauer_Messwerte[:17]

#Sachen ohne falschen Peak
Counts_pro_Channel_fit = np.append(Counts_pro_Channel[4:18], Counts_pro_Channel[27:412])
Lebensdauer_Messwerte_fit = np.append(Lebensdauer_Messwerte[4:18], Lebensdauer_Messwerte[27:412])
Counts_pro_Channel_ex = np.append(np.append(Counts_pro_Channel[:4],Counts_pro_Channel[18:27]),Counts_pro_Channel[412:])
Lebensdauer_Messwerte_ex = np.append(np.append(Lebensdauer_Messwerte[:4],Lebensdauer_Messwerte[18:27]),Lebensdauer_Messwerte[412:])

#print("Messwerte vorher: ",len(Lebensdauer_Messwerte_fit) )
#i = 1
#while(i < (len(Counts_pro_Channel_fit))):
#    if Counts_pro_Channel_fit[i] == 0:
#        np.append(Counts_pro_Channel_ex_begin, Counts_pro_Channel_fit[i])
#        np.append(Lebensdauer_Messwerte_ex_begin, Lebensdauer_Messwerte_fit[i])
#        np.delete(Counts_pro_Channel_fit, i, axis = None)
#        np.delete(Lebensdauer_Messwerte_fit, i, axis = None)
#        #print("wurde entfernt")
#    i = i + 1
#    #print("Hat funktioniert")

#print("Messwerte nachher: ", Counts_pro_Channel_fit )

#Counts_pro_Channel_ex = np.append(Counts_pro_Channel_ex_begin, Counts_pro_Channel[412:])
#Lebensdauer_Messwerte_ex = np.append(Lebensdauer_Messwerte_ex_begin, Lebensdauer_Messwerte[412:])

no_zero_counts = np.copy(Counts_pro_Channel_fit[Counts_pro_Channel_fit>0])
no_zero_counts = no_zero_counts[14:]
no_zero_messwerte = np.copy(Lebensdauer_Messwerte_fit[Counts_pro_Channel_fit>0])
no_zero_messwerte = no_zero_messwerte[14:]

#print(Counts_pro_Channel_fit>0)


#print(len(Counts_pro_Channel_fit), len(Lebensdauer_Messwerte_fit))

def exp(x,a,b,U):
    return a*np.exp(-b*x) + U

x_plot12 = np.linspace(-0.5,2)

params_linear, covariance_matrix_linear = np.polyfit(np.log(no_zero_messwerte), np.log(no_zero_counts), deg=1, cov=True)



fig12,ax12 = plt.subplots(figsize = (6,5))
ax12.plot(np.log(no_zero_messwerte), np.log(no_zero_counts), "x")
ax12.plot(x_plot12, params_linear[0] * x_plot12 + params_linear[1])
plt.grid()
#plt.errorbar()
fig12.savefig("Linearisierung.pdf")


#print("params_linear[0](m)",params_linear[0])
#print("params_linear[1]",params_linear[1])
#uncertainties_linear = np.sqrt(np.diag(covariance_matrix_linear))
#for name, value, uncertainty in zip("ln", params_linear, uncertainties_linear):
#    print(f"{name} = {value:8.4f} ± {uncertainty:.4f}")

#scipy.optimize.curve_fit lambda x,a,b,U: a*np.exp(-b*x) + U
params2, covariance_matrix2 = curve_fit(exp,  Lebensdauer_Messwerte_fit,  Counts_pro_Channel_fit, p0 = (1000,2,4))

uncertainties2 = np.sqrt(np.diag(covariance_matrix2))

for name, value, uncertainty in zip("abU", params2, uncertainties2):
    print(f"{name} = {value:8.4f} ± {uncertainty:.4f}")

#l =  -1.3707 ± 0.0381
#n =   3.2112 ± 0.0557

#N_0 = 306.8146 ± 43.8197
#lambda =   2.7618 ± 0.2214
#U =   3.7704 ± 0.5308

#a = 2153.7204 ± 387.5688
#b =   5.5514 ± 0.3042
#U =   4.5081 ± 0.3879

x_plot2 = np.linspace(0,11.5, 1000)
a = params2[0]
b = params2[1]
U = params2[2]

lamdba = ufloat(b,uncertainties2[1])
tau = 1/lamdba

Fehler_bei_Counts_fit = np.sqrt(Counts_pro_Channel_fit)
Fehler_bei_Counts_ex = np.sqrt(Counts_pro_Channel_ex)

fig2,ax2 = plt.subplots(figsize = (6,5))
ax2.errorbar(x = Lebensdauer_Messwerte_fit,y = Counts_pro_Channel_fit, xerr = None, yerr=Fehler_bei_N(Counts_pro_Channel_fit), fmt="o", alpha = 0.4, capsize = 3.3, color = "mediumpurple", label = "inkludierte Messwerte")
ax2.errorbar(x = Lebensdauer_Messwerte_ex,y = Counts_pro_Channel_ex, xerr = None, yerr=Fehler_bei_N(Counts_pro_Channel_ex), fmt="o", alpha = 0.4, capsize = 3.3, color = "hotpink", label = "exkludierte Messwerte")
#ax2.plot(Lebensdauer_Messwerte_fit,Counts_pro_Channel_fit, ".", mfc='none', color = "mediumpurple", label = "inkludierte Messwerte")
#ax2.plot(Lebensdauer_Messwerte_ex,Counts_pro_Channel_ex, ".", mfc='none', color = "hotpink", label = "exkludierte Messwerte")
# ax2.plot(x_plot2, a*np.exp(-b*x_plot2) + U, color = "darkorange", label = "Ausgleichsfunktion" )
ax2.plot(x_plot2, exp(x_plot2,*params2), color = "darkblue", label = "Ausgleichsfunktion")
#ax2.plot(x_plot2, exp(x_plot2, np.exp(3.2112),1.3707,4.5081), label = "andere Ausgleichsfunktion")
#plt.errorbar()
ax2.set(
    xlabel=r"Zeit $t$ in $\mu\mathrm{s}$",
    ylabel="Counts",
    ylim =[-5, 200],
)
ax2.legend(loc = "best")
#plt.margins(0.075)
plt.grid()
fig2.savefig("Lebensdauer_der_Myonen.pdf")

print("Tau is equal to: ",tau )


#uncertenty 1. Plot [0.20682789]
#halbwerts_links:  -4.638888888888888
#halbwerts_rechts:  6.804878048780487
#Halbwertsbreite_gesamt:  11.443766937669375
#aufloesezeit:  8.556233062330625
#n ist:  0.09650695845298125
#m ist:  0.021712827103746585
#m =   0.0217 ± 0.0000
#n =   0.0965 ± 0.0033
#396 396
#a = 306.8146 ± 43.8197
#b =   2.7618 ± 0.2214
#c =   3.7704 ± 0.5308







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
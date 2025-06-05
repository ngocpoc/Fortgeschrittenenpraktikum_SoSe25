import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.stats import sem

#x = [1, 2, 3, 4, 5]
#err = [0.1, 0.3, 0.1, 0.8, 1.0]

#y = unp.uarray(x, err)
#s = unp.cos(y)
#print(s)
def rel_Abweichung(exp, theo):
    return (np.abs(exp-theo)/(theo)*100) #ist schon in Prozent

B1_ho_An_x_pix, B1_ho_An_y_pix, B1_ho_En_x_pix, B1_ho_En_y_pix, B1_ho_Anzahl, B1_ve_An_x_pix, B1_ve_An_y_pix, B1_ve_En_x_pix, B1_ve_En_y_pix, B1_ve_Anzahl = np.genfromtxt("Messdaten/1482_graphische_Messdaten_1_Bild.txt", unpack=True)
B2_ho_An_x_pix, B2_ho_An_y_pix, B2_ho_En_x_pix, B2_ho_En_y_pix, B2_ho_Anzahl, B2_ve_An_x_pix, B2_ve_An_y_pix, B2_ve_En_x_pix, B2_ve_En_y_pix, B2_ve_Anzahl = np.genfromtxt("Messdaten/1482_graphische_Messdaten_2_Bild.txt", unpack=True)

B3_ho_An_x_pix, B3_ho_An_y_pix, B3_ho_En_x_pix, B3_ho_En_y_pix, B3_ho_Anzahl, B3_ve_An_x_pix, B3_ve_An_y_pix, B3_ve_En_x_pix, B3_ve_En_y_pix, B3_ve_Anzahl = np.genfromtxt("Messdaten/1490_graphische_Messdaten_1_Bild.txt", unpack=True)
B4_ho_An_x_pix, B4_ho_An_y_pix, B4_ho_En_x_pix, B4_ho_En_y_pix, B4_ho_Anzahl, B4_ve_An_x_pix, B4_ve_An_y_pix, B4_ve_En_x_pix, B4_ve_En_y_pix, B4_ve_Anzahl = np.genfromtxt("Messdaten/1490_graphische_Messdaten_2_Bild.txt", unpack=True)

#Fehler einführen 
Fehler = [2,2,2,2]
B1_ho_An_x_pix_Fehler = unp.uarray(B1_ho_An_x_pix, Fehler)
B1_ho_An_y_pix_Fehler = unp.uarray(B1_ho_An_y_pix, Fehler)

B1_ho_En_x_pix_Fehler = unp.uarray(B1_ho_En_x_pix, Fehler)
B1_ho_En_y_pix_Fehler = unp.uarray(B1_ho_En_y_pix, Fehler)

B1_ve_An_x_pix_Fehler = unp.uarray(B1_ve_An_x_pix, Fehler)
B1_ve_An_y_pix_Fehler = unp.uarray(B1_ve_An_y_pix, Fehler)

B1_ve_En_x_pix_Fehler = unp.uarray(B1_ve_En_x_pix, Fehler)
B1_ve_En_y_pix_Fehler = unp.uarray(B1_ve_En_y_pix, Fehler)

#Bild 2
B2_ho_An_x_pix_Fehler = unp.uarray(B2_ho_An_x_pix, Fehler)
B2_ho_An_y_pix_Fehler = unp.uarray(B2_ho_An_y_pix, Fehler)

B2_ho_En_x_pix_Fehler = unp.uarray(B2_ho_En_x_pix, Fehler)
B2_ho_En_y_pix_Fehler = unp.uarray(B2_ho_En_y_pix, Fehler)

B2_ve_An_x_pix_Fehler = unp.uarray(B2_ve_An_x_pix, Fehler)
B2_ve_An_y_pix_Fehler = unp.uarray(B2_ve_An_y_pix, Fehler)

B2_ve_En_x_pix_Fehler = unp.uarray(B2_ve_En_x_pix, Fehler)
B2_ve_En_y_pix_Fehler = unp.uarray(B2_ve_En_y_pix, Fehler)

#Bild 3
B3_ho_An_x_pix_Fehler = unp.uarray(B3_ho_An_x_pix, Fehler)
B3_ho_An_y_pix_Fehler = unp.uarray(B3_ho_An_y_pix, Fehler)

B3_ho_En_x_pix_Fehler = unp.uarray(B3_ho_En_x_pix, Fehler)
B3_ho_En_y_pix_Fehler = unp.uarray(B3_ho_En_y_pix, Fehler)

B3_ve_An_x_pix_Fehler = unp.uarray(B3_ve_An_x_pix, Fehler)
B3_ve_An_y_pix_Fehler = unp.uarray(B3_ve_An_y_pix, Fehler)

B3_ve_En_x_pix_Fehler = unp.uarray(B3_ve_En_x_pix, Fehler)
B3_ve_En_y_pix_Fehler = unp.uarray(B3_ve_En_y_pix, Fehler)

#Bild 4
B4_ho_An_x_pix_Fehler = unp.uarray(B4_ho_An_x_pix, Fehler)
B4_ho_An_y_pix_Fehler = unp.uarray(B4_ho_An_y_pix, Fehler)

B4_ho_En_x_pix_Fehler = unp.uarray(B4_ho_En_x_pix, Fehler)
B4_ho_En_y_pix_Fehler = unp.uarray(B4_ho_En_y_pix, Fehler)

B4_ve_An_x_pix_Fehler = unp.uarray(B4_ve_An_x_pix, Fehler)
B4_ve_An_y_pix_Fehler = unp.uarray(B4_ve_An_y_pix, Fehler)

B4_ve_En_x_pix_Fehler = unp.uarray(B4_ve_En_x_pix, Fehler)
B4_ve_En_y_pix_Fehler = unp.uarray(B4_ve_En_y_pix, Fehler)



#in nm scale bringen 
k = 2.31/256
B1_ho_An_x = k * B1_ho_An_x_pix_Fehler
B1_ho_An_y = k * B1_ho_An_y_pix_Fehler 
B1_ho_En_x = k * B1_ho_En_x_pix_Fehler 
B1_ho_En_y = k * B1_ho_En_y_pix_Fehler 

B1_ve_An_x = k * B1_ve_An_x_pix_Fehler 
B1_ve_An_y = k * B1_ve_An_y_pix_Fehler 
B1_ve_En_x = k * B1_ve_En_x_pix_Fehler
B1_ve_En_y = k * B1_ve_En_y_pix_Fehler

B2_ho_An_x = k * B2_ho_An_x_pix_Fehler
B2_ho_An_y = k * B2_ho_An_y_pix_Fehler 
B2_ho_En_x = k * B2_ho_En_x_pix_Fehler 
B2_ho_En_y = k * B2_ho_En_y_pix_Fehler 

B2_ve_An_x = k * B2_ve_An_x_pix_Fehler
B2_ve_An_y = k * B2_ve_An_y_pix_Fehler
B2_ve_En_x = k * B2_ve_En_x_pix_Fehler
B2_ve_En_y = k * B2_ve_En_y_pix_Fehler



B3_ho_An_x = k * B3_ho_An_x_pix_Fehler
B3_ho_An_y = k * B3_ho_An_y_pix_Fehler 
B3_ho_En_x = k * B3_ho_En_x_pix_Fehler 
B3_ho_En_y = k * B3_ho_En_y_pix_Fehler 

B3_ve_An_x = k * B3_ve_An_x_pix_Fehler
B3_ve_An_y = k * B3_ve_An_y_pix_Fehler
B3_ve_En_x = k * B3_ve_En_x_pix_Fehler
B3_ve_En_y = k * B3_ve_En_y_pix_Fehler

B4_ho_An_x = k * B4_ho_An_x_pix_Fehler
B4_ho_An_y = k * B4_ho_An_y_pix_Fehler
B4_ho_En_x = k * B4_ho_En_x_pix_Fehler
B4_ho_En_y = k * B4_ho_En_y_pix_Fehler

B4_ve_An_x = k * B4_ve_An_x_pix_Fehler
B4_ve_An_y = k * B4_ve_An_y_pix_Fehler
B4_ve_En_x = k * B4_ve_En_x_pix_Fehler
B4_ve_En_y = k * B4_ve_En_y_pix_Fehler



#Berechnung der Länge 
B1_ho_Laenge = unp.sqrt((B1_ho_En_x - B1_ho_An_x)**2 + (B1_ho_En_y - B1_ho_An_y)**2)/B1_ho_Anzahl 
#[30.6022875  30.8130564  30.3747428  30.36906321] in pixeln
#[0.27613783 0.27803969 0.27408459 0.27403334] in nm 
#[0.2761378285789806+/-0.012761067691725974
 #0.27803968857055306+/-0.004253689230575325
 #0.2740845932096956+/-0.006380533845862987
 #0.27403334377140154+/-0.005104427076690391]

#print(B1_ho_Laenge)
B1_ve_Laenge = unp.sqrt((B1_ve_En_x - B1_ve_An_x)**2 + (B1_ve_En_y - B1_ve_An_y)**2)/B1_ve_Anzahl 
#print(B1_ve_Laenge)
# [0.17506399 0.16962984 0.16928342 0.17334925]
#[0.17506398976572912+/-0.005104427076690391
 #0.16962984014427404+/-0.0028357928203835503
 #0.16928342117063672+/-0.0031902669229314936
 #0.1733492528806331+/-0.006380533845862988]

B2_ho_Laenge = unp.sqrt((B2_ho_En_x - B2_ho_An_x)**2 + (B2_ho_En_y - B2_ho_An_y)**2)/B2_ho_Anzahl 
print("B2_ho_Laenge ", B2_ho_Laenge)
#[0.27762289 0.27236901 0.2732685  0.27803969]
B2_ve_Laenge = unp.sqrt((B2_ve_En_x - B2_ve_An_x)**2 + (B2_ve_En_y - B2_ve_An_y)**2)/B2_ve_Anzahl 
print(B2_ve_Laenge)
#[0.17318419 0.17150467 0.17076383 0.17145704]

#2. Paar Bilder
B3_ho_Laenge = unp.sqrt((B3_ho_En_x - B3_ho_An_x)**2 + (B3_ho_En_y - B3_ho_An_y)**2)/B3_ho_Anzahl 
print("B3_ho_Laenge", B3_ho_Laenge)
#[0.2675601  0.28327882 0.27727072 0.2763998 ]

B3_ve_Laenge = unp.sqrt((B3_ve_En_x - B3_ve_An_x)**2 + (B3_ve_En_y - B3_ve_An_y)**2)/B3_ve_Anzahl
print("B3_ve_Laenge", B3_ve_Laenge)
#[0.17794255 0.17457638 0.16866173 0.17226785]

B4_ho_Laenge = unp.sqrt((B4_ho_En_x - B4_ho_An_x)**2 + (B4_ho_En_y - B4_ho_An_y)**2)/B4_ho_Anzahl 
print("B4_ho_Laenge", B4_ho_Laenge)
#[0.2700828  0.2763998  0.27403334 0.27430524

B4_ve_Laenge = unp.sqrt((B4_ve_En_x - B4_ve_An_x)**2 + (B4_ve_En_y - B4_ve_An_y)**2)/B4_ve_Anzahl
print("B4_ve_Laenge", B4_ve_Laenge)
#[0.17293483 0.1676906  0.17055617 0.16748292]

# Winkel berechnen 

B1_ho_En_x_nom = unp.nominal_values(B1_ho_En_x)
B1_ho_An_x_nom = unp.nominal_values(B1_ho_An_x)
B1_ho_En_y_nom = unp.nominal_values(B1_ho_En_y)
B1_ho_An_y_nom = unp.nominal_values(B1_ho_An_y)

B1_ve_En_x_nom = unp.nominal_values(B1_ve_En_x)
B1_ve_An_x_nom = unp.nominal_values(B1_ve_An_x)
B1_ve_En_y_nom = unp.nominal_values(B1_ve_En_y)
B1_ve_An_y_nom = unp.nominal_values(B1_ve_An_y)

B1_ho_Laenge_nom = unp.nominal_values(B1_ho_Laenge)
B1_ve_Laenge_nom = unp.nominal_values(B1_ve_Laenge)

B2_ho_En_x_nom = unp.nominal_values(B2_ho_En_x)
B2_ho_An_x_nom = unp.nominal_values(B2_ho_An_x)
B2_ho_En_y_nom = unp.nominal_values(B2_ho_En_y)
B2_ho_An_y_nom = unp.nominal_values(B2_ho_An_y)

B2_ve_En_x_nom = unp.nominal_values(B2_ve_En_x)
B2_ve_An_x_nom = unp.nominal_values(B2_ve_An_x)
B2_ve_En_y_nom = unp.nominal_values(B2_ve_En_y)
B2_ve_An_y_nom = unp.nominal_values(B2_ve_An_y)

B2_ho_Laenge_nom = unp.nominal_values(B2_ho_Laenge)
B2_ve_Laenge_nom = unp.nominal_values(B2_ve_Laenge)

B3_ho_En_x_nom = unp.nominal_values(B3_ho_En_x)
B3_ho_An_x_nom = unp.nominal_values(B3_ho_An_x)
B3_ho_En_y_nom = unp.nominal_values(B3_ho_En_y)
B3_ho_An_y_nom = unp.nominal_values(B3_ho_An_y)

B3_ve_En_x_nom = unp.nominal_values(B3_ve_En_x)
B3_ve_An_x_nom = unp.nominal_values(B3_ve_An_x)
B3_ve_En_y_nom = unp.nominal_values(B3_ve_En_y)
B3_ve_An_y_nom = unp.nominal_values(B3_ve_An_y)

B3_ho_Laenge_nom = unp.nominal_values(B3_ho_Laenge)
B3_ve_Laenge_nom = unp.nominal_values(B3_ve_Laenge)

B4_ho_En_x_nom = unp.nominal_values(B4_ho_En_x)
B4_ho_An_x_nom = unp.nominal_values(B4_ho_An_x)
B4_ho_En_y_nom = unp.nominal_values(B4_ho_En_y)
B4_ho_An_y_nom = unp.nominal_values(B4_ho_An_y)

B4_ve_En_x_nom = unp.nominal_values(B4_ve_En_x)
B4_ve_An_x_nom = unp.nominal_values(B4_ve_An_x)
B4_ve_En_y_nom = unp.nominal_values(B4_ve_En_y)
B4_ve_An_y_nom = unp.nominal_values(B4_ve_An_y)

B4_ho_Laenge_nom = unp.nominal_values(B4_ho_Laenge)
B4_ve_Laenge_nom = unp.nominal_values(B4_ve_Laenge)

theta_B1 = 180/np.pi * np.arccos(((B1_ho_En_x_nom - B1_ho_An_x_nom) * (B1_ve_En_x_nom - B1_ve_An_x_nom) + (B1_ho_En_y_nom - B1_ho_An_y_nom) * (B1_ve_En_y_nom - B1_ve_An_y_nom))/(B1_ho_Laenge_nom * B1_ho_Anzahl * B1_ve_Laenge_nom * B1_ve_Anzahl))

theta_B2 = 180/np.pi * np.arccos(((B2_ho_En_x_nom - B2_ho_An_x_nom) * (B2_ve_En_x_nom - B2_ve_An_x_nom) + (B2_ho_En_y_nom - B2_ho_An_y_nom) * (B2_ve_En_y_nom - B2_ve_An_y_nom))/(B2_ho_Laenge_nom * B2_ho_Anzahl * B2_ve_Laenge_nom * B2_ve_Anzahl))

theta_B3 = 180/np.pi * np.arccos(((B3_ho_En_x_nom - B3_ho_An_x_nom) * (B3_ve_En_x_nom - B3_ve_An_x_nom) + (B3_ho_En_y_nom - B3_ho_An_y_nom) * (B3_ve_En_y_nom - B3_ve_An_y_nom))/(B3_ho_Laenge_nom * B3_ho_Anzahl * B3_ve_Laenge_nom * B3_ve_Anzahl))

theta_B4 = 180/np.pi * np.arccos(((B4_ho_En_x_nom - B4_ho_An_x_nom) * (B4_ve_En_x_nom - B4_ve_An_x_nom) + (B4_ho_En_y_nom - B4_ho_An_y_nom) * (B4_ve_En_y_nom - B4_ve_An_y_nom))/(B4_ho_Laenge_nom * B4_ho_Anzahl * B4_ve_Laenge_nom * B4_ve_Anzahl))

alle_theta = np.append(np.append(np.append(theta_B1,theta_B2), theta_B3), theta_B4)
print("Abweichung: ", sem(alle_theta))
#print("Mittelwert: ", np.mean(alle_theta))

#Abweichung:  0.38542814761842903
#Mittelwert:  49.369945684627474


#print(theta_B1 * 180/np.pi) 
#theta in Grad [47.8535453  50.25306149 49.84482588 49.02055604]
#print(theta_B2 * 180/np.pi)
#[50.50551077 48.29734637 50.23188131 48.21295391]

#print(theta_B3 * 180/np.pi)
#[44.84294972 48.83118931 50.16613256 50.72514639]

#print(theta_B4 * 180/np.pi)
#[50.69344561 50.2900083  49.4206676  50.72991039]






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
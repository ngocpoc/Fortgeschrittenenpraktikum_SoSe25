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

Hoehen_anderes_Goldbild = np.genfromtxt("Messdaten/Hoehenprofil_anderes/Hoehenprofil_anderes_Daten.txt", unpack=True)
Hoehen_unseres_Goldbild = np.genfromtxt("Messdaten/Hoehenprofile_unseres/Hoehenprofil_unseres_Daten.txt", unpack=True)


N = 2
ind = np.arange(N)
fig, (ax1) = plt.subplots(1, 1, layout="constrained")
plt.hlines(Hoehen_unseres_Goldbild, xmin=1, xmax=4)
plt.hlines(Hoehen_anderes_Goldbild, xmin=6, xmax=9)

plt.ylabel("$\Delta \, z$ / nm")
plt.xticks([2.5, 7.5], ['Eigener Scan', 'Nicht eigener Scan'])
plt.xlim(0, 10)
fig.savefig("build/plot.pdf")




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
#print("B1_ho_Laenge: ", B1_ho_Laenge)
print("Durchschnitt B1_ho", sum(B1_ho_Laenge/4))


B1_ve_Laenge = unp.sqrt((B1_ve_En_x - B1_ve_An_x)**2 + (B1_ve_En_y - B1_ve_An_y)**2)/B1_ve_Anzahl 
print("B1_ve_Laenge: ", B1_ve_Laenge)
# [0.17506399 0.16962984 0.16928342 0.17334925]
#[0.17506398976572912+/-0.005104427076690391
 #0.16962984014427404+/-0.0028357928203835503
 #0.16928342117063672+/-0.0031902669229314936
 #0.1733492528806331+/-0.006380533845862988]

#g1 ist horizontal g2 diagonal
B2_ho_Laenge = unp.sqrt((B2_ho_En_x - B2_ho_An_x)**2 + (B2_ho_En_y - B2_ho_An_y)**2)/B2_ho_Anzahl 
print("B2_ho_Laenge ", B2_ho_Laenge)
#[0.27762289 0.27236901 0.2732685  0.27803969]
B2_ve_Laenge = unp.sqrt((B2_ve_En_x - B2_ve_An_x)**2 + (B2_ve_En_y - B2_ve_An_y)**2)/B2_ve_Anzahl 
print("B2_ve_Laenge: ", B2_ve_Laenge)
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

Durchschnitt_B1_ho = sum(B1_ho_Laenge/4)
Durchschnitt_B2_ho = sum(B2_ho_Laenge/4)
Durchschnitt_B3_ho = sum(B3_ho_Laenge/4)
Durchschnitt_B4_ho = sum(B4_ho_Laenge/4)

Durchschnitt_B1_ve = sum(B1_ve_Laenge/4)
Durchschnitt_B2_ve = sum(B2_ve_Laenge/4)
Durchschnitt_B3_ve = sum(B3_ve_Laenge/4)
Durchschnitt_B4_ve = sum(B4_ve_Laenge/4)

Horizontale_Laengen_Durchschnitt = (Durchschnitt_B1_ho + Durchschnitt_B2_ho + Durchschnitt_B3_ho + Durchschnitt_B4_ho)/4
Vertikale_Laengen_Durchschnitt = (Durchschnitt_B1_ve + Durchschnitt_B2_ve + Durchschnitt_B3_ve + Durchschnitt_B4_ve)/4

print("Horizontale_Laengen_Durchschnitt: ", Horizontale_Laengen_Durchschnitt)
print("Vertikale_Laengen_Durchschnitt: ", Vertikale_Laengen_Durchschnitt)

Allgemeiner_Durchschnitt = (Horizontale_Laengen_Durchschnitt + Vertikale_Laengen_Durchschnitt)/2
print("Allgemeiner_Durchschnitt: ", Allgemeiner_Durchschnitt)

#Horizontale_Laengen_Durchschnitt:  0.2752+/-0.0016
#Vertikale_Laengen_Durchschnitt:  0.1716+/-0.0010


#[0.17293483 0.1676906  0.17055617 0.16748292]

#g1 ist horizontal g2 diagonal
#B1_ho_Laenge:  [0.2761378285789806+/-0.012761067691725974
# 0.27803968857055306+/-0.004253689230575325
# 0.2740845932096956+/-0.006380533845862987
# 0.27403334377140154+/-0.005104427076690391]
#Durchschnitt B1_ho 0.276+/-0.004
#B1_ve_Laenge:  [
# 0.17506398976572912+/-0.005104427076690391
# 0.16962984014427404+/-0.0028357928203835503
# 0.16928342117063672+/-0.0031902669229314936
# 0.1733492528806331+/-0.006380533845862988]
#B2_ho_Laenge  [
# 0.27762288604852114+/-0.005104427076690391
# 0.2723690061080393+/-0.00850737846115065
# 0.27326849757736155+/-0.004253689230575325
# 0.27803968857055306+/-0.004253689230575325]
#B2_ve_Laenge:  [
# 0.1731841916030195+/-0.005104427076690391
# 0.17150466694612654+/-0.006380533845862987
# 0.17076382893206227+/-0.00283579282038355
# 0.17145703846352509+/-0.00283579282038355]
#B3_ho_Laenge [
# 0.26756009576577633+/-0.00850737846115065
# 0.2832788168312946+/-0.006380533845862987
# 0.2772707215692828+/-0.005104427076690391
# 0.276399803453036+/-0.004253689230575325]
#B3_ve_Laenge [
# 0.1779425535619967+/-0.005104427076690389
# 0.17457637848881366+/-0.0036460193404931353
# 0.16866172898581516+/-0.001963241183342458
# 0.172267849378+/-0.004253689230575325]
#B4_ho_Laenge [
# 0.2700828048887638+/-0.005104427076690389
# 0.27639980345303605+/-0.004253689230575326
# 0.27403334377140154+/-0.005104427076690391
# 0.2743052442159912+/-0.00850737846115065]
#B4_ve_Laenge [
# 0.172934832961483+/-0.002552213538345195
# 0.16769060477938058+/-0.004253689230575324
# 0.1705561706553967+/-0.003646019340493136
# 0.16748291788883857+/-0.0028357928203835495]


#1. Tabelle: 
#        0.2761378285789806 \pm 0.012761067691725974   & 0.17506398976572912 \pm 0.005104427076690391   & 0.27762288604852114 \pm 0.005104427076690391  & 0.1731841916030195   \pm 0.005104427076690391  \\
#        0.27803968857055306 \pm 0.004253689230575325  & 0.16962984014427404 \pm 0.0028357928203835503  & 0.2723690061080393  \pm 0.00850737846115065    & 0.17150466694612654 \pm 0.006380533845862987 \\
#        0.2740845932096956 \pm 0.006380533845862987   & 0.16928342117063672 \pm 0.0031902669229314936  & 0.27326849757736155 \pm 0.004253689230575325  & 0.17076382893206227  \pm 0.00283579282038355  \\
#        0.27403334377140154 \pm 0.005104427076690391  & 0.1733492528806331 \pm 0.006380533845862988    & 0.27803968857055306 \pm 0.004253689230575325  & 0.17145703846352509  \pm 0.00283579282038355  \\

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

#Eigener Scan:
Oberere_Haufungspunkt_eigen = np.array([0.841, 0.761, 0.879, 0.781, 0.909, 0.857])
Oberere_Haufungspunkt_eigen_Mittelwert = np.mean(Oberere_Haufungspunkt_eigen)
Oberere_Haufungspunkt_eigen_Abweichung = sem(Oberere_Haufungspunkt_eigen)

Unterer_Haufungspunkt_eigen = np.array([0.631, 0.615, 0.581, 0.591, 0.529])
Unterer_Haufungspunkt_eigen_Mittelwert = np.mean(Unterer_Haufungspunkt_eigen)
Unterer_Haufungspunkt_eigen_Abweichung = sem(Unterer_Haufungspunkt_eigen)

Unterschied_eigener_Haufungspunkt = unp.uarray(Oberere_Haufungspunkt_eigen_Mittelwert, Oberere_Haufungspunkt_eigen_Abweichung) - unp.uarray(Unterer_Haufungspunkt_eigen_Mittelwert, Unterer_Haufungspunkt_eigen_Abweichung)

print("Oberere_Haufungspunkt_eigen_Mittelwert: ", Oberere_Haufungspunkt_eigen_Mittelwert, Oberere_Haufungspunkt_eigen_Abweichung)
print("Unterer_Haufungspunkt_eigen_Mittelwert: ", Unterer_Haufungspunkt_eigen_Mittelwert, Unterer_Haufungspunkt_eigen_Abweichung)
print("Unterschied_eigener_Haufungspunkt: ", Unterschied_eigener_Haufungspunkt)

#Anderer Scan: 
Oberere_Haufungspunkt_anderes = np.array([0.609, 0.546, 0.472, 0.451, 0.392, 0.365, 0.570, 0.479, 0.572, 0.512, 0.407, 0.378])
Oberere_Haufungspunkt_anderes_Mittelwert = np.mean(Oberere_Haufungspunkt_anderes)
Oberere_Haufungspunkt_anderes_Abweichung = sem(Oberere_Haufungspunkt_anderes)

Unterer_Haufungspunkt_anderes = np.array([0.230, 0.211, 0.156, 0.130, 0.255, 0.250, 0.241])
Unterer_Haufungspunkt_anderes_Mittelwert = np.mean(Unterer_Haufungspunkt_anderes)
Unterer_Haufungspunkt_anderes_Abweichung = sem(Unterer_Haufungspunkt_anderes)

Unterschied_anderer_Haufungspunkt = unp.uarray(Oberere_Haufungspunkt_anderes_Mittelwert, Oberere_Haufungspunkt_anderes_Abweichung) - unp.uarray(Unterer_Haufungspunkt_anderes_Mittelwert, Unterer_Haufungspunkt_anderes_Abweichung)

print("Oberere_Haufungspunkt_anderes_Mittelwert: ", Oberere_Haufungspunkt_anderes_Mittelwert, Oberere_Haufungspunkt_anderes_Abweichung)
print("Unterer_Haufungspunkt_anderes_Mittelwert: ", Unterer_Haufungspunkt_anderes_Mittelwert, Unterer_Haufungspunkt_anderes_Abweichung)
print("Unterschied_anderer_Haufungspunkt: ", Unterschied_anderer_Haufungspunkt)

print("Unterschied beider Haufungspunkte_Durchschnitt: ", (Unterschied_eigener_Haufungspunkt + Unterschied_anderer_Haufungspunkt)/2)

#Oberere_Haufungspunkt_eigen_Mittelwert:  0.8380000000000001 0.023288051299611427
#Unterer_Haufungspunkt_eigen_Mittelwert:  0.5894 0.01747455292704222
#Unterschied_eigener_Haufungspunkt:  0.249+/-0.029
#Oberere_Haufungspunkt_anderes_Mittelwert:  0.47941666666666666 0.024024753123107482
#Unterer_Haufungspunkt_anderes_Mittelwert:  0.2104285714285714 0.018453102904463548
#Unterschied_anderer_Haufungspunkt:  0.269+/-0.030

#Unterschied beider Haufungspunkte_Durchschnitt:  0.259+/-0.021

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
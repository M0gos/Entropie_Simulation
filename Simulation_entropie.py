#%%
import numpy as np
from random import gauss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

### ATENTION ### certain argument "bin" dans les fonction ci-dessous doivent être cosidéré
#                comme le nombre de segment dans l'histogramme et d'autre comme le nombre de segment en puissance de 2 (2**{bin}).


# fonction gaussienne
def Gauss(x,mu,sig,A):
    return A*np.exp(-(x-mu)**2/(2*sig**2))

##définition du calcul d'entropie à l'aide d'un array l'entropie
def H(arr_don, minmax_tuple,bin):
    ### arr_don : liste de donnée formant la gaussienne
    ### minmax_tuple : borne de l'histogramme
    ### bin : nombre de séparation de l'histogramme
    long = 2**(bin)
    l = len(arr_don)
    test = np.zeros(2)
    if type(arr_don) != type(test):
         print("la liste de donnée n'est pas un array")
         return
    binn, edge = np.histogram(arr_don, bins = long, range = minmax_tuple)
    prob = binn[np.nonzero(binn)]/l ## on veut enlever les 0 de la liste

    ent = -sum(prob*np.log2(prob))
    return ent #print("l'entropie de la gaussienne est "+str(ent)+" bits" )

def Hdim(array, nb_dim, bin,minmax_tuple,sig):
    ### nb_dim : nombre de dimension de l'histogramme
    ###  Lorsque le nomobre de dimension est plus grand la configuration de l'histogramme est comme ceci [(x1,x2),(x3,x4),....]
    long = len(array)
    don = np.reshape(array,(int(long/nb_dim),nb_dim)) #changement de dimension des donnée pour les mettre dans un matrice de dimension (n x m) et pouvoir utiliser np.histogramdd
    bine = [bin]*nb_dim
    minmax = [minmax_tuple]*nb_dim
    hist, edge = np.histogramdd(don,bins = bine, range = minmax)  #initialisation de l'histogramme
    prob = hist[np.nonzero(hist)]/int(long/nb_dim)      # transformation des compte de l'histogramme en probabilité
    ent = -sum(prob*np.log2(prob))   #calcul d'entropie
    entdiff = ent - np.log2(1/((minmax_tuple[1]-minmax_tuple[0])/bin))
    #return print("l'entropie de la distribution en groupe de " + str(nb_dim)+ " est " + str(round(ent,4)))
    return ent, ("Hdiff exp. = " +str(round(entdiff,3)),"Hdiff théo. = "+str(round(np.log2(np.sqrt(2*np.pi*np.e)*sig),3)))
    # renvoie l'entropie expérimental et ensuite compare l'entropie différentiel d ela théorie et de l'Exp.

def param_fit_gauss(x,y,p_0):
    ## fait un fit gaussien et retourne les paramètres du fit obtenue
    [mu,sig,amp], liste = curve_fit(Gauss,x,y,p_0)
    return mu, sig, amp

def gauss_init(nb_ech,mu,sigma):
    ## initialisation de la gaussienne dans un array
    # mu = moyenne de la gauss.
    #sigma = variance de la gauss.
    gauss_int = np.zeros(nb_ech)
    for  i  in range(len(gauss_int)):
        gauss_int[i] = gauss(mu, sigma)
    return gauss_int, (-3*sigma,3*sigma)

def rand_init(nb_ech, range):
    # génère un array de nombre aléatoire nb_ech entre (range[0],range[1])
    long = range[1]-range[0]
    return (np.random.random(nb_ech)-0.5)*long

def main_hist_fit(arr, bin):
    # fonction qui effectue l'histogramme des données et qui renvoie les données avec les paramètre du fit
    test = np.zeros(2)
    if type(arr) != type(test):
        print("la liste de donnée n'Est pas un array")
        return
    long = 2**(bin)
    binn, nb = np.histogram(arr, bins = long)
    paramfit = param_fit_gauss(nb[0:len(nb)-1], binn,[mu_theo,sig_theo,amp])
    xtheo = nb[0:len(nb)-1]
    ytheo = Gauss(xtheo,*paramfit)
    return xtheo, binn, ytheo, paramfit
    #retourne valeur de x,y, les valeur de l'histograme et les paramêtre du fit donc un fois la fonction appelé il ne suffit que d'Afficher les truc

# définition d'un filtre particulier (passe-bas)
def filtre_plus(arr, rang_tup, bin):
    x = np.zeros(len(arr))
    for i in range(len(arr)-1):
        x[i] = (arr[i]+arr[i+1])/2
    long = 2**bin
    sep, nb = np.histogram(x,bins = long, range = rang_tup) ## même structure que la fonction Hdim
    prob = sep[np.nonzero(sep)]/len(arr)
    ent = -sum(prob*np.log2(prob))
    return x, sep, nb[0:len(nb)-1], ent
    #retourne les donnée filtré, les valeur de l'histogramme et les position sur l'Abscisse et l'Entropie

# même chose que la fonction précédente mais avec un filtre passe-haut
def filtre_moins(arr, rang_tup, bin):
    x = np.zeros(len(arr))
    for i in range(len(arr)-1):
        x[i] = (arr[i]-arr[i+1])/2
    long = 2**bin
    sep, nb = np.histogram(x,bins = long, range = rang_tup) ## même structure que la fonction Hdim
    prob = sep[np.nonzero(sep)]/len(arr)
    ent = -sum(prob*np.log2(prob))
    return x, sep, nb[0:len(nb)-1], ent

def Hdim_gliss(array, nb_dim,bin,minmax_tuple):
    long = len(array)
    # tentive 2 de calcul en considérant les points consécutif ([(x1,x2),(x2,x3),...])
    if nb_dim == 2:
        don1 = np.reshape(array,(int(long/nb_dim),nb_dim))
        don2 = np.reshape(array[1:-1],(int((long - nb_dim)/nb_dim),nb_dim)) #reshape des donnée pour les mettre dans un matrice de dimension (n x m) et pouvoir utiliser np.histogramdd
        don = np.concatenate((don1,don2))
    elif nb_dim == 3:
        don1 = np.reshape(array,(int(long/nb_dim),nb_dim))
        don2 = np.reshape(array[1:-2],(int((long - nb_dim)/nb_dim),nb_dim)) #reshape des donnée pour les mettre dans un matrice de dimension (n x m) et pouvoir utiliser np.histogramdd
        don3 = np.reshape(array[2:-1],(int((long - nb_dim)/nb_dim),nb_dim))
        don = np.concatenate((don1,don2,don3))
    elif nb_dim == 4:
        don1 = np.reshape(array,(int(long/nb_dim),nb_dim))
        don2 = np.reshape(array[1:-3],(int((long - nb_dim)/nb_dim),nb_dim)) #reshape des donnée pour les mettre dans un matrice de dimension (n x m) et pouvoir utiliser np.histogramdd
        don3 = np.reshape(array[3:-1],(int((long - nb_dim)/nb_dim),nb_dim))
        don4 = np.reshape(array[2:-2],(int((long - nb_dim)/nb_dim),nb_dim))
        don = np.concatenate((don1,don2,don3,don4))
    elif nb_dim == 5:
        don1 = np.reshape(array,(int(long/nb_dim),nb_dim))
        don2 = np.reshape(array[1:-4],(int((long - nb_dim)/nb_dim),nb_dim)) #reshape des donnée pour les mettre dans un matrice de dimension (n x m) et pouvoir utiliser np.histogramdd
        don3 = np.reshape(array[4:-1],(int((long - nb_dim)/nb_dim),nb_dim))
        don4 = np.reshape(array[2:-3],(int((long - nb_dim)/nb_dim),nb_dim))
        don5 = np.reshape(array[3:-2],(int((long - nb_dim)/nb_dim),nb_dim))
        don = np.concatenate((don1,don2,don3,don4,don5))
    elif nb_dim == 6:
        don1 = np.reshape(array,(int(long/nb_dim),nb_dim))
        don2 = np.reshape(array[1:-5],(int((long - nb_dim)/nb_dim),nb_dim)) #reshape des donnée pour les mettre dans un matrice de dimension (n x m) et pouvoir utiliser np.histogramdd
        don3 = np.reshape(array[5:-1],(int((long - nb_dim)/nb_dim),nb_dim))
        don4 = np.reshape(array[2:-4],(int((long - nb_dim)/nb_dim),nb_dim))
        don5 = np.reshape(array[4:-2],(int((long - nb_dim)/nb_dim),nb_dim))
        don6 = np.reshape(array[3:-3],(int((long - nb_dim)/nb_dim),nb_dim))
        don = np.concatenate((don1,don2,don3,don4,don5,don6))
    # on définit la liste de point tout dépendant du nombre de dimension de l'histogramme
    bine = [bin]*nb_dim
    minmax = [minmax_tuple]*nb_dim
    hist, edge = np.histogramdd(don,bins = bine, range = minmax)  #initialisation de l'histograme
    prob = hist[np.nonzero(hist)]/int(sum(hist[np.nonzero(hist)]))      # transformation des compte de l'histograme en probabilité
    #if len(hist.shape) == 1:
     #   raise Exception("please use the H function for 1-D array")
    ent = -sum(prob*np.log2(prob))   #calcul d'entropie
    #return print("l'entropie de la distribution en groupe de " + str(nb_dim)+ " est " + str(round(ent,4)))
    return ent


# %%
#définition des constant et des données initiales de simulation
mu_theo = 0
sig_theo = 1
nombre_pts = 2*3*4*5*6*5000
amp = 1/np.sqrt(2*np.pi*sig_theo)
bine = 8
minmax = (-3*sig_theo,3*sig_theo)
x = gauss_init(nombre_pts,mu_theo,sig_theo) #initialisation des données théorique
x2 = gauss_init(nombre_pts,mu_theo,sig_theo)
xrand = rand_init(nombre_pts,minmax)
#xrand2 = rand_init(nombre_pts**2,minmax)

#---------------------------------------------------------------------------------------------------------
#affichage de l'histogramme et de son fit en 1-D
# %%
hist = main_hist_fit(x[0],bine)
x_filt = filtre_plus(*x,bine)
#%%
plt.figure()
plt.plot(hist[0],hist[1], label = "gauss" )
plt.plot(hist[0],hist[2],"k--", label = "fit gauss")
plt.plot(x_filt[2],x_filt[1], label = "gauss filtre")
plt.legend()
plt.show()
#-------------------------------------------------------------------------------------------------------

#tentative 1 de prog en considérant les point en couple()

# %%
# définition et création des donnée théorique et des paramêtre important
nb_pts = 6
donnée = x2[0]
donnéera = xrand #pour donnée_random
xx = np.arange(nb_pts) + 1
print(xx)
donnée_filtre = filtre_plus(donnée, minmax,5)
donnéera_filtre = filtre_plus(donnéera,minmax,5)
donnée_fil_moins = filtre_moins(donnée,minmax,5)
# %%
#-------------------------------------------------------------------
# calcul des entropies avec et sans filtre
enty = np.zeros(nb_pts)
enty_filtre = np.zeros(nb_pts)
enty_rand = np.zeros(nb_pts)
enty_rand_filtre = np.zeros(nb_pts)
ent_gliss = np.zeros(nb_pts)
ent_gliss_filtre = np.zeros(nb_pts)
for i in xx:
    if i == 1:
        enty[i-1] = H(donnée,minmax,5)
        enty_filtre[i-1] = donnée_filtre[3]
        enty_rand[i-1] = H(donnéera,minmax,5)
        enty_rand_filtre[i-1] = donnéera_filtre[3]
        ent_gliss[i-1] = H(donnée,minmax,5)
        ent_gliss_filtre[i-1] = donnée_filtre[3]
    else:
        enty[i-1] = Hdim(donnée,i,32,minmax)[0]
        enty_filtre[i-1] = Hdim(donnée_filtre[0],i,32,minmax)[0]
        enty_rand[i-1] = Hdim(donnéera,i,32,minmax)[0]
        enty_rand_filtre[i-1] = Hdim(donnéera_filtre[0],i,32,minmax)[0]
        ent_gliss[i-1] = Hdim_gliss(donnée,i,32,minmax)
        ent_gliss_filtre[i-1] = Hdim_gliss(donnéera_filtre[0],i,32,minmax)
    print(i)
# %%
print("H_exp = "+ str(round(Hdim(donnée,1,32,minmax,sig_theo)[0],3)),Hdim(donnée,1,32,minmax,sig_theo)[1])
# %%
enty_rand_filtre[0] = donnéera_filtre[3]
fig = plt.figure()
fig.set_facecolor("white")
plt.xlabel("Dimension de l'histogramme")
plt.ylabel("Entropie")
plt.plot(xx,enty_rand,".-", label = "Rand. sans filtre")
plt.plot(xx,enty_rand_filtre,".-", label = "Rand. avec filtre")
plt.plot(xx,ent_gliss,".-", label = "fenêtre glis. no-filtre")
plt.plot(xx,ent_gliss_filtre,".-", label = "fenêtre glis. filtre")
plt.title(str(nombre_pts) + " points")
plt.legend()
plt.savefig("entropie_et_filtre_gliss_rand_points__2021-10-20.png")
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import math
import random

class LabeledSet:

    def __init__(self,input_dimension):
        self.input_dimension=input_dimension
        self.nb_examples=0
        self.labels = dict()
        self.label_count = []

    def addExample(self,vector,label):
        if label not in self.labels:
            self.labels[label] = len(self.labels)
            self.label_count.append(0)
        if (self.nb_examples==0):
            self.x=np.array([vector])
            self.y=np.array([[label]])
        else:
            self.x=np.vstack((self.x,vector))
            self.y=np.vstack((self.y,label))

        self.nb_examples=self.nb_examples+1
        self.label_count[self.labels[label]] += 1

    #Renvoie la dimension de l'espace d'entrée
    def getInputDimension(self):
       return self.input_dimension
   

    #Renvoie le nombre d'exemple dans le set
    def size(self):
        return self.nb_examples

    #Renvoie la valeur de x_i
    def getX(self,i):
        return self.x[i]


    #Renvoie la valeur de y_i
    def getY(self,i):
        return(self.y[i])

    def getDistribution(self):
        suma = reduce((lambda x, y: x + y),self.label_count)
        return list(map((lambda x: float(x)/suma), self.label_count))

    def getMaxLabel(self):
        maxim = -1
        label = None

        for k, v in self.labels.items():
            if self.label_count[v] > maxim:
                label = k
                maxim = self.label_count[v]

        return label

def plot2DSet(set):
    """ LabeledSet -> NoneType
        Hypothèse: set est de dimension 2
        affiche une représentation graphique du LabeledSet
        remarque: l'ordre des labels dans set peut être quelconque
    """
    S_pos = set.x[np.where(set.y == 1),:][0]      # tous les exemples de label +1
    S_neg = set.x[np.where(set.y == -1),:][0]     # tous les exemples de label -1
    plt.scatter(S_pos[:,0],S_pos[:,1],marker='o')
    plt.scatter(S_neg[:,0],S_neg[:,1],marker='x')
    
class Classifier:
    def __init__(self,input_dimension):
        """ Constructeur """
        raise NotImplementedError("Please Implement this method")


    # Permet de calculer la prediction sur x => renvoie un score
    def predict(self,x):
        raise NotImplementedError("Please Implement this method")


    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,labeledSet):
        raise NotImplementedError("Please Implement this method")

    # Permet de calculer la qualité du système
    def accuracy(self,set):
        nb_ok=0
        for i in range(set.size()):
            score=self.predict(set.getX(i))
            if (score*set.getY(i)>0):
                nb_ok=nb_ok+1
        acc=nb_ok/(set.size() * 1.0)
        return acc

def plot_frontiere(set,classifier,step=20):
    """ LabeledSet * Classifier * int -> NoneType
        Remarque: le 3e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax=set.x.max(0)
    mmin=set.x.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))

    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    plt.contourf(x1grid,x2grid,res,colors=["red","cyan"],levels=[-1000,0,1000],linewidth=2)

def shannon(distr):
    k = len(distr)
    if k > 1:
        f = lambda x: 0 if x == 0 else x*math.log(x,k)
        logarithms = list((map (f, distr)))
        return - reduce ((lambda x, y: x+y), logarithms)
    else:
        return 0

def entropie(aSet):
    distr = aSet.getDistribution()
    return shannon(distr)

def discretise(LSet, col):
    """ LabelledSet * int -> tuple[float, float]
        col est le numéro de colonne sur X à discrétiser
        rend la valeur de coupure qui minimise l'entropie ainsi que son entropie.
    """
    # initialisation:
    min_entropie = 1.1  # on met à une valeur max car on veut minimiser
    min_seuil = 0.0
    # trie des valeurs:
    ind= np.argsort(LSet.x,axis=0)

    # calcul des distributions des classes pour E1 et E2:
    inf_plus  = 0               # nombre de +1 dans E1
    inf_moins = 0               # nombre de -1 dans E1
    sup_plus  = 0               # nombre de +1 dans E2
    sup_moins = 0               # nombre de -1 dans E2
    # remarque: au départ on considère que E1 est vide et donc E2 correspond à E.
    # Ainsi inf_plus et inf_moins valent 0. Il reste à calculer sup_plus et sup_moins
    # dans E.
    for j in range(0,LSet.size()):
        if (LSet.getY(j) == -1):
            sup_moins += 1
        else:
            sup_plus += 1
    nb_total = (sup_plus + sup_moins) # nombre d'exemples total dans E

    # parcours pour trouver le meilleur seuil:
    for i in range(len(LSet.x)-1):
        v_ind_i = ind[i]   # vecteur d'indices
        courant = LSet.getX(v_ind_i[col])[col]
        lookahead = LSet.getX(ind[i+1][col])[col]
        val_seuil = (courant + lookahead) / 2.0;
        # M-A-J de la distrib. des classes:
        # pour réduire les traitements: on retire un exemple de E2 et on le place
        # dans E1, c'est ainsi que l'on déplace donc le seuil de coupure.
        if LSet.getY(ind[i][col])[0] == -1:
            inf_moins += 1
            sup_moins -= 1
        else:
            inf_plus += 1
            sup_plus -= 1
        # calcul de la distribution des classes de chaque côté du seuil:
        nb_inf = (inf_moins + inf_plus)*1.0     # rem: on en fait un float pour éviter
        nb_sup = (sup_moins + sup_plus)*1.0     # que ce soit une division entière.
        # calcul de l'entropie de la coupure
        val_entropie_inf = shannon([inf_moins / nb_inf, inf_plus  / nb_inf])
        val_entropie_sup = shannon([sup_moins / nb_sup, sup_plus  / nb_sup])
        val_entropie = (nb_inf / nb_total) * val_entropie_inf + (nb_sup / nb_total) * val_entropie_sup
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (min_entropie > val_entropie):
            min_entropie = val_entropie
            min_seuil = val_seuil
    return (min_seuil, min_entropie)

def divise(LSet, att, seuil):
    plus_petits = LabeledSet(LSet.getInputDimension())
    plus_grands = LabeledSet(LSet.getInputDimension())

    for i in range (LSet.size()):
        if LSet.getX(i)[att] <= seuil:
            plus_petits.addExample(LSet.getX(i), LSet.getY(i)[0])
        else:
            plus_grands.addExample(LSet.getX(i), LSet.getY(i)[0])
    return plus_petits, plus_grands

class ArbreBinaire:
    def __init__(self):
        self.attribut = None   # numéro de l'attribut
        self.seuil = None
        self.inferieur = None # ArbreBinaire Gauche (valeurs <= au seuil)
        self.superieur = None # ArbreBinaire Gauche (valeurs > au seuil)
        self.classe = None # Classe si c'est une feuille: -1 ou +1

    def est_feuille(self):
        """ rend True si l'arbre est une feuille """
        return self.seuil == None

    def ajoute_fils(self,ABinf,ABsup,att,seuil):
        """ ABinf, ABsup: 2 arbres binaires
            att: numéro d'attribut
            seuil: valeur de seuil
        """
        self.attribut = att
        self.seuil = seuil
        self.inferieur = ABinf
        self.superieur = ABsup

    def ajoute_feuille(self,classe):
        """ classe: -1 ou + 1
        """
        self.classe = classe

    def classifie(self,exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple: +1 ou -1
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] <= self.seuil:
            return self.inferieur.classifie(exemple)
        return self.superieur.classifie(exemple)

    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir
            l'afficher
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.attribut))
            self.inferieur.to_graph(g,prefixe+"g")
            self.superieur.to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))

        return g

def construit_AD(LSet, epsilon):
    un_arbre = ArbreBinaire()

    if shannon(LSet.getDistribution()) <= epsilon:
        un_arbre.ajoute_feuille(LSet.getMaxLabel())

    else:
        dim = LSet.getInputDimension()
        minim = float('inf')
        min_seuil = min_index = 0
        for i in range(dim):
            seuil, entropie = discretise(LSet,i)
            if entropie < minim:
                min_seuil, minim, min_index = (seuil, entropie, i)
        smaller, bigger = divise(LSet,min_index,min_seuil)
        un_arbre.ajoute_fils(construit_AD(smaller,epsilon),construit_AD(bigger,epsilon),min_index, min_seuil)

    return un_arbre

class ArbreDecision(Classifier):
    # Constructeur
    def __init__(self,epsilon):
        # valeur seuil d'entropie pour arrêter la construction
        self.epsilon= epsilon
        self.racine = None

    # Permet de calculer la prediction sur x => renvoie un score
    def predict(self,x):
        # classification de l'exemple x avec l'arbre de décision
        # on rend 0 (classe -1) ou 1 (classe 1)
        classe = self.racine.classifie(x)
        if (classe == 1):
            return(1)
        else:
            return(-1)

    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,set):
        # construction de l'arbre de décision
        self.set=set
        self.racine = construit_AD(set,self.epsilon)

    # Permet d'afficher l'arbre
    def plot(self):
        gtree = gv.Digraph(format='png')
        return self.racine.to_graph(gtree)

def createXOR(nb_points,covar):
    a_set = LabeledSet(2)

    var = [[covar,0], [0,covar]]

    positive_center1 = [0,0]
    positive_center2 = [1,1]

    X = np.random.multivariate_normal(positive_center1, var, int(nb_points/4))
    Y = np.random.multivariate_normal(positive_center2, var, int(nb_points/4))

    for i in range(len(X)):
        a_set.addExample(X[i],1)
        a_set.addExample(Y[i],1)

    negative_center1 = [1,0]
    negative_center2 = [0,1]

    X = np.random.multivariate_normal(negative_center1, var, int(nb_points/4))
    Y = np.random.multivariate_normal(negative_center2, var, int(nb_points/4))

    for i in range(len(X)):
        a_set.addExample(X[i],-1)
        a_set.addExample(Y[i],-1)

    return a_set
       
class Perceptron(Classifier):
    def dist_euclidienne_vect(self,x,y):
        v = [(i-j)*(i-j) for i,j in np.column_stack([x,y])]
        return np.sqrt(np.sum(v))

    def __init__(self,input_dimension,learning_rate, eps=0.001, verb = False):
        self.input_dimension = input_dimension
        self.m = learning_rate
        self.weights = np.zeros(input_dimension)
        self.eps = eps
        self.verbose = verb

    #Permet de calculer la prediction sur x => renvoie un score
    def predict(self,x):
        value = np.vdot(self.weights,x)
        if value > 0:
            return 1
        else :
            return -1

    #Permet d'entrainer le modele sur un ensemble de données
    def train(self,labeledSet):
        niter = 0

        taille = labeledSet.size()
        old_weights = [float('inf') for i in range(self.input_dimension)]
        while(self.dist_euclidienne_vect(old_weights, self.weights)>self.eps and niter < 500):
            old_weights = self.weights
            for i in range(taille):
                vector = labeledSet.getX(i)
                label = labeledSet.getY(i)
                prediction =  self.predict(vector)
                if (prediction != label):
                    self.weights = [self.weights[i] + self.m*label*vector[i]  for i in range(self.input_dimension)]
            niter += 1
            if (self.verbose):
                print ("L'accuracy est %f apres %i iterations" %(self.accuracy(labeledSet),niter))


def tirage(v, m, remise):
    if remise:
        ret = [random.choice(v) for _ in range(m)]
    else:
        ret = random.sample(v,m)
    return ret

def echantillonLS(LSet, m, remise, plus_info = False):
    index = tirage(range(LSet.size()),m,remise)

    choix = LabeledSet(LSet.getInputDimension())
    pas_choisis = LabeledSet(LSet.getInputDimension())
    for i in index:
        choix.addExample(LSet.x[i], LSet.y[i][0])
    for i in range(LSet.size()):
        if i not in index:
            pas_choisis.addExample(LSet.x[i], LSet.y[i][0])
    if not plus_info:
        return choix
    else:
        return choix, pas_choisis

class ClassifierBaggingTree(Classifier):

    def __init__(self,nArbres,pourcentageExemples,seuil,remise):
        self.nArbres = nArbres
        self.pourcentage = pourcentageExemples
        self.seuil = seuil
        self.remise = remise
        self.foret = []

    def train(self,LSet):
        N = int(LSet.size() * self.pourcentage)
        for _ in range(self.nArbres):
            echantillon = echantillonLS(LSet,N,self.remise)
            arb_dec = ArbreDecision(self.seuil)
            arb_dec.train(echantillon)
            self.foret.append(arb_dec)
    def predict(self,x):
        votes = np.array([arbre.predict(x) for arbre in self.foret])
        if votes.mean() >= 0 :
            return 1
        else:
            return -1
            
class ClassifierOOBPerceptron(Classifier):

    def __init__(self,nPercep,pourcentageExemples,seuil,remise):
        self.nPercep = nPercep
        self.pourcentage = pourcentageExemples
        self.seuil = seuil
        self.remise = remise
        self.ensemble = dict()
        self.echantillons = dict()
        self.size = 0

    def train(self,LSet):
        N = int(LSet.size() * self.pourcentage)
        for _ in range(self.nPercep):
            echantillon = echantillonLS(LSet,N,self.remise)
            perc = Perceptron(LSet.getInputDimension(),0.05)
            perc.train(echantillon)
            self.ensemble[self.size] = perc
            self.echantillons[self.size] = echantillon
            self.size += 1

    def can_vote(self, k, position):
        echantillon = self.echantillons[k]
        for x in echantillon.x:
            foundInEchantillon = True
            for i in range(len(x)):
                if x[i] != position[i]:
                    foundInEchantillon = False
            if foundInEchantillon:
                return False
        return True

    def predict(self,x):
        right_to_vote = [self.can_vote(key, x) for key in self.echantillons]
        votes = np.array([self.ensemble[i].predict(x) for i in range(self.size) if right_to_vote[i] ])
        try :
            if votes.mean() >= 0 :
                return 1
            else:
                return -1
        except RuntimeWarning:
            return 1

class ClassifierOOBTree(Classifier):

    def __init__(self,nArbres,pourcentageExemples,seuil,remise):
        self.nArbres = nArbres
        self.pourcentage = pourcentageExemples
        self.seuil = seuil
        self.remise = remise
        self.foret = dict()
        self.echantillons = dict()
        self.sizeForet = 0

    def train(self,LSet):
        N = int(LSet.size() * self.pourcentage)
        for _ in range(self.nArbres):
            echantillon = echantillonLS(LSet,N,self.remise)
            arb_dec = ArbreDecision(0.0)
            arb_dec.train(echantillon)
            self.foret[self.sizeForet] = arb_dec
            self.echantillons[self.sizeForet] = echantillon
            self.sizeForet += 1

    def can_vote(self, k, position):
        echantillon = self.echantillons[k]
        for x in echantillon.x:
            foundInEchantillon = True
            for i in range(len(x)):
                if x[i] != position[i]:
                    foundInEchantillon = False
            if foundInEchantillon:
                return False
        return True

    def predict(self,x):
        right_to_vote = [self.can_vote(key, x) for key in self.echantillons]
        votes = np.array([self.foret[i].predict(x) for i in range(self.sizeForet) if right_to_vote[i] ])
        try :
            if votes.mean() >= 0 :
                return 1
            else:
                return -1
        except RuntimeWarning:
            return 1


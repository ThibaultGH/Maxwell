import base

import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as spla

filename = "carre"

base.constru_maillage(filename,10,10,1,1)

filename = "carre"

base.convertMSHtoMESH(filename)

[Nodes, Element] = base.read(filename + ".mesh")

print(Nodes)


#####################################

# QUESTION N°1

#####################################

#filename = "maillages/maillage3.mesh"

#[Nodes, Element] = base.read(filename)

#print(Nodes)

#print(Element)

#base.PlotMesh(filename)


#####################################

# QUESTION N°2

#####################################

# DONNÉES DU PROBLÈME :

w = 5*math.pi
d = [1,0]

#"0", "1", "2", "3", "4", "5", "6", "7"

for i in []:

    filename = "maillages/maillage" + i + ".mesh"
    
    base.FiniteElementP1(filename,w,d,1)


#####################################

# QUESTION N°3

#####################################

#k = 4

#base.RelativeError(k,d)


#####################################

# QUESTION N°4

#####################################

#tab_valp = open("tab_valp.txt", 'w')

for i in [] :

    filename = "maillages/maillage" + i + ".mesh"

    valp, vectp = base.EigVal(filename)

    for k in valp :
        
        tab_valp.write(str(round(k,3)) + "\t")

    tab_valp.write("\n")
    
    print(valp)



#####################################

# QUESTION N°5

#####################################

for i in [] :

    filename = "maillages/maillage" + i + ".mesh"

    base.GenEigVal(filename)



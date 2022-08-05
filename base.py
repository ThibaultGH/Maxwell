import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import math
import cmath
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import scipy.sparse.linalg as spla

###################################################################################################

#           CONSTRUCTION D'UN FICHIER DE MAILLAGE RECTANGULAIRE AU FORMAT GMSH

###################################################################################################




def constru_maillage(fichier,n_h,n_v,l_h,l_v):

        fichier += ".msh"

        dx = l_h/(n_h-1)
        dy = l_v/(n_v-1)

        rect_msh = open(fichier,"w")
        rect_msh.write("$Noeuds \n")
        rect_msh.write(str(n_h*n_v) + "\n")

        c = 0

        for j in range(n_v):
                for i in range(n_h):
                        rect_msh.write(str(c) + " " + str(dx*i) + " " + str(dy*j) + " " + str(0) + "\n")
                        c = c+1

        rect_msh.write("$FinNoeuds\n")
        rect_msh.write("$Elements\n")
        rect_msh.write(str(2*(n_h-1)*(n_v-1)) + "\n")

        c = 0

        for j in range(n_v-1):
                for i in range(n_h-1):
                        rect_msh.write(str(c) + " " + str(i+j*n_h) + " " + str(i+j*n_h+1) + " " + str(i+j*n_h+1+n_h) + "\n")
                        rect_msh.write(str(c+1) + " " + str(i+j*n_h) + " " + str(i+j*n_h+1+n_h) + " " + str(i+j*n_h+n_h) + "\n")
                        c = c+2

        rect_msh.write("$FinElements\n")

        return

###################################################################################################

#           CONVERTION D'UN FICHIER DE MAILLAGE RECTANGULAIRE DE .MSH A .MESH

###################################################################################################




def convertMSHtoMESH(filename):

    source0 = open(filename+".msh",'r')
    source1 = open(filename+".mesh",'w')
    
    line0 = source0.readlines()

    for i,b in enumerate(line0):
        line0[i] = b.replace("\n","")

    source1.write("#Nombres de noeuds\n")
    source1.write(line0[1] + "\n" + "\n" + "#Coordonnees des noeuds" + "\n")

    Nodes = int(line0[1])

    for i in range(Nodes):
            caractere = line0[i+2].split(" ")
            source1.write(caractere[1] + "\t" + caractere[2] + "\t" + caractere[3] + "\n")

    source1.write("\n" + "#Nombre de triangles" + "\n")
    Element = int(line0[Nodes+4])
    source1.write(str(Element) + "\n" + "\n")
    source1.write("#Numeros des sommets de chaque triangle" + "\n")

    for i in range(Element):
            caractere = line0[Nodes+5+i].split(" ")
            source1.write(str(int(caractere[1])+1) + "\t" + str(int(caractere[2])+1) + "\t" + str(int(caractere[3])+1) + "\n")

    return

###################################################################################################

#           EXTRACTION DES DONNÉES D'UN FICHIER .MESH

###################################################################################################




def read(filename):

    source = open(filename, 'r')

    line = source.readlines()

    for i,b in enumerate(line):
        line[i] = b.replace("\n","")

    NbNodes = int(line[1])
    #print(NbNodes)
    NbEle = int(line[NbNodes+6])
    #print(NbEle)
    
    tab_nodes = np.zeros((NbNodes,3))
    tab_ele = np.zeros((NbEle,3))

    for i in range(4,NbNodes+4):
        caractere = line[i].split("\t")
        for k in range(3):
            tab_nodes[i-4,k] = float(caractere[k])

    for j in range(NbNodes+9,NbEle+NbNodes+9):
        caractere = line[j].split("\t")
        for k in range(3):
            tab_ele[j-NbNodes-9,k] = int(caractere[k])

    return [tab_nodes,tab_ele]

###################################################################################################

#           AFFICHAGE D'UN MAILLAGE AU FORMAT .MESH

###################################################################################################




def PlotMesh(filename):

    [tab_nodes,tab_ele] = read(filename)

    #xy = np.asarray(tab_nodes)
    x = tab_nodes[:,0]
    y = tab_nodes[:,1]

    #triangles = np.asarray(tab_ele)

    plt.figure()
    plt.gca().set_aspect('equal')
    plt.triplot(x, y, tab_ele-1, '-go', lw = 1.0)

    plt.show()

    return

###################################################################################################

#           TRACE SUR UN MAILLAGE AU FORMAT .MESH

###################################################################################################




def PlotOnMesh(f,filename,title = ""):

        [tab_nds,tab_ele] = read(filename)
        tab_ele = tab_ele - 1
        
        x = tab_nds[:,0]
        y = tab_nds[:,1]

        plt.figure()
        plt.gca().set_aspect('equal')
        plt.tripcolor(x, y, tab_ele, f, edgecolor = 'k')
        plt.colorbar()
        plt.title(title)

        plt.show()

        return


###################################################################################################

#           CALCUL DE LA MATRICE DE MASSE ELEMENTAIRE SUR UN TRIANGLE

###################################################################################################



def MassElem(s1,s2,s3):
        A = abs((s2[0]-s3[0])*(s3[1]-s1[1])-(s2[1]-s3[1])*(s3[0]-s1[0]))
        M = A/24*(np.ones((3,3),float)+np.eye(3,3))
        return M

###################################################################################################

#           CALCUL DE LA MATRICE DE MASSE

###################################################################################################




def Mass(filename):

        [tab_nodes, tab_ele] = read(filename)
        
        nb_nodes = len(tab_nodes)
        nb_ele = len(tab_ele)
        tab_ele = tab_ele - 1

        Mass = lil_matrix((nb_nodes,nb_nodes), dtype = np.complex_)

        for q in range(nb_ele):
                T = tab_ele[q,:]
                s1 = tab_nodes[int(T[0]),:]
                s2 = tab_nodes[int(T[1]),:]
                s3 = tab_nodes[int(T[2]),:]
                for i in range(3):
                        for j in range(3):
                                Mass[int(T[i]),int(T[j])] += MassElem(s1,s2,s3)[i,j]

        return Mass

###################################################################################################

#           CALCUL DE LA MATRICE DE RIGIDITÉ ÉLÉMENTAIRE SUR UN TRIANGLE

###################################################################################################




def RigElem(s1,s2,s3):
        N = np.zeros((3,3))
        N[:,0] = s3-s2
        N[:,1] = s1-s3
        N[:,2] = s2-s1
        A = abs((s2[0]-s3[0])*(s3[1]-s1[1])-(s2[1]-s3[1])*(s3[0]-s1[0]))
        K = np.zeros((3,3))
        for i in range(0,3):
                for j in range(0,3):
                        K[i,j] = 1./(2*A)*(np.vdot(N[:,i],N[:,j]))
        return K

###################################################################################################

#           CALCUL DE LA MATRICE DE RIGIDITÉ

###################################################################################################




def Rig(filename):

        [tab_nodes, tab_ele] = read(filename)

        nb_nodes = len(tab_nodes)
        nb_ele = len(tab_ele)
        tab_ele = tab_ele - 1

        Rig = lil_matrix((nb_nodes,nb_nodes), dtype = np.complex_)

        for q in range(nb_ele):
                T = tab_ele[q,:]
                s1 = tab_nodes[int(T[0]),:]
                s2 = tab_nodes[int(T[1]),:]
                s3 = tab_nodes[int(T[2]),:]
                for i in range(3):
                        for j in range(3):
                                Rig[int(T[i]),int(T[j])] += RigElem(s1,s2,s3)[i,j]

        return Rig

###################################################################################################

#           DÉTERMINATION DES ARÈTES SUR LE BORD ET DE LEUR TRIANGLE

###################################################################################################




def FindBoundary(filename) :

        source_file = open(filename,"r")

        Nbn = len(read(filename)[0])
        ele_table = read(filename)[1] - 1

        Edge = []
        ThirdPoint = []
        First = -1*np.ones(Nbn,int)
        Next = []
        BoolBoundary = []
        ThirdPointBoundary = []
        Boundary = []
        pos = 0

        for tria in ele_table :

                for i in range(3):
        
                        v0 = min(tria[i], tria[(i+1)%3])
                        v1 = max(tria[i], tria[(i+1)%3])
                        v2 = tria[(i+2)%3]
                        NewEdge = [v0, v1]

                        I = First[int(v0)]
                        Found = 0
    
                        while ((Found == 0) & (I != -1)):
                                if (NewEdge == Edge[I]):
                                        Found = 1
                                        BoolBoundary[I] = 0
                                else :
                                        I = Next[I]
    
                        if (Found == 0) :
                                Edge += [NewEdge]
                                BoolBoundary += [1]
                                ThirdPoint += [v2]
                                Next += [First[int(v0)]]
                                First[int(v0)] = pos
                                pos = pos + 1

        for i in range(len(Edge)) :
                if (BoolBoundary[i] == 1):
                        Boundary += [Edge[i]]
                        ThirdPointBoundary += [ThirdPoint[i]]                        

        #Boundary = np.asarray(Boundary)
        
        return Boundary, ThirdPointBoundary

###################################################################################################

#           CALCUL DE LA NORME D'UN VECTEUR

###################################################################################################




def norm(v):
    return math.sqrt(np.vdot(v,v))

###################################################################################################

#           CALCUL DE LA FONCTION SECOND MEMBRE G

###################################################################################################




def g(x,w,d,Normal):
    return np.vdot(d,Normal)*cmath.exp(1j*w*np.vdot(d,x))

###################################################################################################

#           CALCUL DE LA SOLUTION EXACTE

###################################################################################################




def Uref(Nodes,w,d):
    Uref = np.zeros(len(Nodes), dtype = np.complex_)
    for p in range(len(Nodes)):
        Uref[p] = (-1j/w)*cmath.exp(1j*w*np.vdot(d,Nodes[p,0:2]))
    return Uref

###################################################################################################

#           CALCUL DU VECTEUR UNITAIRE EXTERIEUR A UNE ARÈTE DU BORD

###################################################################################################




def ExtNormal(Edge, ThirdPoint, Nodes):
    Vect1 = Nodes[int(Edge[1]),0:2]-Nodes[int(Edge[0]),0:2]
    Vect1 = Vect1/norm(Vect1)
    Vect2 = Nodes[int(ThirdPoint),0:2]-Nodes[int(Edge[0]),0:2]
    Vect3 = np.vdot(Vect2,Vect1)*Vect1
    Vect4 = Nodes[int(Edge[0]),0:2]+Vect3
    Vect5 = Vect4-Nodes[int(ThirdPoint),0:2]
    #print(np.vdot(Vect1,Vect5))
    #print(np.vdot(Vect5/norm(Vect5),Vect2))
    return Vect5/norm(Vect5)

###################################################################################################

#           CALCUL DE LA CONTRIBUTION D'UNE ARÈTE DU BORD AU SECOND MEMBRE

###################################################################################################




def EdgeElem(g,edge,Nodes,w,d,Normal) :
        G = g(Nodes[int(edge[0])][0:2],w,d,Normal)
        D = g(Nodes[int(edge[1])][0:2],w,d,Normal)
        M = g(0.5*(Nodes[int(edge[0])][0:2] + Nodes[int(edge[1])][0:2]),w,d,Normal)
        L = norm(Nodes[int(edge[0])][0:2] - Nodes[int(edge[1])][0:2])
        return np.asarray([(G + 2*M)*L/6, (2*M + D)*L/6], dtype = np.complex_)

###################################################################################################

#           CALCUL DU SECOND MEMBRE

###################################################################################################



       
def BoundaryCondition(Nodes,Bound,ThirdPoint,w,d) :
        B = np.zeros(len(Nodes), dtype = np.complex_)
        #test = np.zeros(len(Bound))
        for q in range(len(Bound)) :
            #test[int(Bound[q][0])] += 1
            #test[int(Bound[q][1])] += 1
            Normal = ExtNormal(Bound[q],ThirdPoint[q],Nodes)
            b = EdgeElem(g,Bound[q],Nodes,w,d,Normal)
            for l in range(2) :
                B[int(Bound[q][l])] += b[l]
        #print(test)
        return B

###################################################################################################

#           RESOLUTION DU PROBLÈME ET TRACÉ DE LA SOLUTION PAR UNE METHODE D'ELEMENTS FINIS P1

###################################################################################################




def FiniteElementP1(filename,w,d,plot):
        
    print("_ Extraction des informations du fichier de maillage")

    [Nodes, Element] = read(filename)

    print("Fin : Extraction des informations du fichier de maillage \n \n")

    #print(Nodes)

    N = len(Nodes)

    #PlotMesh(filename)

    

    #########################################

    # CALCUL ET TRACÉ DE LA SOLUTION EXACTE :

    #########################################




    Uexa = Uref(Nodes,w,d)

    if(plot == 1):

            print("_ Calcul de la solution exacte et tracé")

            PlotOnMesh(Uexa.real,filename,filename + " - solution exacte, w = " + str(w))

            print("Fin : Calcul de la solution exacte et tracé \n \n")
    

    #########################################

    # ASSEMBLAGE DE LA MATRICE DE MASSE ET TEST :

    #########################################

    

    
    print("_ Assemblage de la matrice de masse")
    
    M = Mass(filename)

    print("Fin : Assemblage de la matrice de masse \n \n")
    
    #print(M[0,0])

    #print(M)

    f = np.ones((N,1))

    #AireOmega = 0

    #for trian in Element :
        #v1 = Nodes[int(trian[1])-1,0:2] - Nodes[int(trian[0])-1,0:2]
        #v2 = Nodes[int(trian[2])-1,0:2] - Nodes[int(trian[0])-1,0:2]
        #v3 = np.vdot(v1,v2)*v2/norm(v2)
        #v4 = v3 + Nodes[int(trian[0])-1,0:2]
        #v5 = v4 - Nodes[int(trian[1])-1,0:2]
        #AireOmega += norm(v5)*norm(v2)/2

    #print(np.vdot(np.transpose(f),M.dot(f))-(math.pi*4-1.7*0.2*2-0.2))#-AireOmega)
    #print(math.pi*4-1.7*0.2*2-0.2)
    #print(AireOmega)
    

    #########################################

    # ASSEMBLAGE DE LA MATRICE DE RIGIDITÉ ET TEST :

    #########################################



    print("_ Assemblage de la matrice de rigidité")

    K = Rig(filename)

    #print(np.dot(K,f))

    print("Fin : Assemblage de la matrice de rigidité \n \n")
    
    
    #########################################

    # ASSEMBLAGE DES TERMES SURFACIQUES ET TEST :

    #########################################



    print("_ Détermination des arètes du bord et de leurs triangles")
    
    [Bound, ThirdPoint] = FindBoundary(filename)

    print("Fin : Détermination des arètes du bord et de leurs triangles \n \n")    

        # TEST SUR L'ORTHOGONALITÉ :

        
    #for j in range(len(Bound)):
        #Normal = ExtNormal(Bound[j],ThirdPoint[j],Nodes)
        #VectEdge = Nodes[int(Bound[j][0]),0:2] - Nodes[int(Bound[j][1]),0:2]
        #print(np.vdot(Normal,VectEdge))
        #print(Bound[j])
        #print(Normal)
        #print(math.sqrt(2)/2)
        

        # ASSEMBLAGE DU SECOND MEMBRE :

    print("_ Assemblage des termes surfaciques")
        
    B = BoundaryCondition(Nodes,Bound,ThirdPoint,w,d)

    print("Fin : Assemblage des termes surfaciques \n \n")
    
    #print(B)

    
        # TEST SUR LE SECOND MEMBRE :

        
    #integ = 0
    
    #for k in range(len(Bound)):
        #Normal = ExtNormal(Bound[k],ThirdPoint[k],Nodes)
        #integ += (g(Nodes[int(Bound[k][0]),0:2],w,d,Normal)+4*g(0.5*(Nodes[int(Bound[k][0]),0:2]+Nodes[int(Bound[k][1]),0:2]),w,d,Normal)+g(Nodes[int(Bound[k][1]),0:2],w,d,Normal))*norm(Nodes[int(Bound[k][0]),0:2]-Nodes[int(Bound[k][1]),0:2])/6

    #print(np.vdot(B,f)-integ)

    

    #########################################

    # ASSEMBLAGE DE LA MATRICE DU SYSTÈME LINÉAIRE, RÉSOLUTION ET TRACÉ DE LA SOLUTION :

    #########################################


    print("_ Assemblage de la matrice du système linéaire et résolution du système")
    
    A = csr_matrix(K-(w**2)*M, dtype = np.complex_)

    X = spla.spsolve(A,B)

    print("Fin : Assemblage de la matrice du système linéaire et résolution du système \n \n")


    if (plot == 1):

            print("_ Tracé de la solution")

            PlotOnMesh(X.real,filename,filename + " - solution approchée, w = " + str(w))

            print("Fin : Tracé de la solution \n \n")
    

    #########################################

    return X

###################################################################################################

#           CALCUL DE LA LONGUEUR CARACTÉRISITQUE D'UN MAILLAGE AU FORMAT .MESH

###################################################################################################




def CaractSize(filename):

    [Nodes, Element] = read(filename)

    H = 0

    for trian in Element:

            for i in range(3) :
                    
                    h = np.zeros(3)
                    v1 = Nodes[int(trian[(i+1)%3])-1,0:2]
                    v2 = Nodes[int(trian[(i+2)%3])-1,0:2]

                    v = np.asarray(v2-v1)

                    h[i] = norm(v)

            if (max(h) > H):
                    H = max(h)

    return H

###################################################################################################

#           CALCUL ET TRACÉ DE L'ERREUR RELATIVE COMMISE PAR LA MÉTHODE D'ELEMENTS FINIS P1

###################################################################################################




def RelativeError(k,d):

        E = np.zeros((k,5))
        H = np.zeros(k)

        W = [5, 10, 20, 40, 80]

        for i in range(1,k+1):

                print("\n")
                
                print("\n")

                print("MAILLAGE N°" + str(i))

                print("\n")

                print("\n")

                filename = "maillages/maillage" + str(i) + ".mesh"

                H[i-1] = CaractSize(filename)

                [Nodes, Element] = read(filename)

                for j in range(len(W)) :

                        Uexa = Uref(Nodes,W[j],d)

                        X = FiniteElementP1(filename,W[j],d,0)

                        E[i-1,j] = norm(X-Uexa)/norm(X)

        plt.figure(1)
        #plt.gca().set_aspect('equal')
        plt.loglog(H,E[:,0])
        plt.loglog(H,E[:,1])
        plt.loglog(H,E[:,2])
        plt.loglog(H,E[:,3])
        plt.loglog(H,E[:,4])
        plt.title("Tracé de l'erreur pour différents w selon les " + str(k) +" premiers maillages")
        plt.legend(["w = 5","w = 10","w = 20","w = 40","w = 80"], loc = 'upper left')

        plt.show()

        return

###################################################################################################

#           CALCUL DES 6 PLUS PETITES VALEURS PROPRES DE (3) ET VECTEURS PROPRES GÉNÉRALISÉS

###################################################################################################




def EigVal(filename):

        M = Mass(filename)

        K = Rig(filename)

        sigma = 0.0001

        k = 6

        valp, vectp = spla.eigsh(K,k,M,sigma,which = 'LM')

        return valp, vectp

###################################################################################################

#           TRACÉ DES VECTEURS PROPRES GÉNÉRALISÉS

###################################################################################################





def GenEigVal(filename):

        valp, vectp = EigVal(filename)

        for i in range(6):

                PlotOnMesh(np.real(vectp[:,i]), filename, filename + " pour la valeur propre " + str(round(valp[i],3)))

        return

###################################################################################################

#             FIN

###################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import*
from tqdm import *
from tqdm import tqdm
import scipy
def varience(img):
    nl,nc=img.shape
    moy=np.sum(img/255)/(nl*nc)
    return np.sum((img/255-moy)**2)/(nl*nc)

def energy(img):
    return np.sum((img/255)**2)


def entropy(mat):
    img=mat/255
    e=np.sum(img*np.log2(img+10e-10))
    return -e



def contrast(mat):
    c=0
    img=mat/255
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            c+=(i-j)**2*img[i,j]
    return c

def moment_dif_invr(mat):
    m=0
    img=mat/255
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            m+=img[i,j]/(1+np.abs(i-j))
    return m

def covariance(mat):
    return np.cov(mat)

def get_stats(img):
    return np.array([varience(img),energy(img),entropy(img),contrast(img),moment_dif_invr(img)])


def Euclidien(vect1,vect2):
    return np.sqrt(sum((vect2-vect1)**2))

def matrix_distance(discrpt_Train,discrpt_Test):
    #nbr ligne de la matrice distance
    nbr_ligne=discrpt_Train.shape[0]
    #nbr colonne de la matrice distance
    nbr_col=discrpt_Test.shape[0]
    mat_dist=np.zeros((nbr_ligne,nbr_col))
    for i in tqdm(range(nbr_ligne)):
        for j in range(nbr_col):
            mat_dist[i][j]=Euclidien(discrpt_Train[i],discrpt_Test[j])
    return mat_dist

def index_similarite(mat_dist):
    return  (np.argsort(mat_dist))

def moyenne_precision(vect):
    return np.mean(vect)*100


#class_global :la class global qu'on sur dans laquel en va extraire les img
#indece_class: l'indice de la class a selectionner
#nbr_img_selectionner:nombre d'image qu'on vaux selectionner
#dimension de l'image : dans notre cas dim_img=28
def selectionner_une_class(class_global,indece_class,nbr_img_selectionner,dim_img):
    class_i=np.zeros((dim_img,dim_img,nbr_img_selectionner))
    conteur=0
    for i in range(class_global.shape[0]):
        if class_global[i,0]==indece_class and conteur<nbr_img_selectionner:
            class_i[:,:,conteur]=np.reshape(class_global[i,1:],(dim_img,dim_img))
            conteur+=1
    return class_i   

def load_img(class_global,nbr_img,dim_img):
    #la base a reteurne 
    base_img=np.zeros((dim_img,dim_img,nbr_img*10))
    for indice in  tqdm(range(10)):
        base_img[:,:,indice*nbr_img:(indice+1)*nbr_img]=selectionner_une_class(class_global,indice,nbr_img,dim_img)
    return  base_img
#pour afficher une class 
def afficher_class(class_i):
    for i in range(class_i.shape[2]):
        plt.figure(figsize=(4,4))
        plt.imshow(class_i[:,:,i],'gray')
        plt.title(i+1)
    plt.show()
    
    
def tableau_recall_prec_moyp(m):
    r=np.zeros((m.shape))
    p=np.zeros((m.shape))
    moy_p=[]
    moy_r=[]
    for i in range(m.shape[0]):
        r[i]=(np.sum(m[:i+1])/np.sum(m))
        p[i]=(np.sum(m[:i+1])/(i+1))
        if m[i]==1:
            moy_p.append(p[i])
            
    return (r,p,sum(moy_p)/len(moy_p))
    


def evaluer_modele(mat_bin):
    mat_rappel=np.zeros(mat_bin.shape)
    mat_precision=np.zeros(mat_bin.shape)
    moy_pre=np.zeros((mat_bin.shape[0],1))
    moy_rap=np.zeros((mat_bin.shape[0],1))
    for i in tqdm(range(mat_bin.shape[0])):
        mat_rappel[i],mat_precision[i],moy_pre[i]=tableau_recall_prec_moyp(mat_bin[i])
    return (mat_rappel,mat_precision,moy_pre)

def discripteur_stats(base_img):
    nbr_img=base_img.shape[2]
    print("nbr img dans train et test:",nbr_img)
    discrpt_base=np.zeros((nbr_img,5))
    for i in tqdm(range(0,nbr_img)):
        discrpt_base[i]=get_stats(base_img[:,:,i])
    return discrpt_base

def discripteur_histo(base_img):
    nl,nc,nbr_img=base_img.shape
    print("nbr img dans train et test:",nbr_img)
    discrptripteur=np.zeros((nbr_img,256))
    for i in tqdm(range(nbr_img)):
        discrptripteur[i]=histo(base_img[:,:,i])#/(nl*nc)
    return discrptripteur

def histo(I):
    [nl,nc]=np.shape(I)
    h=np.zeros(256)
    I_m=np.around(I)
    for i in range(0,nl):
        for j in range(0,nc):
            val=int(I_m[i][j])
            h[val]+=1
    return h
def matrix_boolean(matrice_indice,nbr_img_class_train,nbr_img_class_test):
    #initialisation min max
    minimun=0
    maximum=nbr_img_class_test-1
    #initialisation matrice binare par zeros
    matrice_binair=np.zeros(matrice_indice.shape)
    class_courant=0
    conteur=0
    nbr_ligne=matrice_binair.shape[0]
    for i in tqdm(range(nbr_ligne)):
        s1=matrice_indice[i] >=minimun
        s2=matrice_indice[i] <= maximum
        matrice_binair[i]=s1 * s2
        #print("class:",class_courant,"conteur:",conteur,"[",minimun,'-',maximum,']')
        conteur+=1
        if (conteur==nbr_img_class_train):
            class_courant+=1
            minimun+=nbr_img_class_test
            maximum+=nbr_img_class_test
            conteur=0
            
    return  matrice_binair




def erosion(base_img):
    nbr_img=base_img.shape[2]
    print("nbr img dans train et test:",nbr_img)
    base_traiter=np.zeros(base_img.shape)
    for i in tqdm(range(0,nbr_img)):
        base_traiter[:,:,i]=scipy.ndimage.grey_erosion(base_img[:,:,i],size=(3,3))
    return base_traiter




#######la forme##########
Sx=np.array([[-1,0,1],[-2,0,2],[-1,0,1],])
Sy=np.array([[-1,-2,-1],[0,0,0],[1,2,1],])


def detecter_contour(data,Fx,Fy):
    data_form=np.zeros(data.shape)
    nbr_img=data.shape[2]
    for i in tqdm(range(nbr_img)) :
        data_form[:,:,i]=detect_contour(data[:,:,i],Fx,Fy)
    return data_form

def detect_contour(img,Fx,Fy):
    Gx=convolution(img,Fx)
    Gy=convolution(img,Fy)
    return np.sqrt(Gx**2+Gy**2)

def convolution(img,noyau):
    pas=noyau.shape[0]//2
    nl,nc=img.shape
    I=np.zeros(img.shape)
    for i in range(pas,nl-pas):
        for j in range(pas,nc-pas):
            I[i,j]=np.sum(img[i-pas:i+pas+1,j-pas:j+pas+1]*noyau)
    return I

   
def discripteur_hog(base_img):
    from skimage.feature import hog
    nl,nc,nbr_img=base_img.shape
    size_desc=hog(base_img[:,:,0],orientations=4, pixels_per_cell=(3,3),cells_per_block=(4,4),visualize=False, feature_vector=True).shape[0]
    discrptripteur_hog=np.zeros((nbr_img,size_desc))
    for i in tqdm(range(nbr_img)):
        desc=hog(base_img[:,:,i],orientations=4, pixels_per_cell=(3,3),cells_per_block=(4,4),visualize=False, feature_vector=True)
        discrptripteur_hog[i]= desc/np.linalg.norm(desc)
    return discrptripteur_hog



 




def couleur_moyenne(m):
    return np.sum(m)/(m.shape[0]*m.shape[1])



def pretraitement_1(base_image):
    nl,nc,nb_img=base_image.shape
    base_triter=np.zeros((nl,nc,nb_img))
    for i in range(nb_img):
        max_filter=scipy.ndimage.filters.convolve1d(base_image[:,:,i], weights=[3,3,4,4],axis=-1)
        median_filter = scipy.ndimage.median_filter(max_filter, 3)
        gamma = 0.5
        base_triter[:,:,i] = np.power(  median_filter, gamma)   
    return base_triter
    
    
    
def CMC(m,ic,ir):
    nl,nc=m.shape
    cmc=np.zeros((nl),int)
    check_compare=np.ones((nl),int)
    taux=0
    mx=np.max(m)+1
    j=0
    while  j<nl  and np.sum(check_compare)!=0:
        for col in range(nc):
            if check_compare[col]==1:
                ind_l_min=np.argmin(m[:,col])
                if ic[ind_l_min]==ir[col]:
                    taux+=1
                    check_compare[col]=0
                else:
                    m[ind_l_min,col]=mx
        cmc[j]=taux
        taux=0
        j+=1
    return np.cumsum(cmc)/200
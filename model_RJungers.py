# -*- coding: utf-8 -*-
"""
@author: Aurélien Desart
"""
import numpy as np 
import matplotlib.pyplot as plt
import csv
import pandas as pd
import time
import copy
import math
import sklearn



from sklearn import svm, datasets
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid




#################### Fonctions pour le modèle ####################


def smooth_data(data): 
    smooth_data = np.zeros(data.size)
    for t in range(0,3):
        smooth_data[t] = data[t]
    for t in range(3,data.size-3):
        smooth_data[t] = np.mean(data[t-3:t+4])
    for t in range(data.size-3,data.size):
        i=3
        smooth_data[t] = np.mean(data[t-3:t+i])
        i = i-1
    return smooth_data


def model(N0,j,i0,i1,beta,h0,h1,gamma,c1,d1,c,d,tend):
    #parameters
    N = 10000000 #taille totale population
    kmax = h1+1 #Nombre de jours max qu'une personne reste infectée

    #initialisation des coeff
    coeff = np.zeros(tend)
    nbrpoint = len(d)
    coeff[0:d1] = 1
    if (nbrpoint == 0):
        coeff[d1:tend] = c1
    else:
        coeff[d1:int(d[0])] = c1
        for i in range(0,nbrpoint-1):
            coeff[int(d[i]):int(d[i+1])] = c[i]
        coeff[int(d[nbrpoint-1]):tend] = c[nbrpoint-1]
        
    #Initialisation vecteur beta
    rbeta = np.zeros(kmax+1)
    rbeta[i0:i1+1] = beta

    #Initialisation vecteur gamma
    rgamma = np.zeros(kmax+1)
    rgamma[h0:h1+1] = gamma
    
    #initialisation vecteur systeme
    S = np.zeros(tend+1)
    Ik = np.zeros((tend+1,kmax+1))
    H = np.zeros(tend+1)
    
    #Conditions initiales
    S[0] = N - N0*j
    Ik[0,0:j] = N0
    
    #Systeme
    for t in range(0,tend):
        #Infection et hospitalisation
        S[t+1] = S[t] - Ik[t,0]

        newinfections = coeff[t]*(S[t]/N)*np.dot(rbeta, Ik[t,:])
        if (newinfections > 1000000): 
            return np.zeros((tend+1,kmax+1)), np.zeros(tend+1)
        else:  
            Ik[t+1,0] = newinfections
            for k in range(0,kmax):
                Ik[t+1,k+1] = (1-rgamma[k]) * Ik[t,k]
            H[t+1] =  np.dot(rgamma, Ik[t+1,:])

    return Ik , H

def R0(i0,i1,beta,h0,h1,gamma,c1,d1,c,d,tend):
    kmax = int(h1+1)
    Re = np.zeros(tend)#création du vecteur R0
    
    #initialisation des coeff
    coeff = np.zeros(tend)
    nbrpoint = len(d)
    coeff[0:d1] = 1
    if (nbrpoint == 0):
        coeff[d1:tend] = c1
    else:
        coeff[d1:int(d[0])] = c1
        for i in range(0,nbrpoint-1):
            coeff[int(d[i]):int(d[i+1])] = c[i]
        coeff[int(d[nbrpoint-1]):tend] = c[nbrpoint-1]
    
    #Initialisation vecteur beta
    rbeta = np.zeros(kmax+1)
    rbeta[i0:i1+1] = beta

    #Initialisation vecteur gamma
    rgamma = np.zeros(kmax+1)
    rgamma[h0:h1+1] = gamma
    
    sum_i = 0
    for t in range(tend):
        sum_i = 0
        for i in range(0,kmax+1):
            product_i = 1
            for j in range(0,i-1):
                product_i = product_i * (1-rgamma[j])
            sum_i = sum_i + rbeta[i] * product_i
        Re[t] = coeff[t]*sum_i
    return Re

def gridsearch(parameters,data):
    nbrparam = len(list(ParameterGrid(parameters)))
    error=np.zeros((nbrparam,2))
    index=0
    for param in ParameterGrid(parameters):
        if (param["i1"] - param["i0"] <= 0) or (param["h1"] - param["h0"] <= 0):
            error[index]= (float('inf'),index)
            index = index + 1
            continue
            
        else:
            Ik,H = model(param["N0"],param["j"],param["i0"],param["i1"],param["beta"],param["h0"],param["h1"],
                         param["gamma"],param["c1"],param["d1"],[],[],len(data)-1)
            if((H == 0).all()):
                error[index,:] = (float('inf'),index)
            else:
                error[index] = (mean_absolute_error(data, H),index)
            index = index+1
            if (index%100000 == 0):
                print(index)
    best_results = error[error[:,0].argsort()]
    best_estimators = best_results[0:300]
    index_best_estimators = best_estimators[:,1]
    best300 = xbestparameters(index_best_estimators, list(ParameterGrid(parameters)),300)
    return best300

def testbestx(parameters,data,nbr):
    nbrparam = len(parameters)
    error=np.zeros((nbrparam,2))
    index=0
    for param in parameters:
        if (param["i1"] - param["i0"] <= 0) or (param["h1"] - param["h0"] <= 0):
            continue
        else:
            Ik,H = model(param["N0"],param["j"],param["i0"],param["i1"],param["beta"],param["h0"],param["h1"],
                         param["gamma"],param["c1"],param["d1"],param["c"],param["d"],len(data)-1)
            error[index,:] = (mean_absolute_error(data, H),index)
            index = index+1
        
    best_results = error[error[:,0].argsort()]
    best_estimators = best_results[0:nbr]
    index_best_estimators = best_estimators[:,1]
    best300_6000 = xbestparameters(index_best_estimators, parameters,nbr)
    return best300_6000

def xbestparameters(index_best_estimators,base,nbrbest):
    result = nbrbest * [0] #initilisation comme ça car liste de dic
    index = 0
    for i in index_best_estimators:
       result[index] = base[int(i)]
       index = index + 1
    return result

def print_paramDistrib(best300,data):
    nbrparam = len(best300)
    graphe_beta = np.zeros((nbrparam,2))
    graphe_gamma= np.zeros((nbrparam,2))
    graphe_bgk = np.zeros((nbrparam,2))
    error = np.zeros(nbrparam)
    index=0
    for param in best300:
        Ik,H = model(param["N0"],param["j"],param["i0"],param["i1"],param["beta"],param["h0"],param["h1"],
                     param["gamma"],param["c1"],param["d1"],[],[],len(data)-1)
        error[index] = mean_absolute_error(data, H)
        difi = param["i1"] - param["i0"]
        difh = param["h1"] - param["h0"]
        beta = param["beta"]
        gamma = param["gamma"]
        graphe_beta[index] = [difi,beta]
        graphe_gamma[index] = [difh,gamma]
        graphe_bgk[index] = [beta*difi,gamma*difh]
        index = index+1
        
    beta300 = np.transpose(graphe_beta)
    gamma300 = np.transpose(graphe_gamma)
    bgk300 = np.transpose(graphe_bgk)
    
    plt.rcParams["figure.figsize"] = (50, 10)

    plt.subplot(1, 3, 1)
    plt.title("param beta")
    plt.scatter(beta300[0],beta300[1],c=error,cmap="RdBu")
    plt.xlabel('i1-i0')
    plt.ylabel('beta')
    #plt.show()

    plt.subplot(1, 3, 2)
    plt.title("param gamma")
    plt.scatter(gamma300[0],gamma300[1],c=error,cmap="RdBu")
    plt.xlabel('h1-h0')
    plt.ylabel('gamma')
    #plt.show()

    plt.subplot(1, 3, 3)
    plt.title("param beta/gamma")
    points_bgk = plt.scatter(bgk300[0],bgk300[1],c=error,cmap="RdBu")
    plt.xlabel('sum beta')
    plt.ylabel('sum gamma')
    plt.colorbar(points_bgk)
    plt.savefig('parameters.png')
    plt.show()

    return np.transpose(graphe_beta),np.transpose(graphe_gamma),np.transpose(graphe_bgk),error
    
def newcd(c1,nbrd):
    dict={}
    for i in range(2,nbrd+1):
        keyc=str("c"+str(i))
        keyd=str("d"+str(i))
        if i%2 == 1:
            dict[keyc] = np.geomspace(0.5*c1,2*c1,8).tolist()
        else:
            dict[keyc] = np.geomspace(2*c1,15 *c1,8).tolist()
        dict[keyd] = np.linspace(153,279,22).tolist()

    return dict

def getCD(elem):
    size = int(len(elem)/2)
    c=np.zeros(size)
    d=np.zeros(size)
    for i in range(0,size):
        c[i] = elem["c"+str(i+2)]
        d[i] = elem["d"+str(i+2)]
    return c,d

def is_sorted_asc(ar):
    for i in range(1, len(ar)):
        if ar[i-1] < ar[i]:
            continue
        else:
            return False
    return True

def climit(c):
    for i in range(0,len(c)):
        if c[i] > 2:
            return False
    return True

def reunite(best20,elem,nbrd):
    size = nbrd-1
    for i in range(0,20):
        best20[i]["N0"] = elem["N0"]
        best20[i]["j"] = elem["j"]
        best20[i]["i0"] = elem["i0"]
        best20[i]["i1"] = elem["i1"]
        best20[i]["beta"] = elem["beta"]
        best20[i]["h0"] = elem["h0"]
        best20[i]["h1"] = elem["h1"]
        best20[i]["gamma"] = elem["gamma"]
        best20[i]["c1"] = elem["c1"]
        best20[i]["d1"] = elem["d1"] 
        c= np.zeros(size)
        d= np.zeros(size)
        for j in range(0,size):
            c[j] = float(best20[i]["c"+str(j+2)])
            d[j] = float(best20[i]["d"+str(j+2)])
            del(best20[i]["c"+str(j+2)])
            del(best20[i]["d"+str(j+2)])
        best20[i]["c"] = c.tolist()
        best20[i]["d"] = d.tolist()

def iter_FGS(param,data,nbrd):
    index = 0
    c1 = param["c1"]
    newPGrid = newcd(c1,nbrd)
    nbrparam = len(list(ParameterGrid(newPGrid)))
    error= np.zeros((nbrparam,2))
    for elem in ParameterGrid(newPGrid):
        c,d = getCD(elem)
        if is_sorted_asc(d):
            Ik,H = model(param["N0"],param["j"],param["i0"],param["i1"],param["beta"],param["h0"],param["h1"],
                         param["gamma"],param["c1"],param["d1"],c,d,len(data)-1)
            if((H == 0).all()):
                error[index,:] = (float('inf'),index)
            else:
                s_h = smooth_data(H)
                error[index,:] = (mean_absolute_error(data, s_h),index)
            index = index+1
        else:
            error[index,:] = (float('inf'),index)
            index = index+1
            continue
    
    best_results = error[error[:,0].argsort()]
    best_estimators = best_results[0:20]
    index_best_estimators = best_estimators[:,1]
    best20 = xbestparameters(index_best_estimators, list(ParameterGrid(newPGrid)),20)
    reunite(best20,param,nbrd)
    return best20


def finalGridSearch(best300,data,nbrd):
    size = len(best300)
    index = 0
    result = 20*size * [0] #initilisation comme ça car liste de dic
    for param in best300:        
        param_best20 = iter_FGS(param,data,nbrd)
        result[index:index+20] = param_best20
        index = index + 20
        print(index/20)
    return result
       
def fprint(opti_param,data,fw):
    data_size=len(data)
    nbr_param = len(opti_param)
    resultH = np.zeros((nbr_param,data_size))
    resultI = np.zeros(((nbr_param,data_size)))
    r = np.zeros((nbr_param,data_size))
    for i in range(0,nbr_param):
        if fw:
            I,H = model(opti_param[i]["N0"],opti_param[i]["j"],opti_param[i]["i0"],opti_param[i]["i1"],
                        opti_param[i]["beta"],opti_param[i]["h0"],opti_param[i]["h1"],opti_param[i]["gamma"],
                        opti_param[i]["c1"],opti_param[i]["d1"],[],[],len(data)-1)
            
            Rt = R0(opti_param[i]["i0"],opti_param[i]["i1"],opti_param[i]["beta"],opti_param[i]["h0"],
                    opti_param[i]["h1"],opti_param[i]["gamma"],opti_param[i]["c1"], opti_param[i]["d1"],
                    [], [], len(data))
            
        else:
            I,H = model(opti_param[i]["N0"],opti_param[i]["j"],opti_param[i]["i0"],opti_param[i]["i1"],
                        opti_param[i]["beta"],opti_param[i]["h0"],opti_param[i]["h1"],opti_param[i]["gamma"],
                        opti_param[i]["c1"],opti_param[i]["d1"],opti_param[i]["c"],opti_param[i]["d"],len(data)-1)
            
            Rt = R0(opti_param[i]["i0"],opti_param[i]["i1"],opti_param[i]["beta"],opti_param[i]["h0"],
                    opti_param[i]["h1"],opti_param[i]["gamma"],opti_param[i]["c1"], opti_param[i]["d1"],
                    opti_param[i]["c"],opti_param[i]["d"], len(data))
            
        resultH[i] = smooth_data(H)
        resultI[i] = smooth_data(I[:,0])
        r[i] = Rt
    resultH_t = np.transpose(resultH)
    resultI_t = np.transpose(resultI)
    rtrans = np.transpose(r)
    meanH_result = np.zeros(data_size)
    stdH_result = np.zeros(data_size)
    meanI_result = np.zeros(data_size)
    stdI_result = np.zeros(data_size)
    R_mean = np.zeros(data_size)
    R_std = np.zeros(data_size)
    for i in range(0,data_size):
        meanH_result[i] = np.mean(resultH_t[i])
        stdH_result[i] = np.std(resultH_t[i])
        meanI_result[i] = np.mean(resultI_t[i])
        stdI_result[i] = np.std(resultI_t[i])  
        R_mean[i] = np.mean(rtrans[i])
        R_std[i] = np.std(rtrans[i])
        
    return meanH_result,stdH_result,meanI_result,stdI_result,R_mean, R_std

def moyenne_param1(opti_param):
    nbr_param = len(opti_param)
    vec_N0 = np.zeros(nbr_param)
    vec_j = np.zeros(nbr_param)
    vec_i0 = np.zeros(nbr_param)
    vec_i1 = np.zeros(nbr_param)
    vec_beta = np.zeros(nbr_param)
    vec_h0 = np.zeros(nbr_param)
    vec_h1 = np.zeros(nbr_param)
    vec_gamma = np.zeros(nbr_param)
    vec_c1 = np.zeros(nbr_param)
    vec_d1 = np.zeros(nbr_param)
    
    for i in range(0,nbr_param):
        vec_N0[i]= opti_param[i]["N0"]
        vec_j[i] = opti_param[i]["j"]
        vec_i0[i] = opti_param[i]["i0"]
        vec_i1[i] = opti_param[i]["i1"]
        vec_beta[i] = opti_param[i]["beta"]
        vec_h0[i] = opti_param[i]["h0"]
        vec_h1[i] = opti_param[i]["h1"]
        vec_gamma[i] = opti_param[i]["gamma"]
        vec_c1[i] = opti_param[i]["c1"]
        vec_d1[i] = opti_param[i]["d1"]
        
    mean_param = np.zeros(10)
    std_param = np.zeros(10)
    
    mean_param[0] = np.mean(vec_N0); std_param[0] = np.std(vec_N0)
    mean_param[1] = np.mean(vec_j); std_param[1] = np.std(vec_j)
    mean_param[2] = np.mean(vec_i0); std_param[2] = np.std(vec_i0)
    mean_param[3] = np.mean(vec_i1); std_param[3] = np.std(vec_i1)
    mean_param[4] = np.mean(vec_beta); std_param[4] = np.std(vec_beta)
    mean_param[5] = np.mean(vec_h0); std_param[5] = np.std(vec_h0)
    mean_param[6] = np.mean(vec_h1); std_param[6] = np.std(vec_h1)
    mean_param[7] = np.mean(vec_gamma); std_param[7] = np.std(vec_gamma)
    mean_param[8] = np.mean(vec_c1); std_param[8] = np.std(vec_c1)
    mean_param[9] = np.mean(vec_d1); std_param[9] = np.std(vec_d1)
    
    return mean_param, std_param

def moyenne_param2(opti_param):
    nbr_param = len(opti_param)
    vec_c2 = np.zeros(nbr_param)
    vec_d2 = np.zeros(nbr_param)
    vec_c3 = np.zeros(nbr_param)
    vec_d3 = np.zeros(nbr_param)
    
    for i in range(0,nbr_param):
        c = np.array(opti_param[i]["c"])
        d = np.array(opti_param[i]["d"])

        vec_c2[i]= c[0]
        vec_c3[i] = c[1]
        vec_d2[i] = d[0]
        vec_d3[i] = d[1]

    
    mean_c2 = np.array([np.mean(vec_c2), np.std(vec_c2)])
    mean_c3 = np.array([np.mean(vec_c3), np.std(vec_c3)])
    mean_d2 = np.array([np.mean(vec_d2), np.std(vec_d2)])
    mean_d3 = np.array([np.mean(vec_d3), np.std(vec_d3)])
    
    return mean_c2, mean_c3, mean_d2, mean_d3

#################### Partie gestion de DATA ####################

df_hosp = pd.read_csv('hosp_data2.csv',sep=';', on_bad_lines='skip')
df_cases = pd.read_csv('case_data.csv',sep=';', on_bad_lines='skip')

# On récupère le data qui nous interesse
data_hosp = df_hosp['NEW_IN']
data_newcases = df_cases['CASES']

#On rend le data smooth
s_data_newcases = smooth_data(data_newcases)
s_data_i = s_data_newcases[0:281]
s_data_hosp = smooth_data(data_hosp)
s_data_fw = s_data_hosp[0:108] #107 jours entre le 1 mars compris et le 15 juin compris
s_data_test = s_data_hosp[0:281] #281 jours entre le 1 mars compris et le 6 décembre


#t = np.linspace(0,803,803)
t = np.linspace(0,281,281)
plt.title("Real Data vs Smooth Data")
plt.plot(t, data_hosp[0:281],label="Real Data")
plt.plot(t, s_data_test, label="Smooth Data")
plt.legend()
plt.xlabel('Days')  
plt.ylabel('Number of hospitalization')
plt.show()

#################### Premiere étape gridsearch ####################

#On définit la grille de recherche pour le gridsearch
parameters = {'N0':[5,31,57,84,110,137,163,190], #8
              'j':[1,2,3,4,5,6,7,8,9,10], #10
              'i0':[5], #1
              'i1':[7,9,11,13,15,17], #6
              'beta':[0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0], #10
              'h0':[5,7,9,11], #4
              'h1':[13,15,17,19,21], # 5
              'gamma':[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01], #10
              'c1':[0.03,0.05266123,0.09244017,0.16226711], #4
              'd1':[10, 14, 18, 22]}#4


start = time.time()
best300 =gridsearch(parameters,s_data_fw)
end = time.time()
print(time.strftime('%H:%M:%S', time.gmtime(end - start)))

'''
fichier = open("best300.txt", "r")
best300 = list(eval(fichier.read()))
fichier.close()
'''
#graphe_beta,graphe_gamma,graphe_bgk,errorfw = print_paramDistrib(best300,s_data_fw)# on print la distribution des parametres

meanH_resultfw,stdH_resultfw,meanI_resultfw,stdI_resultfw, R0mean, R0std = fprint(best300, s_data_fw,True)
mean_errorfw = mean_absolute_error(s_data_fw, meanH_resultfw)

t_fw = np.linspace(0,len(s_data_fw),len(s_data_fw))

plt.title("Evolution of the first wave")
plt.plot(t_fw, s_data_fw,label="data")
plt.plot(t_fw, meanH_resultfw, label="estimation")
plt.fill_between(t_fw,meanH_resultfw-stdH_resultfw,meanH_resultfw+stdH_resultfw,alpha=.2,label="standard deviation")
plt.legend()
plt.xlabel('Days')
plt.ylabel('Number of new hospitalization')
plt.show()

plt.title("Evolution of the first wave")
plt.plot(t_fw, s_data_fw,label="data")
plt.plot(t_fw, meanI_resultfw, label="estimation")
plt.fill_between(t_fw,meanI_resultfw-stdI_resultfw,meanI_resultfw+stdI_resultfw,alpha=.2,label="standard deviation")
plt.legend()
plt.xlabel('Days')
plt.ylabel('Number of new Infection')
plt.show()

t_test = np.linspace(0,len(s_data_test),len(s_data_test))
plt.title("Evolution of the reproduction number")
plt.plot(t_fw, R0mean,label="data")
plt.fill_between(t_fw,R0mean-R0std,R0mean+R0std,alpha=.2,label="standard deviation")
plt.legend()
plt.xlabel('Days')
plt.ylabel('Reproduction Number')
plt.show()


#################### On trouve les points de changement ####################
nbrCD = 3 #nombre de point de change point qu'on veut en comptant deja d1 !!! si on veut d1,d2,d3 -> nbrCD = 3


start = time.time()
opti_param2 = finalGridSearch(best300,s_data_test,nbrCD)
end = time.time()
print(time.strftime('%H:%M:%S', time.gmtime(end - start)))

'''
fichier = open("best6000best.txt", "r")
opti_param = list(eval(fichier.read()))
fichier.close()
'''

meanH_result,stdH_result,meanI_result,stdI_result, R0mean_result, R0std_result = fprint(opti_param2, s_data_test,False)
errorfinal = mean_absolute_error(s_data_test, meanH_result)


t_test = np.linspace(0,len(s_data_test),len(s_data_test))
plt.title("Evolution of the desease")
plt.plot(t_test, s_data_test,label="data")
plt.plot(t_test, meanH_result, label="estimation")
plt.fill_between(t_test,meanH_result-stdH_result,meanH_result+stdH_result,alpha=.2,label="standard deviation")
plt.legend()
plt.xlabel('Days')
plt.ylabel('Number of hospitalization')
plt.show()

t_test = np.linspace(0,len(s_data_test),len(s_data_test))
plt.title("Evolution of the desease")
plt.plot(t_test, s_data_i,label="data")
plt.plot(t_test, meanI_result, label="estimation")
plt.fill_between(t_test,meanI_result-stdI_result,meanI_result+stdI_result,alpha=.2,label="standard deviation")
plt.legend()
plt.xlabel('Days')
plt.ylabel('Number of new infection')
plt.show()

t_test = np.linspace(0,len(s_data_test),len(s_data_test))
plt.title("Evolution of the reproduction number")
plt.plot(t_test, R0mean_result,label="data")
plt.fill_between(t_test,R0mean_result-R0std_result,R0mean_result+R0std_result,alpha=.2,label="standard deviation")
plt.legend()
plt.xlabel('Days')
plt.ylabel('Reproduction Number')
plt.show()

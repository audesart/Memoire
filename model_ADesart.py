# -*- coding: utf-8 -*-
"""
@author: Desart Aurélien

"""


from scipy.integrate import odeint
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

def cont_model(y, t,  beta, lambd, gamma, delta1, delta2, tau, c1, d1):
        
    dy = np.zeros(y.shape)
    S = y[0]
    E = y[1]
    I = y[2]
    H = y[3]
    R = y[4]
    D = y[5]
    
    if t < d1:
        coeff = 1
    else: 
        coeff = c1
    
    dy[0] = -coeff * S * beta * (I/ N)
    dy[1] = coeff * S * beta * (I/ N) - lambd* E
    dy[2] = lambd*E - delta1*I - gamma*I
    dy[3] = gamma*I - delta2*H - tau*H
    dy[4] = delta1*I + delta2 * H
    dy[5] = tau * H
	
    return dy


df_hosp = pd.read_csv('hosp_data2.csv',sep=';', on_bad_lines='skip')
df_cases = pd.read_csv('case_data.csv',sep=';', on_bad_lines='skip')
df_mort = pd.read_csv('data_mort2.csv',sep=';', on_bad_lines='skip') #Données d'hospitalisations journalières

# On récupère le data qui nous interesse
data_hosp = df_hosp['NEW_IN']
data_newcases = df_cases['CASES']
data_mort= df_mort['Total général']

#On rend le data smooth
s_data_newcases = smooth_data(data_newcases)
s_data_i = s_data_newcases[0:281]

s_data_hosp = smooth_data(data_hosp)
s_data_fw = s_data_hosp[0:108] #107 jours entre le 1 mars compris et le 15 juin compris
s_data_test = s_data_hosp[0:281] #281 jours entre le 1 mars compris et le 6 décembre

mort_in_hosp = 0.43
s_data_mort = smooth_data(data_mort)*mort_in_hosp
s_data_mortfw = s_data_mort[0:108]


N = 11520654
y0 = np.zeros(6)
I0 = 626
y0[0] = N - I0
y0[1] = I0

t0 = 0; tend = len(s_data_fw); tend2 = len(s_data_test)
t = np.arange(t0, tend, 1) # Simulating over t_end - t0 time points  
t2 = np.arange(t0, tend2, 1)

beta = 1.7
lambd = 0.172
gamma = 0.0454
delta1 = 0.453
delta2 = 0.587
tau = 0.108
c1 = 0.184
d1 = 22

sol = odeint(cont_model,y0,t,args=(beta, lambd, gamma, delta1, delta2, tau, c1, d1))
resH = sol[:,3]

plt.title("Evolution of the first wave")
plt.plot(t, s_data_fw,label="Data")
plt.plot(t, resH, label="Mean estimation")
plt.legend()
plt.xlabel('Days')  
plt.ylabel('Number of new hospitalization')
plt.show()

parameters = {'beta':[0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ], #10
              'lambda':[0.1       , 0.14444444, 0.18888889, 0.23333333, 0.27777778,
                     0.32222222, 0.36666667, 0.41111111, 0.45555556, 0.5], #10
              'gamma':[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1], #10
              'delta1':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.], # 10
              'delta2':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.], #10
              'tau': [0.01      , 0.03111111, 0.05222222, 0.07333333, 0.09444444,
                     0.11555556, 0.13666667, 0.15777778, 0.17888889, 0.2       ], #10
              'c1':[0.03      , 0.05266123, 0.09244017, 0.16226711, 0.28483953,
                     0.5  ], #6
              'd1': [10, 14, 18, 22,26,30,34,38,42,46,50,54,58]}#13



def gridsearch(parameters,dataH, dataD):
    nbrparam = len(list(ParameterGrid(parameters)))
    error=np.zeros((nbrparam,2))
    index=0
    for param in ParameterGrid(parameters):
        beta = param["beta"]
        lambd = param["lambda"]
        gamma = param["gamma"]
        delta1 = param["delta1"]
        delta2 = param["delta2"]
        tau = param["tau"]
        c1 = param["c1"]
        d1 = param["d1"]
        sol = odeint(cont_model,y0,t,args=(beta, lambd, gamma, delta1, delta2, tau, c1, d1))
        
        resH = sol[:,3]
        
        error[index] = (mean_absolute_error(dataH, resH),index)
        index = index+1
        if (index%100000 == 0):
            print(index)
        
    best_results = error[error[:,0].argsort()]
    best_estimators = best_results[0:300]
    index_best_estimators = best_estimators[:,1]
    best300 = xbestparameters(index_best_estimators, list(ParameterGrid(parameters)),300)
    return best300

def xbestparameters(index_best_estimators,base,nbrbest):
    result = nbrbest * [0] #initilisation comme ça car liste de dic
    index = 0
    for i in index_best_estimators:
       result[index] = base[int(i)]
       index = index + 1
    return result
    

best300 = gridsearch(parameters,s_data_fw, s_data_mortfw)

def fprint(opti_param,data,fw):
    data_size=len(data)
    nbr_param = len(opti_param)
    resultH = np.zeros((nbr_param,data_size))
    rv = np.zeros((nbr_param,data_size))
    for i in range(0,nbr_param):
        beta = opti_param[i]["beta"]
        lambd = opti_param[i]["lambda"]
        gamma = opti_param[i]["gamma"]
        delta1 = opti_param[i]["delta1"]
        delta2 = opti_param[i]["delta2"]
        tau = opti_param[i]["tau"]
        c1 = opti_param[i]["c1"]
        d1 = opti_param[i]["d1"]
        if fw:
            sol = odeint(cont_model,y0,t,args=(beta, lambd, gamma, delta1, delta2, tau, c1, d1))
            r = R0(beta,delta1,gamma,c1,d1,[],[],len(data))
        else: 
            c = opti_param[i]["c"]
            d = opti_param[i]["d"]
            sol = odeint(cont_model2,y0,t2,args=(beta, lambd, gamma, delta1, delta2, tau, c1, d1, c[0], d[0], c[1], d[1]))
            r = R0(beta,delta1,gamma,c1,d1,c,d,len(data))
        
        resH = sol[:,3]
        resultH[i] = smooth_data(resH)
        rv[i] = r
        
    resultH_t = np.transpose(resultH)
    rt = np.transpose(rv)
    meanH_result = np.zeros(data_size)
    stdH_result = np.zeros(data_size)
    rmean = np.zeros(data_size)
    rstd = np.zeros(data_size)
    for i in range(0,data_size):
        meanH_result[i] = np.mean(resultH_t[i])
        stdH_result[i] = np.std(resultH_t[i])
        rmean[i] = np.mean(rt[i])
        rstd[i] = np.std(rt[i])
  
    return meanH_result,stdH_result, rmean, rstd

meanH_resultfw,stdH_resultfw, R0fw_mean,R0fw_std  = fprint(best300, s_data_fw, True)
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



def moyenne_param1(opti_param):
    nbr_param = len(opti_param)
    vec_beta = np.zeros(nbr_param)
    vec_lambda = np.zeros(nbr_param)
    vec_gamma = np.zeros(nbr_param)
    vec_delta1 = np.zeros(nbr_param)
    vec_delta2 = np.zeros(nbr_param)
    vec_tau = np.zeros(nbr_param)
    vec_c1 = np.zeros(nbr_param)
    vec_d1 = np.zeros(nbr_param)
    
    for i in range(0,nbr_param):
        vec_beta[i] = opti_param[i]["beta"]
        vec_lambda[i] = opti_param[i]["lambda"]
        vec_gamma[i] = opti_param[i]["gamma"]
        vec_delta1[i] = opti_param[i]["delta1"]
        vec_delta2[i] = opti_param[i]["delta2"]
        vec_tau[i] = opti_param[i]["tau"]
        vec_c1[i] = opti_param[i]["c1"]
        vec_d1[i] = opti_param[i]["d1"]
        
    mean_param = np.zeros(8)
    std_param = np.zeros(8)
    
    mean_param[0] = np.mean(vec_beta); std_param[0] = np.std(vec_beta)
    mean_param[1] = np.mean(vec_lambda); std_param[1] = np.std(vec_lambda)
    mean_param[2] = np.mean(vec_gamma); std_param[2] = np.std(vec_gamma)
    mean_param[3] = np.mean(vec_delta1); std_param[3] = np.std(vec_delta1)
    mean_param[4] = np.mean(vec_delta2); std_param[4] = np.std(vec_delta2)
    mean_param[5] = np.mean(vec_tau); std_param[5] = np.std(vec_tau)
    mean_param[6] = np.mean(vec_c1); std_param[6] = np.std(vec_c1)
    mean_param[7] = np.mean(vec_d1); std_param[7] = np.std(vec_d1)
    
    return mean_param, std_param

meanparam1, stdparam2 = moyenne_param1(best300)


def cont_model2(y, t, beta, lambd, gamma, delta1, delta2, tau, c1, d1, c2, d2, c3, d3):

    if t < d1:
        coeff = 1
    elif t >= d1 and t< d2:
        coeff = c1
    elif t >= d2 and t< d3:
        coeff = c2
    else: 
        coeff = c3
        
    dy = np.zeros(y.shape)
    S = y[0]
    E = y[1]
    I = y[2]
    H = y[3]
    R = y[4]
    D = y[5]
    
    dy[0] = -coeff * S * beta * (I/ N)
    dy[1] = coeff * S * beta * (I/ N) - lambd* E
    dy[2] = lambd*E - delta1*I - gamma*I
    dy[3] = gamma*I - delta2*H - tau*H
    dy[4] = delta1*I + delta2 * H
    dy[5] = tau * H
	
    return dy


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

def iter_FGS(param,data,nbrd):
    index = 0
    c1 = param["c1"]
    newPGrid = newcd(c1,nbrd)
    nbrparam = len(list(ParameterGrid(newPGrid)))
    error= np.zeros((nbrparam,2))
    for elem in ParameterGrid(newPGrid):
        c,d = getCD(elem)
        if is_sorted_asc(d):
            beta = param["beta"]
            lambd = param["lambda"]
            gamma = param["gamma"]
            delta1 = param["delta1"]
            delta2 = param["delta2"]
            tau = param["tau"]
            c1 = param["c1"]
            d1 = param["d1"]
            sol = odeint(cont_model2,y0,t2,args=(beta, lambd, gamma, delta1, delta2, tau, c1, d1, c[0], d[0], c[1], d[1]))
            resH = sol[:,3]
            error[index] = (mean_absolute_error(data, resH),index)
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

def newcd(c1,nbrd):
    dict={}
    for i in range(2,nbrd+1):
        keyc=str("c"+str(i))
        keyd=str("d"+str(i))
        if i%2 == 1:
            dict[keyc] = np.geomspace(0.5*c1,2*c1,8).tolist()
            dict[keyd] = np.linspace(215,245,6).tolist()        
        else:
            dict[keyc] = np.geomspace(2*c1,15 *c1,8).tolist()
            dict[keyd] = np.linspace(160,190,6).tolist()
        


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

def reunite(best20,elem,nbrd):
    size = nbrd-1
    for i in range(0,20):
        best20[i]["beta"] = elem["beta"]
        best20[i]["lambda"] = elem["lambda"]
        best20[i]["gamma"] = elem["gamma"]
        best20[i]["delta1"] = elem["delta1"]
        best20[i]["delta2"] = elem["delta2"]
        best20[i]["tau"] = elem["tau"]
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

def R0(beta,delta1,gamma,c1,d1,c,d,tend):
    k = beta/(delta1+gamma)
    coeff = np.zeros(tend)
    nbrpoint = len(d)
    coeff[0:int(d1)] = 1
    
    if (nbrpoint == 0):
        coeff[int(d1):tend] = c1
    else:
        coeff[int(d1):int(d[0])] = c1
        for i in range(0,nbrpoint-1):
            coeff[int(d[i]):int(d[i+1])] = c[i]
        coeff[int(d[nbrpoint-1]):tend] = c[nbrpoint-1]
    
    return coeff * k
    
    
nbrCD = 3#nombre de point de change point qu'on veut en comptant deja d1 !!! si on veut d1,d2,d3 -> nbrCD = 3

opti_param2 = finalGridSearch(best300,s_data_test,nbrCD)


meanH_result,stdH_result, meanR0_result, stdR0_result = fprint(opti_param2, s_data_test,False)
errorfinal = mean_absolute_error(s_data_test, meanH_result)

plt.title("Evolution of the desease")
plt.plot(t2, s_data_test,label="data")
plt.plot(t2, meanH_result, label="estimation")
plt.fill_between(t2,meanH_result-stdH_result,meanH_result+stdH_result,alpha=.2,label="standard deviation")
plt.legend()
plt.xlabel('Days')
plt.ylabel('Number of hospitalisation')
plt.show()

plt.title("Evolution of the reproduction number")
plt.plot(t2, meanR0_result, label="estimation")
plt.fill_between(t2,meanR0_result-stdR0_result,meanR0_result+stdR0_result,alpha=.2,label="standard deviation")
plt.legend()
plt.xlabel('Days')
plt.ylabel('Reproduction Number')
plt.show()


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

mean_c2, mean_c3, mean_d2, mean_d3 = moyenne_param2(opti_param2)


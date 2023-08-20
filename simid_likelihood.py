# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 19:48:42 2023

@author: aure0
"""

from math import * # useful math functions
import numpy as np # useful array objects
import scipy.stats as stats #useful statistics functions
import copy #allows for deep copies of objec
from scipy.integrate import odeint as ode # ode solver
import matplotlib.pyplot as plt # nice plotting commands, very similar to Matlab commands
import pandas as pd
from scipy.stats import poisson
from scipy.stats import norm

import scipy.optimize as optimize


def contact_matrice(alpha,beta,rho,q_sym):
    C_asym = np.array(pd.read_csv('Simid/data/C_asym.csv',sep=',', on_bad_lines='skip'))
    C_home = np.array(pd.read_csv('Simid/data/C_home.csv',sep=',', on_bad_lines='skip'))
    C_work = np.array(pd.read_csv('Simid/data/C_work.csv',sep=',', on_bad_lines='skip'))
    C_school = np.array(pd.read_csv('Simid/data/C_school.csv',sep=',', on_bad_lines='skip'))
    C_transport = np.array(pd.read_csv('Simid/data/C_transport.csv',sep=',', on_bad_lines='skip'))
    C_leisure = np.array(pd.read_csv('Simid/data/C_leisure.csv',sep=',', on_bad_lines='skip'))
    C_other = np.array(pd.read_csv('Simid/data/C_other.csv',sep=',', on_bad_lines='skip'))

    C_asym = C_home + alpha*C_work + beta*C_school + alpha*C_transport + rho*C_leisure + rho*C_other
    C_sym = C_home + alpha*0.09*C_work + beta*0.09*C_school + alpha*0.13*C_transport + rho*0.06*C_leisure + rho*0.25*C_other

    r_beta = 0.51
    
    q_asym = r_beta * q_sym
    
    beta_sym = q_sym * C_sym
    beta_asym = q_asym * C_asym

    
    return beta_sym, beta_asym

def model(ini, time_step, params):
    
    dy = np.zeros(100) #column vector for the state variables
    X = ini
    
    gamma = params[0]; theta = params[1]; delta1 = params[2]; delta2_star = params[3]
    delta3_star = params[4]; beta1_star = params[5]; R0 = params[6]
    
    o1 = params[7]; o2 = params[8]; o3 = params[9]; o4 = params[10]; o5 = params[11]
    o6 = params[12]; o7 = params[13]; o8 = params[14]; o9 = params[15]; o10 = params[16]

    phi01 = params[17]; phi02 = params[18]; phi03 = params[19]; phi04 = params[20]; phi05 = params[21]
    phi06 = params[22]; phi07 = params[23]; phi08 = params[24]; phi09 = params[25]; phi010 = params[26]
    
    mu2 = params[27]; mu4 = params[28]; mu5 = params[29]; mu6 = params[30]
    mu7 = params[31]; mu8 = params[32]; mu9 = params[33]; mu10 = params[34]
        
    #on transforme certains param en vecteurs
    phi_0 = np.zeros(10)
    phi_0[0] = phi01; phi_0[1] = phi02; phi_0[2] = phi03; phi_0[3] = phi04; phi_0[4] = phi05
    phi_0[5] = phi06; phi_0[6] = phi07; phi_0[7] = phi08; phi_0[8] = phi09; phi_0[9] = phi010
    
    mu = np.zeros(10)
    mu[0] = 0 #assumptions
    mu[1] = mu2; mu[2] = mu2 #assumptions
    mu[3] = mu4; mu[4] = mu5; mu[5] = mu6; mu[6] = mu7; mu[7] = mu8; mu[8] = mu9; mu[9] = mu10; 
    
    omega = np.zeros(10)
    omega[0] = o1; omega[1] = o2; omega[2] = o3; omega[3] = o4; omega[4] = o5 
    omega[5] = o6; omega[6] = o7; omega[7] = o8; omega[8] = o9; omega[9] = o10
    
    
    #on derive les parametres restants 
    delta_1 = delta1
    delta_2 = delta2_star * phi_0
    delta_3 = delta3_star * (1-mu)
    delta_4 = delta_3 #assumptions
    
    tau_1 = mu*delta3_star
    tau_2 = tau_1 #assumptions
    
    psi = (1-phi_0)* delta2_star
    
    phi_1 = 0.75
    
    q_sym = 4.4651e-08

    
    #Construction contact matrice
    alpha = 1; beta = 1; rho=1
    beta_sym, beta_asym = contact_matrice(alpha,beta,rho,q_sym)
    
    S = X[0:10]
    E = X[10:20]
    I_presym = X[20:30]
    I_asym = X[30:40]
    I_mild = X[40:50]
    I_sev = X[50:60]
    I_hosp = X[60:70]
    I_icu = X[70:80]
    D = X[80:90]
    R = X[90:100]      
        
    lambda_t = np.dot(beta_asym, I_presym + I_asym) + np.dot(beta_sym, I_mild + I_sev)            
    
    dy[0:10] = -lambda_t * S
    dy[10:20] = lambda_t * S - gamma*E
    dy[20:30] = gamma*E - theta*I_presym
    dy[30:40] = theta*p*I_presym - delta_1 * I_asym
    dy[40:50] = theta*(1-p)*I_presym - (psi+delta_2) * I_mild
    dy[50:60] = psi*I_mild - omega*I_sev
    dy[60:70] = phi_1*omega*I_sev - (delta_3 + tau_1) * I_hosp
    dy[70:80] = (1-phi_1)*omega*I_sev - (delta_4 + tau_2) * I_icu
    dy[80:90] = tau_1 * I_hosp + tau_2 * I_icu
    dy[90:100] = delta_1 * I_asym + delta_2 * I_mild + delta_3 * I_hosp + delta_4 * I_icu
    return dy




#### Load Data ####
df_hosp = pd.read_csv('hosp_data2.csv',sep=';', on_bad_lines='skip') #Données d'hospitalisations journalières
data_fw = np.array(df_hosp['NEW_IN'][0:21]) 

t0 = 0; t_end = 21 # du 1er au 22 mars
times = np.arange(t0, t_end, 1) # Simulating over t_end - t0 time points
#DEFINE THE PARAMETER SET

p= np.array([0.94, 0.9, 0.84, 0.61, 0.49, 0.21, 0.02, 0.02, 0.02, 0.02])


gamma = 0.5
theta = 1/3.2
delta1 = 1/3.5
delta2_star = 1/3.5
delta3_star = 1/12
phi0 = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]
beta1_star = 1.404
omega = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3] 
mu = [0.001,0.01,0.02,0.05,0.2,0.3,0.4,0.6]
R0 = 3


[o1, o2, o3, o4, o5, o6, o7, o8, o9, o10] = omega
[phi01, phi02, phi03, phi04, phi05, phi06, phi07, phi08, phi09, phi010] = phi0
[mu2, mu4, mu5, mu6, mu7, mu8, mu9, mu10] = mu 

params = np.array([gamma, theta, delta1, delta2_star, delta3_star, beta1_star, R0,
          o1, o2, o3, o4, o5, o6, o7, o8, o9, o10,
          phi01, phi02, phi03, phi04, phi05, phi06, phi07, phi08, phi09, phi010,
          mu2, mu4, mu5, mu6, mu7, mu8, mu9, mu10])

def x0fcn():
     
    age2020 = np.array([1273745, 1304811, 1416753, 1499416, 1507429, 1591660, 1347513, 924351, 539381, 115595])
    n0 = np.array([4,7,27,34,47,42,27,29,22,8])
    I0 = n0*(1/(1-p))

    y0 = np.zeros(100)
    y0[0:10] = age2020 - I0
    y0[20:30] = I0
    X0 = y0.tolist()
    return X0

ini = x0fcn()

def yfcn(res):
    
     hosp_model = np.zeros(np.size(times))
         
     for i in range(20):
         hosp_model = hosp_model + res[:,60+i]
     return hosp_model

#### Simulate the model ####

res = ode(model, ini, times, args=(params,))

sim_measure = yfcn(res)


plt.plot(times, sim_measure, 'b-', linewidth=3, label='Model simulation')
plt.plot(times, data_fw, 'ko', linewidth=2, label='Data')
plt.xlabel('Time')
plt.ylabel('Individuals')
plt.legend()
plt.show()




def NLL(params, data, times):
     
     
    params = np.abs(params)
    
    res = ode(model, x0fcn(), times, args=(params,))
    
    y = yfcn(res) 
    
    #Calculate the NLL for Poisson distribution
    nll = sum(y) - sum(data*np.log(y)) 
    #nll = -sum(np.log(poisson.pmf(np.round(data),np.round(y))))
    
    
    return nll

optimizer = optimize.minimize(NLL, params, args=(data_fw, times), method='Nelder-Mead' ,options={'maxiter': 100}, bounds = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,2),(0,4),
                                                                                                 (0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),
                                                                                                 (0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),
                                                                                                 (0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)) )
                                                                                               

paramests = np.abs(optimizer.x)

iniests = x0fcn()

xest = ode(model, iniests, times, args=(paramests,))

est_measure = yfcn(xest) 


plt.plot(times, est_measure, 'b-', linewidth=3, label='Model simulation')
plt.plot(times, data_fw, 'ko', linewidth=2, label='Data')
plt.xlabel('Time')
plt.ylabel('Individuals')
plt.legend()
plt.show()

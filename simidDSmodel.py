# -*- coding: utf-8 -*-
"""
@author: Aurelien Desart

"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import binom
import pandas as pd

#construction intervention contact matrice

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

def DSmodel(y0, params):

    #################### On extrait les parametres de params #########################
    
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
    delta2 = delta2_star * phi_0
    delta3 = delta3_star * (1-mu)
    delta4 = delta3 #assumptions
    
    tau1 = mu*delta3_star
    tau2 = tau1 #assumptions
    
    psi = (1-phi_0)* delta2_star
    
    phi1 = 0.75
    
    
    #Initialisation vecteurs systeme
    S = np.zeros((tend+1,divage))
    E = np.zeros((tend+1,divage))
    I_presym = np.zeros((tend+1,divage))
    I_asym = np.zeros((tend+1,divage))
    I_mild = np.zeros((tend+1,divage))
    I_sev = np.zeros((tend+1,divage))
    I_hosp = np.zeros((tend+1,divage))
    I_icu = np.zeros((tend+1,divage))
    R = np.zeros((tend+1,divage))
    D = np.zeros((tend+1,divage))
    
    S[0] = y0[0:10]
    E[0] = y0[10:20]
    I_presym[0] = y0[20:30]
    I_asym[0] = y0[30:40]
    I_mild[0] = y0[40:50]
    I_sev[0] = y0[50:60]
    I_hosp[0] = y0[60:70]
    I_icu[0] = y0[70:80]
    R[0] = y0[80:90]
    D[0] = y0[90:100]
    
    #System
    pstar = np.zeros(divage);
    NE = np.zeros(divage)
    NI_presym = np.zeros(divage)
    NI_asym = np.zeros(divage)
    NI_mild = np.zeros(divage)
    NI_sev = np.zeros(divage)
    NI_hosp = np.zeros(divage)
    NI_icu = np.zeros(divage)
    ND_hosp = np.zeros(divage)
    ND_icu = np.zeros(divage)
    NR_asym= np.zeros(divage)
    NR_mild = np.zeros(divage)
    NR_hosp = np.zeros(divage)
    NR_icu = np.zeros(divage)
    
    
    for t in range(0,tend):
        if t<tI:
            alpha = 1; beta = 1; rho=1
        else: 
            alpha = 0.2; beta=0; rho=0.1
            
            
        beta_sym, beta_asym = contact_matrice(alpha,beta,rho,q_sym)
            
        lambda_t = np.dot(beta_asym, I_presym[t] + I_asym[t]) + np.dot(beta_sym, I_mild[t] + I_sev[t])
        pstar =  1 - np.exp(-h* lambda_t)
        for age in range(10):
            NE[age] = np.random.binomial(S[t][age], pstar[age])
            NI_presym[age] = np.random.binomial(E[t][age], 1 - np.exp(-h*gamma))
            NI_asym[age] = np.random.binomial(I_presym[t][age], 1 - np.exp(-h*p[age]*theta))
            NI_mild[age] = np.random.binomial(I_presym[t][age],  1 - np.exp(-h*(1-p[age])*theta))
            NI_sev[age] = np.random.binomial(I_mild[t][age], 1 - np.exp(-h*psi[age]))
            NI_hosp[age] = np.random.binomial(I_sev[t][age], 1 - np.exp(-h*phi1*omega[age]))
            NI_icu[age] = np.random.binomial(I_sev[t][age], 1 - np.exp(-h*(1-phi1)*omega[age]))
            ND_hosp[age] = np.random.binomial(I_hosp[t][age], 1 - np.exp(-h*tau1[age]))
            ND_icu[age] = np.random.binomial(I_icu[t][age], 1 - np.exp(-h*tau2[age]))
            NR_asym[age] = np.random.binomial(I_asym[t][age], 1 - np.exp(-h*delta1))
            NR_mild[age] = np.random.binomial(I_mild[t][age], 1 - np.exp(-h*delta2[age]))
            NR_hosp[age] = np.random.binomial(I_hosp[t][age], 1 - np.exp(-h*delta3[age]))
            NR_icu[age] = np.random.binomial(I_icu[t][age], 1 - np.exp(-h*delta4[age]))
        
        S[t+1] = S[t] - NE
        E[t+1] = E[t] + NE - NI_presym 
        I_presym[t+1] = I_presym[t] + NI_presym - NI_asym - NI_mild
        I_asym[t+1] = I_asym[t] + NI_asym - NR_asym
        I_mild[t+1] = I_mild[t] + NI_mild - NI_sev - NR_mild
        I_sev[t+1] = I_sev[t] + NI_sev - NI_hosp - NI_icu
        I_hosp[t+1] = I_hosp[t] + NI_hosp - ND_hosp - NR_hosp
        I_icu[t+1] = I_icu[t] + NI_icu - ND_icu - NR_icu
        D[t+1] = D[t] + ND_hosp + ND_icu
        R[t+1] = R[t] + NR_asym + NR_mild + NR_hosp + NR_icu
        
        
    ipresym, iasym, isev, hosp, icu, death = sum_h(I_presym[0:tend],I_asym[0:tend],I_sev[0:tend],I_hosp[0:tend], I_icu[0:tend], D[0:tend])
 
    return ipresym, iasym, isev, hosp, icu, death


def sum_h(I_presym, I_asym, I_sev, I_hosp, I_icu, Death):
    lend = len(data_hosp_tot)
    ipresym = np.zeros((lend,10))
    iasym = np.zeros((lend,10))
    isev = np.zeros((lend,10))
    hosp = np.zeros((lend,10))
    icu = np.zeros((lend,10))
    death = np.zeros((lend,10))
    step = int(1/h)

    for i in range(lend):
        for k in range(10):
            ipresym[i][k] = sum(I_presym[i*step:(i+1)*step,k])
            iasym[i][k] = sum(I_asym[i*step:(i+1)*step,k])
            isev[i][k] = sum(I_sev[i*step:(i+1)*step,k])
            hosp[i][k] = sum(I_hosp[i*step:(i+1)*step,k])
            icu[i][k] = sum(I_icu[i*step:(i+1)*step,k])
            death[i][k] = sum(Death[i*step:(i+1)*step,k])
    return ipresym, iasym, isev, hosp, icu, death
    
def get_totage_data(I_hosp, I_icu,Death):
    hosp_model = np.zeros(len(data_hosp_tot))
    mort = np.zeros(len(data_hosp_tot))
    for i in range(10):
        hosp_model = hosp_model + I_hosp[:,i] + I_icu[:,i]
        mort = mort + Death[:,i]
    return hosp_model,mort
    
def get_newdeath(death):
    
    len_death = len(death)
    new_death = np.zeros(len_death-1)
    new_death[0] = death[0]
    for i in range(1,len_death-1):
        new_death[i] = death[i+1] - death[i]
    return new_death

def delta_matrix(M,V):
    len_vec = len(V)
    result = np.zeros(np.shape(M))
    for i in range(len_vec):
        result[i] = M[i]* V[i]
    return result

def vdotM(M,V):
    len_vec = len(V)
    result = np.zeros(np.shape(M))
    for i in range(len_vec):
        result[i] = M[:,i]* V[i]
    return result

def get_R0(alpha,beta,rho, params):
    #Parameters maladie
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
    delta2 = delta2_star * phi_0
    delta3 = delta3_star * (1-mu)
    delta4 = delta3 #assumptions
    
    tau1 = mu*delta3_star
    tau2 = tau1 #assumptions
    
    psi = (1-phi_0)* delta2_star
    
    phi1 = 0.75
    
    beta_sym, beta_asym = contact_matrice(alpha,beta,rho,q_sym)
    
    f_asym = delta_matrix(beta_asym,age2020)
    f_sym = delta_matrix(beta_sym,age2020)
    
    result = (f_asym/theta) + vdotM(f_asym,p) + vdotM(f_sym,(1-p)/(psi+delta2)) + vdotM(f_sym,((1-phi_0)*(1-p))/omega)
    
    eig_value, eig_vector = np.linalg.eig(result)
    
    return max(eig_value)


def fprint(param,nbr_simu,data):
    data_size=len(data)
    resultH = np.zeros((nbr_simu,data_size))
    resultD = np.zeros((nbr_simu,data_size)) 
    
    for i in range(0,nbr_simu):
        I_presym, I_asym, I_sev, I_hosp, I_icu, Death = DSmodel(y0, params)
        tot_hosp, tot_death = get_totage_data(I_hosp, I_icu,Death)
        resultH[i] = tot_hosp
        resultD[i] = tot_death
        
    meanH_result = np.zeros(data_size)
    stdH_result = np.zeros(data_size)
    meanD_result = np.zeros(data_size)
    stdD_result = np.zeros(data_size)
    
    for i in range(0,data_size):
        meanH_result[i] = np.mean(resultH[:,i])
        stdH_result[i] = np.std(resultH[:,i]) 
        meanD_result[i] = np.mean(resultD[:,i])
        stdD_result[i] = np.std(resultD[:,i]) 
        
    return meanH_result,stdH_result, meanD_result,stdD_result


##################################  Constantes modèle  ##########################################
divage = 10
age2020 = np.array([1273745, 1304811, 1416753, 1499416, 1507429, 1591660, 1347513, 924351, 539381, 115595]) #repartion d'age 
p= np.array([0.94, 0.9, 0.84, 0.61, 0.49, 0.21, 0.02, 0.02, 0.02, 0.02]) # Proba d'etre asymptomatique
h = 1/24  #Pas de temps 

nbr_jour = 64 #Nombres de jours entre 1er mars et le 4 mai
t0 = 0
tend =  int(nbr_jour * (1/h)) #Nombres de pas de simulations = nbr de jours * h
#exit_vec = np.array([64,71,78,93,])
tI = int(22* (1/h)) #jour du confinement -> 04 mai

#q_sym = 0.051
#q_sym = 2e-08
q_sym = 3.0592205e-08
#q_sym = 5.01e-08


        
################################ PARTIE DATA/ TRANSFORMATION DE DONNEES ######################################

#Data hospitalisation 
df_hosp = pd.read_csv('hosp_data2.csv',sep=';', on_bad_lines='skip') #Données d'hospitalisations journalières
data_hosp_tot = np.array(df_hosp['NEW_IN'][0:nbr_jour])
distrib_hosp = pd.read_csv('agehospdata.csv',sep=';', on_bad_lines='skip') #Données d'hospitalisations journalières
data_hosp_age1 = np.array(distrib_hosp['0-10'])*data_hosp_tot
data_hosp_age2 = np.array(distrib_hosp['10-20'])*data_hosp_tot
data_hosp_age3 = np.array(distrib_hosp['20-30'])*data_hosp_tot
data_hosp_age4 = np.array(distrib_hosp['30-40'])*data_hosp_tot
data_hosp_age5 = np.array(distrib_hosp['40-50'])*data_hosp_tot
data_hosp_age6 = np.array(distrib_hosp['50-60'])*data_hosp_tot
data_hosp_age7 = np.array(distrib_hosp['60-70'])*data_hosp_tot
data_hosp_age8 = np.array(distrib_hosp['70-80'])*data_hosp_tot
data_hosp_age9 = np.array(distrib_hosp['80-90'])*data_hosp_tot
data_hosp_age10 = np.array(distrib_hosp['90+'])*data_hosp_tot


#Data mort
df_mort = pd.read_csv('data_mort2.csv',sep=';', on_bad_lines='skip') #Données d'hospitalisations journalières
data_mort_tot= df_mort['Total général'][0:nbr_jour]
mort_in_hosp = 0.43
data_mort = df_mort['Total général'][0:nbr_jour]*mort_in_hosp


#Data seropositivité
spos1 = np.array([2,7,7,10,13,16,16,12,13,4])
spos2 = np.array([4,20,23,18,25,29,16,12,21,25])
stot1 = np.array([36,294,436,461,468,498,507,506,493,211])
stot2 = np.array([85,442,375,407,406,430,426,316,315,195])

sero1_data = spos1/stot1
sero2_data = spos2/stot2

################################## Définitions des conditions initiales ##########################
n0 = np.array([4,7,27,34,47,42,27,29,22,8])
I0 = n0*(1/(1-p))
y0 = np.zeros(100)
y0[0:10] = age2020 - I0
y0[20:30] = I0 #on introduit les cas dans les cas dans I_presym



################################### Parametres test ###############################################

params = np.array([4.89055989e-01, 2.88060447e-01, 2.95604288e-01, 2.76847087e-01,
        8.63549384e-02, 1.38971901e+00, 2.96850274e+00, 2.95655809e-01,
        2.96576334e-01, 2.96447310e-01, 2.94752369e-01, 2.93341051e-01,
        2.90732037e-01, 2.95383815e-01, 2.95944817e-01, 2.97001474e-01,
        3.10060613e-01, 9.01373352e-01, 9.30027941e-01, 9.29562325e-01,
        9.45235479e-01, 9.52151163e-01, 9.56249698e-01, 9.41221852e-01,
        9.29259786e-01, 9.21524610e-01, 8.64667664e-01, 9.91723197e-04,
        9.96520470e-03, 1.98787525e-02, 4.93664368e-02, 1.99809674e-01,
        2.98634698e-01, 3.98338730e-01, 5.92613632e-01])


##################################### Obtention des résultats #######################################
I_presym, I_asym, I_sev, I_hosp, I_icu, Death = DSmodel(y0, params)
R0_model = get_R0(1,1,1,params)
meanH_result,stdH_result, meanD_result,stdD_result = fprint(params,5,data_hosp_tot)


hosp_model, death_model = get_totage_data(I_hosp, I_icu,Death)
meanH_result[0:15] = data_hosp_tot[0:15]
stdH_result[0:15] = np.zeros(15)

####################################### Plot des résultats ###########################################    
t = np.arange(t0, nbr_jour, 1/24) # Simulating over t_end - t0 time points   
t_data = np.arange(t0,nbr_jour,1) 

plt.plot(t_data, hosp_model, 'b-', linewidth=3, label='Model simulation')
plt.plot(t_data, data_hosp_tot, 'ko', linewidth=2, label='Data')
plt.xlabel('Time')
plt.ylabel('Individuals')
plt.legend()
plt.show()

t_mort = np.arange(t0,nbr_jour-1,1)
ouip = get_newdeath(death_model)
plt.plot(t_mort, ouip, 'b-', linewidth=3, label='Model simulation')
plt.plot(t_data, data_mort, 'ko', linewidth=2, label='Data')
plt.xlabel('Time')
plt.ylabel('Individuals')
plt.legend()
plt.show()


t_sero = np.arange(t0,10,1)
plt.plot(t_sero, sero1_data, 'ko', linewidth=3, label='30 March 2020',color = 'b')
plt.plot(t_sero, sero2_data, 'ko', linewidth=2, label='26 April 2020',color = 'r')
plt.xlabel('Age group')
plt.ylabel('Proportion of seropositive individuals')
plt.legend()
plt.show()


plt.title("Evolution of the first wave")
plt.plot(t_data, data_hosp_tot,label="data_hosp_tot")
plt.plot(t_data, meanH_result, label="estimation")
plt.fill_between(t_data,meanH_result-stdH_result,meanH_result+stdH_result,alpha=.2,label="standard deviation")
plt.legend()
plt.xlabel('Days')
plt.ylabel('Number of new hospitalization')
plt.show()




params = np.array([6.59163184e-01, 9.64630423e-01, 8.45459852e-01, 2.11747743e-01,
       2.30051398e-01, 3.79772647e-01, 3.94385231e+00, 0.00000000e+00,
       2.41212784e-02, 2.28322102e-01, 1.72689086e-01, 9.49541979e-02,
       4.64155012e-02, 3.37605487e-01, 1.82331798e-01, 8.80205406e-01,
       4.57957526e-01, 9.49030503e-01, 9.99990136e-01, 9.64106804e-01,
       9.94090544e-01, 9.67828689e-01, 1.49015329e-01, 9.99918460e-01,
       8.50954530e-01, 9.80895682e-01, 7.32337415e-01, 2.41321128e-03,
       2.17681424e-02, 1.26138325e-02, 3.45757379e-02, 5.05171666e-01,
       8.89718715e-01, 1.73393912e-01, 6.15856456e-02])

######################################" MCMC ####################################""

def new_theta():
    gamma = np.random.normal(0.5, pow(0.05,2))
    theta = np.random.normal(0.286, pow(0.05,2))
    delta1 = np.random.normal(0.286, pow(0.05,2))
    delta2_star = np.random.uniform(0,1)
    delta3_star = np.random.uniform(0,1)
    omega = np.random.uniform(0,1,10)
    beta1_star = np.random.uniform(0,5)
    phi0 = np.random.uniform(0,1,10)
    mu = np.random.uniform(0,1,8)
    R0 = np.random.normal(2.5, pow(0.1,2))
    
    params = np.zeros(35)
    params[0] = gamma
    params[1] = theta 
    params[2] = delta1 
    params[3] =  delta2_star
    params[4] = delta3_star
    params[5] = beta1_star
    params[6] = R0
    params[7:17] = omega
    params[17:27] = phi0
    params[27:35] = mu
    
    return params

def mcmc_hosp(Niter,theta0,k):
    ntheta = len(theta0)
    theta= np.zeros((Niter,ntheta))
    theta[0] = theta0
    ipresym_prec, iasym_prec, isev_prec, hosp_prec, icu_prec = DSmodel(y0, theta0)
    
    n = len(isev_prec[:,k])
    r= list(range(n + 1))
    
    like_prec = binom.pmf(r,isev_prec[:,k], 1-np.exp(-h*theta0[27+k]))
    #bin_prec = np.random.binomial(isev_prec[:,k], 1-np.exp(-h*theta0[27+k])) 
    post_prec = like_prec * theta0[27+k]
    for i in range(1,Niter): 
        theta_pred = theta[i-1]
        theta_new = new_theta()
        ipresym_new, iasym_new, isev_new, hosp_new, icu_new = DSmodel(y0, theta_new)
        
        
        Yt = binom.pmf(r,isev_prec[:,k], 1-np.exp(-h*theta0[27+k]))
        #Yt = np.random.binomial(isev_new[:,k], 1-np.exp(-h*omega[k])) #Likelihood
        
        post_new = Yt * theta_new[27+k]
        
        r = post_new/post_prec
        
        alpha = min(1,r)
        
        u = np.random.uniform(0,1)
        
        if u < alpha:
            theta[i] = theta_new
            like_prec = Yt
            post_prec = post_new
        else: 
            theta[i] = theta_pred
            
    return np.mean(theta,0), np.std(theta,0,ddof = 1)
        


theta0 = np.array([6.59163184e-01, 9.64630423e-01, 8.45459852e-01, 2.11747743e-01,
       2.30051398e-01, 3.79772647e-01, 3.94385231e+00, 0.00000000e+00,
       2.41212784e-02, 2.28322102e-01, 1.72689086e-01, 9.49541979e-02,
       4.64155012e-02, 3.37605487e-01, 1.82331798e-01, 8.80205406e-01,
       4.57957526e-01, 9.49030503e-01, 9.99990136e-01, 9.64106804e-01,
       9.94090544e-01, 9.67828689e-01, 1.49015329e-01, 9.99918460e-01,
       8.50954530e-01, 9.80895682e-01, 7.32337415e-01, 2.41321128e-03,
       2.17681424e-02, 1.26138325e-02, 3.45757379e-02, 5.05171666e-01,
       8.89718715e-01, 1.73393912e-01, 6.15856456e-02])

f,g = mcmc_hosp(10,theta0,9)

      
        
        
        

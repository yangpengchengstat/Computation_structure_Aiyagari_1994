
# coding: utf-8

# """
# Computation structure for Aiyagari 1994
# 
# Step 1, define Tauchen 1986 as a function: inputs are persistence, standard deviation, multiple of std and number of state, 
#         and this function reurns Transition Matrix and the state space.
# 
# Step 2, define policy generator for Aiy 1994: inputs are interest rate, utility function parameter, and the parameters for the AR(1).
# This function perform the value function iteration, and return value function, asset policy function, state space and transition matrix.
# 
#    i)call Tauch function to generate state of income shocks and transotion matrix. Given the AR(1), std=sigma (1-rho^2)^0.5
#   
#    ii)set state space of asset a and prespace to store the value and policy
#    
#    iii) set loop for L   
#           set loop for a        
#             calculate max of a' since c>=0           
#                set loop for control a'
#             caculate V(L,a) for each a' and store each V(L,a;a') in a 3-d matrix.
#       So for each combination (L,a)$we get a bounch of V(L,a;a'). 
#       select out the max V(L,a;a') and use the position of max V(L,a;a') to get the policy function a'.
# 
# Step 3, use bisection method to calculate the Ea and K   
# 
#     i) set low and high of r    
#     ii) guess=(low+high)/2 
#     iii) while loop
#        call policy generator
#        get Q and Phi
#        d=Ea-K
#        if d>0, low=guess; else, high=guess
# """
# The code starts from In[1]
# For the following code we et u=5, rho=0.9,sigma=0.4, result is r=0.00497910534591(4.0432$\%$ in Aiy), k=9.54274650444
# 
# My results are similar to Aiy1994 except the case where u=5,rho=0.9 and sig=0.4. When  u=5,rho=0.9 and sig=0.4, r in my results is bout 0.4$\%$ but r is -0.3456$\%$. However the results follows the pattern: the higher the aggregate capital shock(larger rho or sigma), the higher the saving rate, the lower interest rate.

# In[ ]:

import numpy as np
import scipy.stats
import math
import copy
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def markov_chain_approximation_Tauchen1986(persistence, std, multiple, num_partition):
    std_unconditional = math.sqrt((float(std))**2/(1.0-(float(persistence))**2))
    y_N = multiple*std_unconditional
    y_1 = -y_N
    state_space = np.linspace(y_1,y_N,num_partition)
        #shadow_space = np.concatenate((np.array([0]),state_space[1:len(state_space)-1]))
        #transitionMatrix = np.zeros((num_partition,num_partition))
    transitionMatrix = np.empty((num_partition, num_partition), dtype=np.float)
    for j in np.arange(0,len(state_space)):
        for k in np.arange(1,len(state_space)-1):
            transitionMatrix[j,k] = scipy.stats.norm.cdf((state_space[k]-persistence*state_space[j]+(state_space[k]-state_space[k-1])/2)/std,0, 1)-scipy.stats.norm.cdf((state_space[k]-persistence*state_space[j]-(state_space[k]-state_space[k-1])/2)/std,0, 1)
    
    for j in np.arange(0,len(state_space)):
        transitionMatrix[j,0]=scipy.stats.norm.cdf((state_space[0]-persistence*state_space[j]+(state_space[1]-state_space[0])/2)/std,0,1)
        transitionMatrix[j,len(state_space)-1]=1-scipy.stats.norm.cdf((state_space[len(state_space)-1]-persistence*state_space[j]-(state_space[len(state_space)-1]-state_space[len(state_space)-2])/2)/std,0, 1)
    #return {1:transitionMatrix, 2:state_space}
    #return state_space
    result = {0:transitionMatrix, 1:state_space}
    return result
def policyGenerator(r,u,rho,sigma):
    beta=0.96
    alpha=0.36
    delta=0.08
    A=1.0#tech
    multiple=3
    num_partition=7
    persistence=rho
    std=sigma*math.sqrt((1-rho**2))
    w=A*(1-alpha)*(A*alpha/(r+delta))**(alpha/(1-alpha))
    k=((r+delta)/(A*alpha))**(1/(alpha-1))
    #lnL=np.linspace(-3*sigma,3*sigma,7)
    #L=np.exp(lnL)
    L=np.exp(markov_chain_approximation_Tauchen1986(persistence, std, multiple, num_partition)[1])
    #MP=qe.markov.approximation.tauchen(persistence, std, 3, 7)
    #transition_L=copy.deepcopy(np.array(MP))
    transition_L=markov_chain_approximation_Tauchen1986(persistence, std, multiple, num_partition)[0]
    #transition_L=LA.matrix_power(transition,200)
    a_min=0
    a_max=20
    a_len=200
    a_space=np.linspace(a_min,a_max,a_len)
    Vstore=np.zeros((len(L),len(a_space),a_len),dtype=float)
    a_policy=np.zeros((len(L),len(a_space)),dtype=float)
    er=1e8
    V_0=np.zeros((len(L),len(a_space)),dtype=float)
    #c=(beta*(1+r)/(1-beta))**(-1/u)
    #UU=(c**(1-u)-1)/(1-u)
    #V_0=np.ones((len(L),len(a_space)),dtype=float)*UU
    V_1=np.zeros((len(L),len(a_space)),dtype=float)
    while er>0.00001:
            
        Vstore=np.zeros((len(L),len(a_space),a_len),dtype=float)
        a_policystore=np.zeros((len(L),len(a_space),a_len),dtype=float)
        for ind_L in range(len(L)):
            for ind_a in range(len(a_space)):
                a_primeMax=w*L[ind_L]+(1+r)*a_space[ind_a]
                a_primePosition=np.where(a_space<=a_primeMax)
            #print a_primePosition[0]
                for ind_ap in a_primePosition[0]:
                    if u==1:
                        Vstore[ind_L,ind_a,ind_ap]=math.log(w*L[ind_L]+(1+r)*a_space[ind_a]-a_space[ind_ap])+beta*np.dot(transition_L[ind_L,:],V_0[:,ind_ap])
                    else:
                        Vstore[ind_L,ind_a,ind_ap]=((w*L[ind_L]+(1+r)*a_space[ind_a]-a_space[ind_ap])**(1-u)-1)/(1-u)+beta*np.dot(transition_L[ind_L,:],V_0[:,ind_ap])
                    a_policystore[ind_L,ind_a,ind_ap]=a_space[ind_ap]
                V_1[ind_L,ind_a]=np.max(Vstore[ind_L,ind_a,:])
                a_policy[ind_L,ind_a]=a_policystore[ind_L,ind_a,np.argmax(Vstore[ind_L,ind_a,:])]
        er=(np.max(np.absolute(V_1-V_0)))
        #er=(np.sum(np.absolute(V_1-V_0)))
        #print er
        #print np.shape(V_1-V_0)
        V_0=copy.deepcopy(V_1[:,:])
    return {0:V_0,1:a_policy,2:a_space,3:L,4:transition_L}




# In[ ]:

beta=0.96
alpha=0.36
delta=0.08
A=1.0#tech


u=3.0
rho=0.3
sigma=0.2

low=-0.001
high=1/(1+beta)
guess=(low+high)/2.0
#guess=0.0329
epsilon=1e-3
d=1
ctr=0
while abs(d)>epsilon and ctr<=200:
    ap=policyGenerator(guess,u,rho,sigma)
    a_policy=ap[1]
    a_space=ap[2]
    L=ap[3]
    transition_L=ap[4]
    #print transition_L
    Q={(i,j):{i:np.zeros(len(a_space),dtype=float) for i in range(len(L))} for i in range(len(L)) for j in range(len(a_space))}
    for i in range(len(L)):
        for j in range(len(a_space)):
            position=np.where(a_space==a_policy[i,j])
            #print position[0]
            for k in range(len(L)):
                if len(position[0])==1:
                    Q[(i,j)][k][position[0]]=transition_L[i,k]
              
    temp={(i,j):np.zeros(len(L)*len(a_space),dtype=float) for i in range(len(L)) for j in range(len(a_space))}        

    for i in range(len(L)):
        for j in range(len(a_space)):
            temp[(i,j)]=np.array(list(Q[(i,j)][0])+list(Q[(i,j)][1])+list(Q[(i,j)][2])+list(Q[(i,j)][3])+list(Q[(i,j)][4])+list(Q[(i,j)][5])+list(Q[(i,j)][6]))
    
    Q_r=LA.matrix_power(np.array([temp[(i,j)] for i in range(len(L)) for j in range(len(a_space))]),500)   
        
    phi=np.zeros(len(L)*len(a_space))
    for i in range(len(phi)):
        phi[i]=Q_r[0,i]/np.sum(Q_r[0,:])
    print (phi)
    #aT=a_policy.transpose()
    a_policyVector=a_policy.flatten()
    #a_policyVector=aT.flatten()
    k=((guess+delta)/(A*alpha))**(1/(alpha-1))
    Ea=np.dot(a_policyVector,phi)
    #print Ea
    d=k-Ea
    print ('d is',d)
    if d>0:
        low=guess
    else:
        high=guess
    guess=(low+high)/2.0
    ctr +=1
    print (ctr,'guess is',guess,'k is',k)
    assert ctr<=200,'iteration count exceeded'


# In[ ]:




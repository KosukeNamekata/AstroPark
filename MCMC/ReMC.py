#!/usr/bin/env python3         # 1
# coding: utf-8                # 2 not necessary for python3

#This program model stellar light curves of rotating-spotted stars.
#input parameters are set for the solar-type stars.
#Method: Bayesian estimation with MCMC including parallel tempering and adaptive algorithm

#Applicable limit ~ 400 days
#it is preferable that times set to be t=t-t0 (if not, the change in timespan is necessary)
#This program is suited to the Kepler-17, so the application to the rapid rotator requires the increase in number_of_local_min

#Basically, spots are distinguished by the "peak" time, so the initial values must be set so.
#Otherwise, the endless loops will begin.

#In defolt, multi-processing core is set to be 4 (ref, core_of_your_computer) 

#To avoid clash by too many warnings (you can set off this two sentenses).
import warnings
warnings.filterwarnings('ignore')

#Import necesary classes
import numpy as np
import copy
from multiprocessing import Pool
import sys;
#import matplotlib.pyplot as plt
#:"matplotlib" should be setted if you want to visualize in "jupyter notebook"



#Return the temporal evolutions of star spot area from the emergence and decay rates
def spotcandedate2(times, nmax, emergence_rates, decay_rates, amax):
    out = np.empty(len(times))

    if (nmax <= 1) :
        out = decay_rates*(times - times[nmax]) + amax
    elif (nmax >= len(times)) :
        out = emergence_rates*(times - times[nmax]) + amax
    else :
        out[:nmax] = emergence_rates*(times[:nmax] - times[nmax]) + amax #a/(b*tminspotA[s])
        out[nmax:] = decay_rates*(times[nmax:] - times[nmax]) + amax #a/(c*tminspotA[s])

    ss = np.where(out <= 0)[0]
    out[ss] = 0.0
    return out


#Calculate the Likelihood Function
def Probability(theta, x, y, thetamin, thetamax, n_loop, inverse_temperature, number_of_spot):
    
    #These are the parameter selected with MCMC in this loop
    amax = np.exp(theta[0:number_of_spot])
    t_max = theta[number_of_spot:(number_of_spot*2)]
    emerge = np.exp(theta[(number_of_spot*2):(number_of_spot*3)])
    decay = - np.exp(theta[(number_of_spot*3):(number_of_spot*4)])
    rot_period = theta[(number_of_spot*4):(number_of_spot*5)]
    t_max_area = theta[(number_of_spot*5):(number_of_spot*6)]
    
    #Set the reconstructed light curve array
    depth = np.ones(len(x))
    diff = np.zeros(len(x))
    area = np.zeros(len(x))
    
    #This is a kind of unique value for Kepler-17
    #Because Kepler-17's inclination angle are well studied by previous works,
    #and we can simply fit the typical light curves with Gaussian !!
    sigma = float(0.11074091662197166*12.25764769) #Obtained by Fiting light curve with inclination = 87, spot on the equator.

    #This value is the number of local minima for each individual spot (which is also unique to Kepler-17)
    number_of_local_min = 50

    #This modeling can cover less than 400 days (but parameters were selected for 200 days' Kepler-17 data).
    timespan = [0, (np.max(x) - np.min(x))]

    #Here, we made the spot areal temporal evolutions by "spotcandidate2" method.
    for ii in range(number_of_spot):
        area = spotcandedate2(x, np.where(abs(x-t_max_area[ii]) == np.min(abs(x-t_max_area[ii])))[0][0], emerge[ii], decay[ii], amax[ii])
        tmin = np.zeros(number_of_local_min)
        tmin = (np.array(range(number_of_local_min))-int(number_of_local_min*0.5))*rot_period[ii] + t_max[ii]
        tmin_local_min = tmin[np.where( (tmin > timespan[0]) & (tmin < timespan[1]))[0]]
        for jj in range(len(tmin_local_min)):
            #Here, we assume Gaussian-shape spot-origin stellar brightness variations.
            depth += - area*np.exp( - (x - tmin_local_min[jj])**2 / (2 * sigma**2))

    #Q4 = np.array([55184.8777, 55274.7038])-55184
    #Q5 = np.array([55275.9912, 55370.6600])-55184
    #Q6 = np.array([55371.9473, 55461.7939])-55184
    QC = [91.34749999999985, 184.89254999999685]

    #Here, diff means the difference between the modeled and original light curves.
    qc = [np.where(x < QC[0]), np.where( (x > QC[0]) & (x < QC[1])), np.where(x > QC[1])]
    for q in range(3):
    	y[qc[q]] = y[qc[q]] - np.mean( y[[qc[q]]] )
    	diff[qc[q]] = (depth[qc[q]]-np.mean(depth[qc[q]]))/np.mean(depth[qc[q]])
    #y = y - np.max(y)
    #diff = (depth-np.mean(depth))/np.mean(depth)
    #diff = diff - np.max(diff)
    
    ##PLOT 
    #if (n_loop % 10 == 0):
        #plt.figure(figsize=(12, 5))
        #plt.plot(x, diff, label="model")
        #plt.plot(x, y, label="observation")
        #plt.plot(x, y - diff, linestyle="dashed",alpha=0.2, label="errors")
        #plt.xlim([200,400])
        #plt.ylabel("Brightness Variation (dF/F)",fontsize=15)
        #plt.xlabel("Date (from 2455000)",fontsize=15)
        #plt.legend(fontsize=15) 
        #plt.show()

    #lp is initially instoled as a kind of special factor to avoid the parameter to go out of the parameter ranges 
    #However, this part is unnecessary now!! Please ignore!! I just remain this part for some future applications!!
    #if all( (theta[kk]-thetamin[kk])*(theta[kk]-thetamax[kk]) <= 0 for kk in range(number_of_spot*6)):
    #    lp = 0.0
    #else:
    #    lp = - np.inf
    lp = 0.0

    #This is a noise level of Kepler light curve which is derived for Kepler-17, but this ~0.1% noise level would be a typical values
    #for all of the Kepler data!!
    noise = 0.001 
    
    #Return Log(Likelihood)
    p = lp - np.sum( (( (y-diff))**2*inverse_temperature/noise**2 ) ) - len(x)*(0.5*np.log(2*np.pi) + np.log(noise))

    return float(p)



#Randomly select the next step of MCMC
def Q(c, mu, sigma, thetamin, thetamax, number_of_spot,parameter_priority):
    a = c + np.random.normal(mu, sigma)
    num_errors = 0
    for kk in range(len(a)):

        while ( (a[kk]-thetamin[kk])*(a[kk]-thetamax[kk]) > 0 ):
            a[kk] = c[kk] + np.random.normal(mu[kk], sigma[kk])
            if (num_errors == 100):
                #If the parameter frequently go out of the parameter range (> 100), then something strange can happend.
                #But according my experience, this rarely happen if the initial parameters (ranges) are properly selected!!
                print("Some Errors happend！! Please set properly the initial values!!", "Sigma = ",sigma[kk], "Parameter = ",kk)
                sys.exit(1)
            num_errors += 1

    #In this part, the exchange of Spot A-B to Spot B-A is avoided for the artifitial bimodality of posterior distribution.
    #We used the peak time as the order of the spots, but there may another appropriate way to escape this problem.        
    for jj in range(number_of_spot):
        index = jj+number_of_spot*parameter_priority
        if (jj==0): continue
        if (a[index] < a[index-1]):
            a[index] = copy.copy(a[index-1])
    
    #Return the next MCMC step parameters
    return a.tolist()



##Metropolis Algorithm
def metropolis(inputs):
    N = inputs[0]
    x = inputs[1]
    y = inputs[2]
    theta0 = inputs[3]
    sigma = inputs[4]
    mu = inputs[5]
    thetamax = inputs[6]
    thetamin = inputs[7]
    inverse_temperature = inputs[8]
    visualize = inputs[9]
    loop_number_out_replica = inputs[10]
    scale_parameter = inputs[11]
    number_of_spot = inputs[12]
    parameter_priority = inputs[13]
    burn_in_adaptive = inputs[14]

    current = theta0.tolist() 
    candidate = theta0.tolist() 
    sample = []
    accept_ratio = []
    likelihood = [] ##Not yet installed
    
    i = 0
    T_next, T_prev = 0.0, 0.0
    a = 1.0
  
    loop_number = loop_number_out_replica
    #sample.append(current)

    while (i <  N):
        
        candidate = Q(np.array(current), np.zeros(len(theta0)), sigma*scale_parameter, thetamin, thetamax, number_of_spot, parameter_priority)

        if (i  == 0):
            T_prev = Probability(np.array(current), x, y, thetamin, thetamax, int(visualize), inverse_temperature, number_of_spot)
        T_next = Probability(np.array(candidate), x, y, thetamin, thetamax, -1, inverse_temperature, number_of_spot)
        #if ((inverse_temperature == 1) & ((loop_number % 100) == 0)): print('Likelihood =', T_next, ', The ratio =', np.exp(T_next-T_prev))
        probability_ratio = np.exp( T_next - T_prev )

        if probability_ratio > 1 or probability_ratio > np.random.uniform(0, 1):
            # Update state
            current = copy.copy(candidate)
            accept_ratio.append(loop_number)
            likelihood.append(copy.copy(T_next))
            T_prev = copy.copy(T_next)
            index_accept = 1
            #if ((inverse_temperature == 1) & ((loop_number % 100) == 0)): print('Accepted!!')
        else:
            index_accept = 0
            likelihood.append(copy.copy(T_prev))
            
        #Adaptive Part: 次のステップの分散(と言うより、ここでは標準偏差)を計算する。
        #bn = 10/(burn_in_adaptive*0.01 + loop_number) #Araki et al とは少し違うが、これでも十分だと判断
        #bn = 1/(100+loop_number)
        #When Searching
        bn = (1-np.exp(-loop_number/100000))*(10/(10000+loop_number)) 
        #When Sampling
        #bn = (10/(10000+loop_number)) 

        sigma_prev = copy.copy(sigma)
        sigma = np.abs( ( sigma**2 + bn*( (candidate - mu)**2 - sigma**2 ) )**0.5 )
        mu += bn*(candidate - mu)

        if (len(np.where(sigma <= 0)[0]) != 0):
            sigma[np.where(sigma <= 0)[0]] = copy.copy(sigma_prev[np.where(sigma <= 0)[0]])
        scale_parameter_prev = copy.copy(scale_parameter)
        scale_parameter += bn*(index_accept - 0.2) #Ideal acceptance ratio = 0.234
        if (scale_parameter <= 0):
            scale_parameter = copy.copy(scale_parameter_prev)
        #print(loop_number, "分散の最大：",  np.max(sigma), "最小：", np.min(sigma),"スケ：", int(scale_parameter))
        
        sample.append(current)
        loop_number += 1 #もしかしたら、アクセプトされたものだけに適応するのかもしれない、、まぁAraki＋１３にはタイムステップで書いてあったので違うだろうけど。
        i += 1
        
    return [sample, accept_ratio, sigma, mu, scale_parameter, likelihood, inverse_temperature]




def mcmc_replica_exchange(size_simulation, x, y, sigma0, thetamax, thetamin, theta_init = None, number_of_spot = 5,
    size_replica = 15, frequency_exchange = 10, core_of_your_computer = 4, parameter_priority = 5, burn_in=0.2, likelihood_threshold = 0):

    theta = np.random.uniform(thetamin, thetamax)
    number_of_parameter = number_of_spot*6
    scale_parameter = np.ones(size_replica)
    burn_in_adaptive = int(burn_in*size_simulation)
    sigma_for_each_replica = np.array(range(size_replica))/size_replica + 1

    #Initial t_spot_area_is_max is set to be located along the order (t0<t1<t2<...<tN)
    t_min = thetamin[number_of_spot*parameter_priority]
    t_max = thetamax[number_of_spot*parameter_priority]
    #t_step = (np.array(range(number_of_spot))+0.5)/number_of_spot*(t_max-t_min)
    t_step = np.random.uniform(thetamin[number_of_spot*parameter_priority: number_of_spot*(parameter_priority+1)], thetamax[number_of_spot*(parameter_priority): number_of_spot*(parameter_priority+1)])
    t_step.sort()
    theta[(number_of_spot*parameter_priority):(number_of_spot*(parameter_priority+1))] = copy.copy(t_step[:])

    #set with the list format...
    sigma0list = sigma0.tolist()
    theta_prev = theta.tolist()
    #Here, we set the initial inverse_temperature (which will be adaptively modified in every exchange)
    #inverse_temperature = (np.array(range(size_replica, 0, -1)))/size_replica*0.01*100 #等間隔なら、これくらい。
    inverse_temperature = np.exp(-np.array(range(0, size_replica, 1))) #Ikuta method.
    orders = np.array(range(size_replica))
    condition = [copy.copy(orders.tolist())]
    theta_next = []
    theta_next_tentative = []
    accept_ratio = []
    likelihood = []
    #likelihood_threshold = 0 #-14000 for spot number = 5
    likelihood_now = -1000000

    #If you want to see the fitted results during simulation, then, you can see (not yey adapted)
    visualize = np.ones(size_replica)*(-1)
    visualize[0] = 0
    q = 0
    kk = 0
    
    #cdef list current = theta.tolist() 
    sample = []
    sample.append(theta.tolist()[0:(len(thetamin))])
    likelihood.append(-np.inf)
    
    #ここで初期値を決定している
    for jj in range(size_replica-1):
        #t_step = np.random.uniform(thetamin[number_of_spot*5: number_of_spot*6], thetamax[number_of_spot*5: number_of_spot*6])
        #t_step.sort()
        t_step = np.random.uniform(thetamin[number_of_spot*parameter_priority: number_of_spot*(parameter_priority+1)], thetamax[number_of_spot*(parameter_priority): number_of_spot*(parameter_priority+1)])
        t_step.sort()

        theta_random = np.random.uniform(thetamin, thetamax)
        theta_random[(number_of_spot*parameter_priority):(number_of_spot*(parameter_priority+1))] = copy.copy(t_step[:])
        #If we want to set the initial condition
        if ((jj == 0) & (theta_init is not None)):
            theta_random = theta_init
         
        theta_prev.extend( copy.copy( theta_random.tolist() ) )
        sigma_init_for_each_replica = sigma_for_each_replica[jj+1]*sigma0
        sigma0list.extend( sigma_init_for_each_replica.tolist() )
    
    sigma = np.array(sigma0list)
    mu = np.array(copy.copy(theta_prev) )
    
    #ここからMCMC chainが始まる。
    while (q <=  (size_simulation/frequency_exchange) ):
        #range( int(size_simulation/frequency_exchange) )
        #初期化
        #if ((kk % 1000) == 0): print("Repluca Routine: No. ", kk)
        theta_next = []
        
        if ( (((kk*frequency_exchange) % 1000) == 0) & (kk > 0) ):
            accept_ratio_print = np.array(accept_ratio)
            if (q == 0):
                print("loop number: ", kk*frequency_exchange, ' | L: ', int(likelihood_now[0]),' | q: ', q )
            else:
                print("loop number: ", kk*frequency_exchange, " | acceptance ratio: ", int(len(np.where(accept_ratio_print >= (kk*frequency_exchange*(1 - burn_in)) )[0])/(kk*frequency_exchange*burn_in)*100), '% | L: ', int(likelihood_now[0]),' | q: ', q )
            

        #レプリカ数に応じて、**並行して**ループを回す。
        inputs = []
        for index_replica in range(size_replica):
            inputs.append([frequency_exchange , x, y, np.array(theta_prev[index_replica*number_of_parameter:(index_replica+1)*number_of_parameter]),
                sigma[index_replica*number_of_parameter:(index_replica+1)*number_of_parameter],
                np.array(theta_prev[index_replica*number_of_parameter:(index_replica+1)*number_of_parameter]),
                thetamax, thetamin, inverse_temperature[index_replica], visualize[index_replica],kk*frequency_exchange, 
                scale_parameter[index_replica], number_of_spot, parameter_priority, burn_in_adaptive])

        p = Pool(core_of_your_computer)
        outputs = p.map( metropolis, inputs )
        p.close()  # add this.
        p.terminate()  # add this. 
        outputs = np.array(outputs)

        for index_replica in range(size_replica):
            ss = np.where(outputs[:,6] == inverse_temperature[index_replica])[0][0]
            theta_next_tentative, accept_ratio_tentative, sigma_tentative, mu_tentative, scale_parameter[index_replica], likelihood_tentative, invs = copy.copy(outputs[ss])
            #theta_next_tentative, accept_ratio_tentative, sigma_tentative, mu_tentative, scale_parameter[index_replica] 

            
            if (index_replica == 0): #ここで、求めたい逆温度T=1のサンプルのみ全て記憶する。
                theta_next = theta_next_tentative[len(theta_next_tentative)-1]
                likelihood_now = copy.copy(likelihood_tentative)
                
                if ( (likelihood_now[0] >= likelihood_threshold) or (q >= 1)): #十分高いLikelihoodでないと、記憶しない
                    for mm in range(len(theta_next_tentative)):
                        sample.append( copy.copy(theta_next_tentative[mm]) )
                    likelihood.extend( likelihood_tentative )
                    accept_ratio.extend( accept_ratio_tentative )
                    q += 1
                
                mu[index_replica*number_of_parameter:(index_replica+1)*number_of_parameter] = copy.copy(mu_tentative)
                sigma[index_replica*number_of_parameter:(index_replica+1)*number_of_parameter] = copy.copy(sigma_tentative)
            
            else: #それ以外の場合は、最終形態だけを記憶しておく。
                theta_next.extend( (theta_next_tentative[len(theta_next_tentative)-1]) )
        
        #if ( ((kk % frequency_exchange) == 0) &  (kk != 0) ):
        index_exchange = int(np.random.uniform(0, size_replica-1))
        #print("Index Exchange is", index_exchange)
        theta1 = theta_next[index_exchange*number_of_parameter:(index_exchange+1)*number_of_parameter]
        theta2 = theta_next[(index_exchange+1)*number_of_parameter:(index_exchange+2)*number_of_parameter] 
        mu1 = mu[index_exchange*number_of_parameter:(index_exchange+1)*number_of_parameter]
        mu2 = mu[(index_exchange+1)*number_of_parameter:(index_exchange+2)*number_of_parameter] 
        #No. 2の方が、"逆温度"が高い方。つまり、「移動しやすい方」
        
        T1_numerator = Probability(np.array(theta1), x, y, thetamin, thetamax, -1, inverse_temperature[index_exchange+1], number_of_spot)
        T2_numerator = Probability(np.array(theta2), x, y, thetamin, thetamax, -1, inverse_temperature[index_exchange], number_of_spot)
        T1_denominator = Probability(np.array(theta1), x, y, thetamin, thetamax, -1, inverse_temperature[index_exchange], number_of_spot)
        T2_denominator = Probability(np.array(theta2), x, y, thetamin, thetamax, -1, inverse_temperature[index_exchange+1], number_of_spot)
        
        if (kk == 0):
            condition = [copy.copy(orders.tolist())]
        
        #パラレルテンパリングの実行部分>>>>>>>>>>>>>>>>>>>>>>>>>>
        if (np.random.uniform(size=1) < np.exp(T1_numerator+T2_numerator-T1_denominator-T2_denominator)):
            #print("Index Exchange is", index_exchange, "Did Echange Occur?-----Yes")
            a_parallel_index = 1
            a_append = copy.copy(orders[np.where(orders == index_exchange)[0][0]])
            b_append = copy.copy(orders[np.where(orders == index_exchange+1)[0][0]])
            na_append = np.where(orders == index_exchange)[0][0]
            nb_append = np.where(orders == index_exchange+1)[0][0]
            orders[na_append] = copy.copy(b_append)
            orders[nb_append] = copy.copy(a_append)
            if (q >= 1): 
                condition.append(copy.copy(orders.tolist()))
            
            theta_next[index_exchange*number_of_parameter:(index_exchange+1)*number_of_parameter], theta_next[(index_exchange+1)*number_of_parameter:(index_exchange+2)*number_of_parameter] = copy.copy(theta2),copy.copy(theta1)
            mu[index_exchange*number_of_parameter:(index_exchange+1)*number_of_parameter], mu[(index_exchange+1)*number_of_parameter:(index_exchange+2)*number_of_parameter] = copy.copy(mu2),copy.copy(mu1)

            if (index_exchange==0):
                visualize[0] = 0
            else:
                visualize[0] = 1
        else:
            #print("Index Exchange is", index_exchange, "Did Echange Occur?-----No")
            a_parallel_index=0
            visualize[0] = 1
            if (q >= 1):
                condition.append(copy.copy(orders.tolist()))
        #パラレルテンパリングの実行部分<<<<<<<<<<<<<<<<<<<<<<<<<<
            
        #Adaptive MCMC: Change the tempering temperatures
        #an = 1/(1+(kk*frequency_exchange)/(20+10*(index_exchange+1)))+np.log(np.exp(- np.log(inverse_temperature[index_exchange+1]) )+1)
        #an = 1/(burn_in_adaptive*0.1 + kk*frequency_exchange)
        an = (1-np.exp(-kk*frequency_exchange/50000))*(10/(50000+kk*frequency_exchange))
        
        #inverse_temperature[index_exchange+1] = np.exp(np.log(copy.copy(inverse_temperature[index_exchange+1])) - copy.copy(an)*(1 - 0.5))
        if ( (index_exchange + 1) == (size_replica -1) ):
            next_temperature = np.exp(np.log(inverse_temperature[index_exchange+1]) - an*(a_parallel_index - 0.5))
            prev_temperature_max = inverse_temperature[index_exchange]
            if ( next_temperature < prev_temperature_max ):
                inverse_temperature[index_exchange+1] = np.exp(np.log(inverse_temperature[index_exchange+1]) - an*(a_parallel_index - 0.5))
            else:
                inverse_temperature[index_exchange+1] = 0.9*prev_temperature_max
        else:
            next_temperature = np.exp(np.log(inverse_temperature[index_exchange+1]) - an*(a_parallel_index - 0.5))
            prev_temperature_min = inverse_temperature[index_exchange+2]
            prev_temperature_max = inverse_temperature[index_exchange]
            if ( (next_temperature > prev_temperature_min) & (next_temperature < prev_temperature_max) ):
                inverse_temperature[index_exchange+1] = np.exp(np.log(inverse_temperature[index_exchange+1]) - an*(a_parallel_index - 0.5))
            elif ( (next_temperature <= prev_temperature_min) ):
                inverse_temperature[index_exchange+1] = 0.99*prev_temperature_min+0.01*prev_temperature_max
            elif ( (next_temperature >= prev_temperature_max) ):
                inverse_temperature[index_exchange+1] = 0.01*prev_temperature_min+0.99*prev_temperature_max
        
        theta_prev = copy.copy(theta_next)
        kk += 1
        
    sigma_return = sigma[0:number_of_parameter]*scale_parameter[0]
    return np.array(sample), np.array(accept_ratio), inverse_temperature, np.array(condition), np.array(likelihood), sigma_return




##Parameter tuning is one of the most important task!!
def input_parameter(period, t_max, t_min, number_of_spot):
    sigma = 3*np.hstack((np.ones(number_of_spot)*0.001, np.ones(number_of_spot)*0.001*period, np.ones(number_of_spot)*0.001, np.ones(number_of_spot)*0.001, np.ones(number_of_spot)*0.0001, np.ones(number_of_spot)*0.05))
    thetamin = np.hstack((np.log(np.ones(number_of_spot)*0.001), np.ones(number_of_spot)*0, np.log(np.ones(number_of_spot)*0.00001), np.log(np.ones(number_of_spot)*0.00001), period*np.ones(number_of_spot)*0.95, np.ones(number_of_spot)*t_min))
    thetamax = np.hstack((np.log(np.ones(number_of_spot)*0.05), np.ones(number_of_spot)*period, np.log(np.ones(number_of_spot)*0.1), np.log(np.ones(number_of_spot)*0.1), period*np.ones(number_of_spot)*1.05, np.ones(number_of_spot)*t_max))
    return thetamin, thetamax, sigma


##Calculate the Bayesian Factors
def bayesian_factor_calculator(thetamin, thetamax, theta):
	return 0


def BIC(likelihood, number_of_observation, number_of_parameter):
	return - 2*np.log(np.max(likelihood)) + number_of_parameter*np.log(number_of_observation)

#def WBIC()










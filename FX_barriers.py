import numpy as np
import math
from scipy.stats import norm
from scipy.stats import ncx2
from scipy.special import gammainc
from scipy.optimize import minimize 

class barrier_price:

    def Wt_simulation(dW, T, N):
        ''' 
        Simulates brownian paths
        
        Input: 
        - dW: BM increments  
        - T: total time for simulation 
        - N: total number of simulations 

        Outptìut:
        -  Wt: simulated BMs
        ''' 

        Wt = np.empty((T, N))
        # cumulative sum of the single BM increments  
        np.cumsum(dW, axis=0, out=Wt)
        return Wt

    def sigma_simulation(Wt, sigma0, alpha , t):
        ''' 
        Exact GBM simulation with mu=0 
        '''
        return sigma0 * np.exp(alpha * Wt - 0.5 * (alpha ** 2) * t[1:])

    def integrated_var_small_disturbances(N, rho, alpha, sigmat, dt, dW, U):
        ''' 
        Small disturbance expansion Chen B. & al (2011)
        '''
        dW_2, dW_3, dW_4 = np.power(dW, 2), np.power(dW, 3), np.power(dW, 4)

        m1 = alpha * dW
        m2 = (1. / 3) * (alpha ** 2) * (2 * dW_2 - dt / 2)
        m3 = (1. / 3) * (alpha ** 3) * (dW_3 - dW * dt)
        m4 = (1. / 5) * (alpha ** 4) * ((2. / 3) * dW_4 - (3. / 2) * dW_2 * dt + 2 * np.power(dt, 2))
        m = (sigmat ** 2) * dt * (1. + m1 + m2 + m3 + m4)

        v = (1. / 3) * (sigmat ** 4) * (alpha ** 2) * np.power(dt, 3)
        mu = np.log(m) - (1. / 2) * np.log(1. + v / m ** 2)
        sigma2 = np.log(1. + v / (m ** 2))
        A_t = np.exp(np.sqrt(sigma2) * norm.ppf(U) + mu)
        v_t = (1. - rho ** 2) * A_t
        return v_t

    def absorption_conditional_prob(a, b):
        ''' 
        Probability that F_(ti+1) is absorbed by the 0 barrier conditional on inital value S0 
        '''
        cprob = 1. - gammainc(b / 2, a / 2)  # scipy gammainc is normalized by gamma(b) 
        return cprob

    def QE_scheme(ai, b):
        ''' 
        Test for Andersen L. (2008) Quadratic exponential Scheme (Q.E.) 
        '''
        k = 2. - b
        lbda = ai
        s2 = (2 * (k + 2 * lbda))
        m = k + lbda
        psi = s2 / m ** 2 
        return m, psi

    def target_function(c, a, b, u):
        return 1 - ncx2.cdf(a, b, c) - u  

    def root_chi2(a, b, u):
        ''' 
        Inversion of the non central chi-square distribution 
        '''
        c0 = a
        bnds = [(0., None)]
        res = minimize(barrier_price.target_function, c0, args=(a, b, u), bounds=bnds)
        return res.x[0]
    
    def SABR_MC(F0=0.04, sigma0=0.07, nu=0.5, beta=0.25, rho=0.4, psi_threshold=2., n_years=1.0, T=252, N=1000):
        """
        Simulation of SABR process with absoption at 0.

        Inputs:
        - F0: forward underlying value 
        - sigma0: initial volatility 
        - nu, beta, rho: SABR parameters 
        - psi_threshold: threshold of applicability of Andersen Quadratic Exponential (QE) algorithm
        - n_years: number of year fraction for the simulation
        - T: number of Monte Carlo steps
        - N: number of simulated paths

        Outputs: 
        -------
        Ft: array with each MC path stored in a column
        """
        # grid - vector of time steps - starts at 1e-10 to avoid comutational problems 
        tis = np.linspace(1E-10, n_years, T + 1)  
        t = np.expand_dims(tis, axis=-1) # for numpy broadcasting 
        dt = 1. / (T) # delta_t = time steps 
        
        # Distributions samples
        dW2 = np.random.normal(0.0, math.sqrt(dt), (T, N))
        U1 = np.random.uniform(size=(T, N))
        U = np.random.uniform(size=(T, N))
        Z = np.random.normal(0.0, 1., (T, N))
        W2t = barrier_price.Wt_simulation(dW2, T, N)
        
        # volatility process
        sigma_t = barrier_price.sigma_simulation(W2t, sigma0, nu, t)
        
        # integrated variance-values = integrals between ti-1 and ti (not over entire interval [0,ti])
        v_t = barrier_price.integrated_var_small_disturbances(N, rho, nu, sigma_t, dt, dW2, U1)
            
        b = 2. - ((1. - 2. * beta - (1. - beta) * (rho ** 2)) / ((1. - beta) * (1. - rho ** 2)))

        # initialize underlying values
        Ft = np.zeros((T-1, N))
        Ft = np.insert(Ft, 0, F0 * np.ones(N), axis=0)
        
        for n in range(0, N):
            for ti in range(1, T):
                
                if Ft[ti - 1, n] == 0.:
                    Ft[ti, n] = 0.
                    continue
                a = (1. / v_t[ti - 1, n]) * (((Ft[ti - 1, n] ** (1. - beta)) / (1. - beta) + (rho / nu) * (sigma_t[ti, n] - sigma_t[ti - 1, n])) ** 2)
                # absorption probabilities Formula 2.10
                pr_zero = barrier_price.absorption_conditional_prob(a, b)
                
                if pr_zero > U[ti - 1, n]:
                    Ft[ti, n] = 0.
                    continue
                
                m, psi = barrier_price.QE_scheme(a, b)

                if m >= 0 and psi <= psi_threshold:
                    # QE scheme
                    e2 = (2. / psi) - 1. + math.sqrt(2. / psi) * math.sqrt((2. / psi) - 1.)
                    d = m / (1. + e2)
                    Ft[ti, n] = np.power(((1. - beta) ** 2) * v_t[ti - 1, n] * d * ((math.sqrt(e2) + Z[ti - 1, n]) ** 2), 1. / (2.* (1. - beta))) 
                    
                elif psi > psi_threshold or (m < 0 and psi <= psi_threshold):
                    # direct inversion for small values
                    c_star = barrier_price.root_chi2(a, b, U[ti - 1, n])
                    Ft[ti, n] = np.power(c_star * ((1. - beta) ** 2) * v_t[ti - 1, n], 1. / (2. - 2. * beta))

                # print Ft[ti, n]
            
        return Ft
    
    def barrier_selection(barrier_type, MC_T, K, mask):
        
        barrier_types = {

            'C_di': lambda mask: np.mean(np.maximum(MC_T[:,-1] - K, 0) * mask), 
            'C_do': lambda mask: np.mean(np.maximum(MC_T[:,-1] - K, 0) * (1 - mask)), 
            'C_ui': lambda mask: np.mean(np.maximum(MC_T[:,-1] - K, 0) * mask), 
            'C_uo': lambda mask: np.mean(np.maximum(MC_T[:,-1] - K, 0) * (1 - mask)), 
            'P_di': lambda mask: np.mean(np.maximum(K - MC_T[:,-1], 0) * mask), 
            'P_do': lambda mask: np.mean(np.maximum(K - MC_T[:,-1], 0) * (1 - mask)), 
            'P_ui': lambda mask: np.mean(np.maximum(K - MC_T[:,-1], 0) * mask), 
            'P_uo': lambda mask: np.mean(np.maximum(K - MC_T[:,-1], 0) * (1 - mask))
            
            }
        
        if barrier_type not in barrier_types:
            raise ValueError('Invalid barrier type.\n\n'
                             'Availabe options:\n'
                             '\'C_di\': Down-and-in call,\n'
                             '\'C_do\': Down-and-out call,\n' 
                             '\'C_ui\': Up-and-in call,\n' 
                             '\'C_uo\': Up-and-out call,\n'
                             '\'P_di\': Down-and-in put,\n' 
                             '\'P_do\': Down-and-out put,\n'
                             '\'P_ui\': Up-and-in put,\n'
                             '\'P_uo\': Up-and-out put.')
        
        else:
            return barrier_types[barrier_type](mask)

    def MC_barrier_pricing(barrier_type, H, F0, rd, K, nu, alpha, beta, rho, psi_threshold, tau, n_MC, n_steps, n_simulation):
        '''
        Price the barrier option desired

        Input: 
        - barrier_type: type of barrier that we want to price 
        - H: barrier level 
        - F0: forward underlying value 
        - rd: domestic risk free rate
        - K: option strike 
        - nu, alpha, beta, rho: SABR parameters 
        - psi_threshold: threshold that defines when to use QE scheme or direct inversion scheme 
        - tau: option time to maturity 
        - n_MC: total number of Monte Carlo simulations 
        - n_steps: number of time steps in a single simulation  
        - n_simulation: number of simulations to performorm to get a single Monte Carlo price 

        Output: 
        - barrier_MC_price: Monte Carlo barrier price 
        '''
        # instantiate empty array for barrier options 
        barrier = np.zeros(n_MC)

        # if continuous_monitoring == True:
        #     H = H*np.exp(-np.sign(H-F0)*0.5826*sigma*np.sqrt(dt))

        # for n_MC times perform Monte carlo simulation 
        for i in range(n_MC):

            # montecarlo simulation 
            MC_T = barrier_price.SABR_MC(F0, nu, alpha, beta, rho, psi_threshold, tau, n_steps, n_simulation).T

            # create a mask for the barrier payoff (given the type provided as input) 
            if barrier_type == 'C_ui' or barrier_type == 'C_uo' or barrier_type == 'P_ui' or barrier_type == 'P_uo':
                mask = np.sum(MC_T >= H, axis=1)
                mask[mask > 0] = 1
            else: 
                mask = np.sum(MC_T <= H, axis=1)
                mask[mask > 0] = 1

            # price the barrier using the Monte Carlo simulated values 
            barrier[i] = barrier_price.barrier_selection(barrier_type, MC_T, K, mask)
        
        # compute the average discounted montecarlo price given n_MC Monte Carlo simulations 
        barrier_MC_price = np.mean(np.exp(-rd*tau) *(barrier))

        # print option price
        print('Price', barrier_type, ': ', barrier_MC_price)
        
        return barrier_MC_price
    
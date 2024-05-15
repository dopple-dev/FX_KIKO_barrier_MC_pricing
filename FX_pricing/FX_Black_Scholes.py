import numpy as np
import scipy.stats as st
from scipy.stats import norm

class Black_Scholes:
    
    def __init__(self, Option_info, Process_info):

        self.S0 = Option_info.S0 # underlying 
        self.K = Option_info.K # strike 
        self.T = Option_info.T # tenor 
        self.payoff = Option_info.payoff # payoff 
        self.exercise_style = Option_info.exercise_style

        self.r = Process_info.r # interest rate 
        self.sigma = Process_info.sigma # diffusion coefficient 
        self.rd = Process_info.rd
        self.rf = Process_info.rf
        self.fx = Process_info.fx

    def get_payoff(self, S): # obtain payoff function 
        if (self.payoff == 'call' or self.payoff == 'c' or self.payoff == 'Call' or self.payoff == 'C'):
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == 'put' or self.payoff == 'p' or self.payoff == 'Put' or self.payoff == 'P':
            Payoff = np.maximum(self.K - S, 0)
        return Payoff
    
    def BS_pricer_self(self):
            
        d1 = (np.log(self.S0 / self.K) + (self.rd - self.rf + .5 * self.sigma**2) * self.T) / (self.sigma* np.sqrt(self.T))
        d2 = (np.log(self.S0 / self.K) + (self.rd - self.rf - .5 * self.sigma**2) * self.T) / (self.sigma *np.sqrt(self.T))
        if (self.payoff == 'call' or self.payoff == 'c' or self.payoff == 'Call' or self.payoff == 'C'):
            return self.S0*st.norm.cdf(d1)-np.exp(-(self.rd-self.rf)*self.T)*self.K*st.norm.cdf(d2) 
        elif self.payoff == 'put' or self.payoff == 'p' or self.payoff == 'Put' or self.payoff == 'P':
            return self.K*st.norm.cdf(-d2)*np.exp(-(self.rd-self.rf)*self.T) - self.S0*st.norm.cdf(-d1)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    @staticmethod 
    def PC_Parity(given, find, option_value, S0, K, T, rd, rf):
            
        if ((str(find).lower() == 'call' or str(find).lower() == 'c' or find == 1) and 
            (str(given).lower() == 'put' or str(given).lower() == 'p' or given == -1)):
            return option_value + S0 - K * np.exp((rd-rf) * T)
    
        elif ((str(find).lower() == 'put' or str(find).lower() == 'p' or find == -1) and 
              (str(given).lower() == 'call' or str(given).lower() == 'c' or given == 1)):
            return option_value - S0 + K * np.exp((rd-rf) * T)

        else:
            raise ValueError('invalid option payoff type.')
        
    @staticmethod
    def BS_pricer(payoff, S0, K, T, sigma, rd, rf):
            
            if (isinstance(payoff, np.ndarray) and (payoff.dtype == np.int64 or payoff.dtype == np.int32) 
                and np.all(np.logical_or(payoff == 1, payoff == -1)) ):
            
                d1 = (np.log(S0 / K) + (rd - rf + .5 * sigma**2) * T) / (sigma* np.sqrt(T))
                d2 = (np.log(S0 / K) + (rd - rf - .5 * sigma**2) * T) / (sigma *np.sqrt(T))
                return payoff*S0*norm.cdf(payoff*d1)-payoff*np.exp(-(rd-rf)*T)*K*norm.cdf(payoff*d2)

            else:
                d1 = (np.log(S0 / K) + (rd - rf + .5 * sigma**2) * T) / (sigma* np.sqrt(T))
                d2 = (np.log(S0 / K) + (rd - rf - .5 * sigma**2) * T) / (sigma *np.sqrt(T))
                if str(payoff).lower() == 'call' or str(payoff).lower() == 'c' or payoff == 1:
                    return S0*st.norm.cdf(d1)-np.exp(-(rd-rf)*T)*K*st.norm.cdf(d2) 
                elif str(payoff).lower() == 'put' or str(payoff).lower() == 'p' or payoff == -1:
                    return K*st.norm.cdf(-d2)*np.exp(-(rd-rf)*T) - S0*st.norm.cdf(-d1)
                else:
                    raise ValueError("invalid type. Set 'call' or 'put'")

        
    

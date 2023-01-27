import numpy as np
import math 
from scipy.linalg import hadamard


#applied on one inpt vector. unnormalized
def fwht(x):
    d=x.size
    power=math.log(d,2)
    if not (int(power)==power):
        raise ValueError("The size of an array must must be a power of 2")
    if(d==1):
        return x
    halves=np.split(x,2)
    res_1=fwht(halves[0])
    res_2=fwht(halves[1])
    part_1=np.vstack((res_1, res_1))
    part_2=np.vstack((res_2, res_2*(-1)))
    return(part_1+part_2)

#the data is in the columns of X, X of size d by N
def fwht_matrix(data):
    d, N=data.shape
    result=np.zeros((N,d))
    print(result)
    for i in range(N):
        data_point=np.transpose(data)[i]
        result[i]=np.reshape(fwht(data_point),(d,))
    return np.transpose(result)



class FastJLT:
    def __init__(self, target_dim="auto", epsilon="0.1"):
        self.target_dim=target_dim
        self.epsilon=epsilon
        
        
    
    def _sample_D_matrix(self, size):
        rand_diag=np.random.randint(0,2, size=size)*2-1
        return(np.diag(rand_diag))
    
   # def _sample_P_matrix(self, rows, cols, sparcity):
        
    #the data matrix X contains data vectors in the columns, size: d by N    
    def fit_transform(self, X, sparsity_q="auto", sparse=False):
        d,N=X.shape
        power=math.log(d,2) #if d is not a power of 2, padd it
        if not (int(power)==power):
            int_power=math.ceil(power)
            d=int(math.pow(2, int_power))
            padding_array=np.zeros((d-X.shape[0],N))
            X=np.vstack((X, padding_array))            #padding the input with zeros    
        if isinstance(self.target_dim, str):
            #using the default num of new dims, set k=8log N/epsilon^2
            #as proved to have the preservation guarantees
            new_dims=(8/self.epsilon**2)*math.log(N,2)
        else:
            new_dims=self.target_dim   
        D_matrix=self._sample_D_matrix(d)
        #H_matrix=hadamard(d)
        



 
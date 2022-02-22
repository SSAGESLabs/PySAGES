import numpy as np

def sum_gaussians(x, hills):
    
    fes = np.zeros(len(x))
    
    for i in range(len(x)):
        
        for j in range(len(hills.cv.values)):
            
            delta_xi = x[i] - hills.cv.values[j]
            
            fes[i] += hills.height.values[j] * np.exp(- (delta_xi * delta_xi)/(2 * hills.sigma.values[j] * hills.sigma.values[j]) )
            
    
    return fes
            
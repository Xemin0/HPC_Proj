"""
Python File to Plot Roofline Mdoel given CPU specs
along with selected meaured performances 

"""

import matplotlib.pyplot as plt
import numpy as np

peak_performance = 130.5 * 1e12 # Tera FLOPs/s
memory_BW = 672 * 1e9 #GB/s

intensity = np.logspace(-2, np.log10(500), 500)
performance_bound = np.ones_like(intensity) * peak_performance / 1e12 # convert to TFLOP/s
BW_bound = intensity * memory_BW / 1e12 # convert to TFLOP/s

# Take the minimum of the two bounds
RF_bound = np.minimum(performance_bound, BW_bound)

#Plot
plt.figure(figsize = (12, 8))
#plt.loglog(intensity, performance_bound, 'r--', label = 'Performance Bound')
#plt.loglog(intensity, BW_bound, 'b--', label = 'Bandwidth Bound')
plt.loglog(intensity, RF_bound, 'r--', label = 'Bandwidth Bound')
plt.loglog(intensity, performance_bound, 'b--', label = 'Performance Bound')

# Add achieved performances
achieved_performance = [2.65e7, 2.24e7, 8.87e6]
AI = [1.0 / 16] * len(achieved_performance)
plt.loglog(AI, achieved_performance, 'go',mfc = 'none', label = 'Achieved Performance')


plt.ylabel('Performance [TFLOP/s]')
plt.xlabel('Arithmetic Intensity [FLOP/Byte]')

plt.title('Roofline Model for NVIDIA RTX 6000')

plt.grid(True, which = 'both', ls = '--')
plt.legend(loc = 4) # Lower Right corner
plt.show()

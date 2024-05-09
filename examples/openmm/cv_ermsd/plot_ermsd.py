import matplotlib.pyplot as plt
import numpy as np

a = np.loadtxt("ermsd.txt")

b = np.loadtxt("ermsd_barnaba.txt")

mi, ma = np.min(a), np.max(a)

plt.figure(figsize=(5, 5))

plt.scatter(a[1::], b[:-1], alpha=0.5)
# plt.plot(a[100::100], '-o')
# plt.plot(b[:-1], '-o')

plt.plot([mi * 0.99, ma * 1.01], [mi * 0.99, ma * 1.01])

plt.xlim(mi * 0.99, ma * 1.01)

plt.ylim(mi * 0.99, ma * 1.01)
plt.show()

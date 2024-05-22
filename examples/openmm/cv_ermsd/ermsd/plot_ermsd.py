import matplotlib.pyplot as plt
import numpy as np

a = np.loadtxt("ermsd.txt")

b = np.loadtxt("ermsd_barnaba.txt")

mi, ma = np.min(a), np.max(a)

plt.figure(figsize=(5, 5))

plt.scatter(a[1::], b[:-1], alpha=0.5)
# plt.plot(a[1::], '-o')
# plt.plot(b[:-1], '-o')

plt.plot([mi * 0.99, ma * 1.01], [mi * 0.99, ma * 1.01])

plt.xlim(mi * 0.99, ma * 1.01)

plt.ylim(mi * 0.99, ma * 1.01)
plt.xlabel("eRMSD from pysages", fontsize=15)
plt.ylabel("eRMSD from barnaba", fontsize=15)
plt.savefig("ermsd_comparison.png", dpi=300)
plt.show()

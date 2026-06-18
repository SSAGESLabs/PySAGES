import matplotlib.pyplot as plt
import numpy as np

a = np.loadtxt("Q.txt")

b = np.loadtxt("Q_posthoc.txt")

mi, ma = np.min(a), np.max(a)

plt.figure(figsize=(5, 5))

plt.scatter(a[1::], b[:-1], alpha=0.5)
# plt.plot(a[1::], '-o')
# plt.plot(b[:-1], '-o')

plt.plot([mi * 0.999, ma * 1.001], [mi * 0.999, ma * 1.001])

plt.xlim(mi * 0.999, ma * 1.001)

plt.ylim(mi * 0.999, ma * 1.001)
plt.xlabel("Q from pysages", fontsize=15)
plt.ylabel("Q from post-hoc calculation", fontsize=15)
plt.savefig("Q_comparison.png", dpi=300)
plt.show()

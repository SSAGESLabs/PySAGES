import matplotlib.pyplot as plt
import numpy as np

a = np.loadtxt("ermsd.txt")

b = np.loadtxt("ermsd_cg.txt")

mi, ma = np.min(a), np.max(a)

plt.figure(figsize=(5, 5))

# plt.scatter(a[:], b[:], alpha=0.5)
plt.plot(a[:], "-o")
plt.plot(b[:], "-o")

# plt.plot([mi * 0.99, ma * 1.01], [mi * 0.99, ma * 1.01])

# plt.xlim(mi * 0.99, ma * 1.01)

plt.ylim(mi * 0.99, ma * 1.01)
plt.show()

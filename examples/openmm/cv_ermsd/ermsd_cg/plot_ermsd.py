import matplotlib.pyplot as plt
import numpy as np

a = np.loadtxt("ermsd.txt")

b = np.loadtxt("ermsd_cg.txt")

mi, ma = np.min(a), np.max(a)

plt.figure(figsize=(5, 5), dpi=300)

plt.scatter(a[:], b[:], alpha=0.25)

plt.plot([mi * 0.99, ma * 1.01], [mi * 0.99, ma * 1.01])

plt.xlim(mi * 0.99, ma * 1.01)
plt.ylim(mi * 0.99, ma * 1.01)
plt.xlabel("AA eRMSD")
plt.ylabel("Inferred eRMSD")
plt.savefig("ermsd_aa_cg_comparison.png", bbox_inches="tight", dpi=300)
plt.show()

plt.figure(figsize=(5, 4), dpi=300)
plt.plot(a[:], "-o", label="AA eRMSD", alpha=0.5)
plt.plot(b[:], "-o", label="Inferred eRMSD", alpha=0.5)
plt.xlabel("Time step")
plt.ylabel("eRMSD")
plt.legend()
plt.ylim(mi * 0.99, ma * 1.01)
plt.savefig("ermsd_aa_cg_traj.png", bbox_inches="tight", dpi=300)
plt.show()

#!/usr/bin/env python3
# Reference plot script for the matching CSV. CI does not run this.
# Usage: ./plot.py <path-to-csv>
import sys, csv
import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt(sys.argv[1], delimiter=',', names=True)
for K in np.unique(data['K_sub']):
    sel = data['K_sub']==K
    plt.loglog(data['x'][sel], data['norm_delta_omega'][sel], label=f'K_sub={int(K)}')
plt.xlabel('x'); plt.ylabel('||delta||_omega'); plt.legend(); plt.tight_layout()
plt.savefig(sys.argv[1].replace('.csv', '.png'), dpi=120)


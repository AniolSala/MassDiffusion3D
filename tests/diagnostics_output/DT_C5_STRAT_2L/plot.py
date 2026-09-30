#!/usr/bin/env python3
# Reference plot script for the matching CSV. CI does not run this.
# Usage: ./plot.py <path-to-csv>
import sys, csv
import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt(sys.argv[1], delimiter=',', names=True)
modes = np.unique(np.stack([data['mode_n'], data['mode_m']], axis=1), axis=0)
for n, m in modes:
    sel = (data['mode_n']==n) & (data['mode_m']==m)
    plt.loglog(data['h'][sel], data['abs_rho'][sel], label=f'(n={int(n)}, m={int(m)})')
plt.xlabel('h'); plt.ylabel('|rho|'); plt.legend(); plt.tight_layout()
plt.savefig(sys.argv[1].replace('.csv', '.png'), dpi=120)


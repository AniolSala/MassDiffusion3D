#!/usr/bin/env python3
# Reference plot script for the matching CSV. CI does not run this.
# Usage: ./plot.py <path-to-csv>
import sys, csv
import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt(sys.argv[1], delimiter=',', names=True)
for x in np.unique(data['x']):
    sel = data['x']==x
    plt.semilogy(data['mode_index_k'][sel], data['abs_C_tilde'][sel], label=f'|C_tilde|, x={x:.0e}')
plt.xlabel('mode_index_k'); plt.ylabel('|C_tilde_k(x)|'); plt.legend(); plt.tight_layout()
plt.savefig(sys.argv[1].replace('.csv', '.png'), dpi=120)


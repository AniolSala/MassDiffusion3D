#!/usr/bin/env python3
# Reference plot script for the matching CSV. CI does not run this.
# Usage: ./plot.py <path-to-csv>
import sys, csv
import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt(sys.argv[1], delimiter=',', names=True)
ratio = np.divide(data['bound'], np.maximum(data['norm_correction_omega'], 1e-300))
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].loglog(data['x'], data['bound'],                  label='Gronwall bound')
ax[0].loglog(data['x'], data['norm_correction_omega'],  label='||correction||_omega')
ax[0].legend()
ax[1].loglog(data['x'], ratio, 'k-', label='bound / ||correction||_omega')
ax[1].axhline(1.0, color='r', ls=':')
ax[1].set_xlabel('x'); ax[1].legend()
plt.tight_layout(); plt.savefig(sys.argv[1].replace('.csv', '.png'), dpi=120)


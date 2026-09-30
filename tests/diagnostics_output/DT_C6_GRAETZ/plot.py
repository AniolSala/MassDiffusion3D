#!/usr/bin/env python3
# Reference plot script for the matching CSV. CI does not run this.
# Usage: ./plot.py <path-to-csv>
import sys, csv
import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt(sys.argv[1], delimiter=',', names=True)
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].loglog(data['x'], data['norm_R_omega'],     label='||R||_omega')
ax[0].loglog(data['x'], data['norm_R_omega_inv'], label='||R||_omega_inv')
ax[0].loglog(data['x'], data['norm_S_omega'],     label='||S||_omega', ls='--')
ax[0].loglog(data['x'], data['norm_D_omega'],     label='||D||_omega', ls='--')
ax[0].legend()
ax[1].loglog(data['x'], data['relative_residual'], 'k-', label='|R|/(|S|+|D|)')
ax[1].set_xlabel('x'); ax[1].legend()
plt.tight_layout(); plt.savefig(sys.argv[1].replace('.csv', '.png'), dpi=120)


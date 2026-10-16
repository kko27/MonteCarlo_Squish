import numpy as np
import os
from matplotlib import pyplot as plt


fold_list = ['./rot000/24-procs/', './rot045/24-procs/', './rot090/24-procs/',
             './rot135/24-procs/', './rot180/24-procs/']

name_list = ['0°', '45°', '90°', '135°', '180°']

TIMESTART = 100
TIMEEND = 125

plt.figure(figsize=(8, 5))

# Platen force from the mid-height cross-section (section_force.csv, written by 05).
# Top and bottom platens carry the same force in equilibrium, so one plot covers both.
for fold in fold_list:
    file_name = "section_force.csv"
    data = np.loadtxt(os.path.join(fold, file_name), delimiter=',', skiprows=1)
    time_s = data[TIMESTART:TIMEEND,0]
    platen_Fz = data[TIMESTART:TIMEEND,1]/1e6       # Convert to µN
    plt.plot(time_s, np.abs(platen_Fz), label= name_list[fold_list.index(fold)], linewidth=1)

plt.xlabel("Time (s)", fontsize = 14)
plt.ylabel("Force (µN)", fontsize = 14)
plt.legend(fontsize = 14)
plt.grid(True)
plt.savefig("reaction_force.png", dpi=450)
plt.clf()

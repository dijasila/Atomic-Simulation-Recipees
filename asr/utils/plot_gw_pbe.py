import numpy as np
from ase.io.jsonio import read_json, write_json
from gpaw import GPAW
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from gpaw.spinorbit import soc_eigenstates

def get_edges(energies, ef):
    allcb = []
    allvb = []
    for en in energies:
        if en[0] - ef > 0:
            allcb.append(min(en))
        elif en[0] - ef < 0:
            allvb.append(max(en))
    return [max(allvb), min(allcb)] 

# G0W0
dct11 = read_json('results-asr.gs.json')
data11 = dct11["kwargs"]["data"]
evac1 = data11["evac"]
dct1 = read_json('results-asr.gw.json')
data1 = dct1["kwargs"]["data"]
path1 = data1['bandstructure']['path']
ef1 = data1['efermi_gw_soc'] - evac1
e1 = data1['bandstructure']['e_int_mk'] - evac1
vbm1 = data1["vbm_gw"] - evac1
cbm1 = data1["cbm_gw"] - evac1
edg1 = get_edges(e1, ef1)
x1, X1, labels1 = path1.get_linear_kpoint_axis()

# LCAO / PW
calc2 = GPAW('bs_pw.gpw', txt=None)
soc = soc_eigenstates(calc2)
bs = calc2.band_structure()
x2, X2, labels2 = bs.get_labels()
evac2 = np.mean(np.mean(calc2.get_electrostatic_potential(), axis=0), axis=0)[0]
e2 = soc.eigenvalues().T
e2 -= evac2
ef2 = soc.fermi_level - evac2
edg2 = get_edges(e2, ef2)

refs = [ef1, ef2]
ylim = [min(refs) - 4, max(refs) + 4]

#getting shifts for homobilayer
shifts = {}
shifts["shift_c1"] = abs(edg1[1] - edg2[1])
shifts["shift_c2"] = abs(edg1[1] - edg2[1])
shifts["shift_v1"] = abs(edg1[0] - edg2[0])
shifts["shift_v2"] = abs(edg1[0] - edg2[0])
write_json("shifts.json", shifts)

# plotting
style1 = dict(
    color='C1',
    ls='-',
    lw=1.5,
    zorder=0)

style2 = dict(
    color='#546caf',
    ls='-',
    lw=1.5,
    zorder=0)

ax = plt.figure(figsize=(12, 9)).add_subplot(111)


ax.plot(x1, e1[0], **style1, label="G0W0")
for e in e1[1:]:
    ax.plot(x1, e, **style1)

ax.plot(x2, e2[0], **style2, label="PW")
for e in e2[1:]:
    ax.plot(x2, e, **style2)

for X in X1[1:-1]:
    ax.axvline(X, color='#bbbbbb')
ax.set_xticklabels([lab.replace('G', r'$\Gamma$') for lab in labels1], fontsize=20)
ax.axhline(ef1, color=style1["color"], ls=':')
ax.axhline(ef2, color=style2["color"], ls=':')
ax.axhline(edg2[0], color='#bbbbbb', ls=':', lw=2.0)

ax.set_title("G0W0 vs PW", fontsize = 24)
#ax.set_ylim([-2, 2])
ax.set_ylim(ylim)
ax.set_xlim([x1[0], x1[-1]])
ax.set_ylabel("E - Evac", fontsize=24)
ax.set_xticks(X1)
plt.yticks(fontsize=20)
ax.xaxis.set_tick_params(width=3, length=10)
ax.yaxis.set_tick_params(width=3, length=10)
plt.setp(ax.spines.values(), linewidth=3)
plt.legend(loc="upper left", fontsize = 20)

plt.show()

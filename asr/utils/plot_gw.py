from ase.io.jsonio import read_json
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

dct = read_json('results-asr.gw.json')
data = dct["kwargs"]["data"]
path = data['bandstructure']['path']
reference = data['efermi_gw_soc']
ef = reference
fontsize = 24

e_mk = data['bandstructure']['e_int_mk'] - reference
x, X, labels = path.get_linear_kpoint_axis()
label = "E - Ef"

# hse with soc
style = dict(
    color='C1',
    ls='-',
    lw=1.5,
    zorder=0)

ax = plt.figure(figsize=(12, 9)).add_subplot(111)

for e_m in e_mk:
    ax.plot(x, e_m, **style)

for l in X[1:-1]:
    ax.axvline(l, color='#bbbbbb')

ax.set_title(f"GW on bilayer MoS2\nEcut = 100 eV", fontsize = 24)
ax.set_ylim([-4, 4])
ax.set_xlim([x[0], x[-1]])
ax.set_ylabel(label, fontsize=24)
ax.set_xticks(X)
ax.set_xticklabels([lab.replace('G', r'$\Gamma$') for lab in labels], fontsize=20)
plt.yticks(fontsize=20)
xlim = ax.get_xlim()
ax.axhline(ef - reference, c='0.5', ls=':')
ax.xaxis.set_tick_params(width=3, length=10)
ax.yaxis.set_tick_params(width=3, length=10)
plt.setp(ax.spines.values(), linewidth=3)

plt.show()

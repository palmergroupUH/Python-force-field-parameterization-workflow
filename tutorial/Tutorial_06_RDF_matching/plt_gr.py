#!/share/apps/canopy-1.4.1/Canopy_64bit/User/bin/python
import numpy as np
import matplotlib
#matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator) 
import colormaps as cmaps
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
import os

import IO.reader

fig_width = 3.0  # width in inches 
fig_height = fig_width/1.333   # height in inches 
fig_size =  [fig_width,fig_height] 
params = {'backend': 'Agg',
          'axes.labelsize': 8,
          'axes.titlesize': 8,
		  'font.size': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'figure.figsize': fig_size,
          'savefig.dpi' : 600,
          'font.family': 'sans-serif',
          'axes.linewidth' : 0.5,
          'xtick.major.size' : 2,
          'ytick.major.size' : 2,
          'font.size' : 8,
          'svg.fonttype' : 'none',
          'pdf.fonttype' : 42
          }

rcParams.update(params) 

# Create inset 
fig = plt.figure()
ax1 = fig.add_subplot(111)
lwidth=0.8
msize=4
#left, bottom, width, height = [0.3, 0.3, 0.3, 0.3] 
#ax2 = fig.add_axes([left, bottom, width, height]) 

# colormap
n_curves = 11
values = list(range(n_curves))
plt.register_cmap(name='magma', cmap=cmaps.magma)
jet = cm = plt.get_cmap('magma')
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

# Data
# column 1: q, column 2: P(q) 

	

#Colors
col=list(range(n_curves))
col[0] = scalarMap.to_rgba(values[0])
col[1] = scalarMap.to_rgba(values[1])
col[2] = scalarMap.to_rgba(values[2])
col[3] = scalarMap.to_rgba(values[3])
col[4] = scalarMap.to_rgba(values[4])
col[5] = scalarMap.to_rgba(values[5])
col[6] = scalarMap.to_rgba(values[6])
col[7] = scalarMap.to_rgba(values[7])
col[8] = scalarMap.to_rgba(values[8])
col[9] = scalarMap.to_rgba(values[9])

#Labels

ref_gr_data = IO.reader.np_loadtxt("../Data_samples_and_scripts/ReferenceData/rdf/mW_300K_1bar/Ref.gr")
guess_gr_data = IO.reader.np_loadtxt("fit_3_paras_440461/Output/mW_300K_1bar_guess.gr")
best_gr_data = IO.reader.np_loadtxt("fit_3_paras_440461/Output/mW_300K_1bar_best.gr")

ax1.plot(ref_gr_data[:,0],ref_gr_data[:,1],color="k",label="Ref")

ax1.plot(guess_gr_data[:,0],guess_gr_data[:,1],color="r",label="Guess")

#plt.plot(best_gr_data[:,0],best_gr_data[:,1],color="r",label="Best predicted")

#ax1.scatter(T,predicted_data_T,color="r",label="Best Predicted")

#plt.ylim([0.99,1.48]) 
#ax1.set_ylim([0.995,1.01]) 
#ax1.set_ylim([0.8,1.05]) 
ax1.set_xlabel("r (A)")
#ax1.set_xticks([0.995,1.0,1.005,1.01])
ax1.set_ylabel("g(r)")
ax1.legend(loc="upper right")

# Plot P(q) vs q: 
#ax1.set_title("production")
#ax1.scatter(tersoff[:,0],tersoff [:,1],s=6,label=plot_ID[0],color=col[0])
#ax1.scatter(tersoff_table[:,0],tersoff_table[:,1],s=6,label=plot_ID[1],color=col[5])

minorLocator =  MultipleLocator(0.5)
majorLocator=MultipleLocator(5) 
ax = plt.subplot(111)

#ax1.yaxis.set_minor_locator( minorLocator) 
#ax1.xaxis.set_minor_locator(majorLocator) 

#ax1.tick_params(axis='x',direction='in', pad=2.0)
#ax1.tick_params(axis='y',direction='in', pad=2.0)

#ax1.tick_params(axis='y',which='minor',length=1) 
#ax1.tick_params(axis="x",which="minor",length=1) 

#ax1.tick_params() 


ax1.set_xlabel(r'r($\AA$)',labelpad=-1)
#ax1.set_ylabel(r'$\rho$'+  '(g/cm$^{-3}$)', labelpad=0) 

#ax1.set_xlim([225,290.0])
#ax1.set_xticks([240,260,280,300,320,340]) 

handles, labels = ax.get_legend_handles_labels()

plt.legend(handles[::-1],labels[::-1],loc="upper right",fontsize=5,frameon=False, labelspacing=0.07,ncol=2)
#plt.legend(loc="upper right") 

#plt.legend(loc=(0.1,0.385),fontsize=7,frameon=False, labelspacing=0.15,ncol=1)

left, bottom, width, height = [0.48, 0.62, 0.48, 0.3] 

plt.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.90, wspace=0.0, hspace=0.0)

#plt.savefig('fig1a.pdf',transparent=True)
plt.savefig('gr_mW_300K_guess_ref.png',transparent=False)
#plt.savefig('fig1a.eps',transparent=True)

plt.show()

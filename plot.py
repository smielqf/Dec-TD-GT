import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.style.use('seaborn')
# plt.rc('xtick', labelsize=12)
# plt.rc('ytick', labelsize=12)
# plt.rc('legend', fontsize='x-large', loc='best')
# plt.rc('xlabel', fontsize=20)
# plt.rc('ylabel', fontsize=20)
matplotlib.rcParams['xtick.labelsize'] = 23
matplotlib.rcParams['ytick.labelsize'] = 23
matplotlib.rcParams['figure.subplot.left'] = 0.2
matplotlib.rcParams['figure.subplot.bottom'] = 0.15
matplotlib.rcParams['figure.subplot.right'] = 0.95
matplotlib.rcParams['figure.subplot.top'] = 0.9


files = ['sbe_gt_list.txt', 'sbe_list.txt', 'sbe_gc_list.txt', 'sbe_gt_gc_list.txt', 'sbe_cta_list.txt']

sbe_gt = np.loadtxt(files[0]) 
sbe = np.loadtxt(files[1]) 
sbe_gc = np.loadtxt(files[2]) 
sbe_gt_gc = np.loadtxt(files[3]) 
sbe_cta = np.loadtxt(files[4])

diff_sbe_gt_or = sbe_gt - sbe
diff_sbe_gc_or = sbe_gc - sbe
diff_sbe_gt_gc_or = sbe_gt_gc - sbe
diff_sbe_cta_or = sbe_cta - sbe

plt.plot([(np.mean(diff_sbe_gt_or[:x+1])) for x in range(len(diff_sbe_gc_or))], label='GT', linewidth=3)
plt.plot([(np.mean(diff_sbe_gc_or[:x+1])) for x in range(len(diff_sbe_gc_or))], label='GC', linewidth=3)
plt.plot([(np.mean(diff_sbe_gt_gc_or[:x+1])) for x in range(len(diff_sbe_gc_or))], label='GT+GC', linewidth=3)
# plt.plot([(np.mean(diff_sbe_cta_or[:x+1])) for x in range(len(diff_sbe_gc_or))], label='CTA')
plt.legend(loc='best', fontsize=30)
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2),)
plt.ylabel('$\Delta$MSBE', fontsize=30)
plt.xlabel('Step', fontsize=30)
plt.savefig('diff_sbe.pdf')

files = ['ce_gt_list.txt', 'ce_list.txt', 'ce_gc_list.txt', 'ce_gt_gc_list.txt', 'ce_cta_list.txt']
ce_gt = np.loadtxt(files[0])
ce = np.loadtxt(files[1])
ce_gc = np.loadtxt(files[2])
ce_gt_gc = np.loadtxt(files[3])
ce_cta = np.loadtxt(files[4])
diff_ce = ce_gt - ce
plt.clf()
plt.plot(diff_ce)
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2),)
plt.savefig('diff_ce.pdf')

# scale = [ k / np.log(k) for k in range(len(ce))]
plt.clf()
plt.plot(([np.mean(ce[1:x+1]) for x in range(1,len(ce))]), label='Dis-TD(0)', linewidth=3)
plt.plot(([np.mean(ce_gt[1:x+1]) for x in range(1,len(ce))]), label='Dec-TD(0)+GT', linewidth=3)
plt.plot(([np.mean(ce_gc[1:x+1]) for x in range(1,len(ce))]), label='Dec-TD(0)+GC', linewidth=3)
plt.plot(([np.mean(ce_gt_gc[1:x+1]) for x in range(1,len(ce))]), label='Dec-TD(0)+GT+GC', linewidth=3)
# plt.plot(([np.mean(ce_cta[1:x+1]) for x in range(1,len(ce))]), label='Dis-TD(0)+CTA')
plt.legend(loc='best', fontsize=30)
plt.xlabel('Step', fontsize=30,)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
plt.ylabel('MCE', fontsize=30)
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2),)
plt.savefig('ce.pdf')

plt.clf()
plt.plot(np.log10([np.mean(sbe[:x+1]) for x in range(len(sbe))]), label='TD(0)')
plt.plot(np.log10([np.mean(sbe_gt[:x+1]) for x in range(len(sbe))]), label='TD(0)+GT')
plt.plot(np.log10([np.mean(sbe_gt_gc[:x+1]) for x in range(len(sbe))]), label='TD(0)+GT+GC')
plt.plot(np.log10([np.mean(sbe_gc[:x+1]) for x in range(len(sbe))]), label='TD(0)+GC')
plt.plot(np.log10([np.mean(sbe_cta[:x+1]) for x in range(len(sbe))]), label='TD(0)+CTA')
plt.legend()
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2),)
plt.savefig('sbe.pdf')
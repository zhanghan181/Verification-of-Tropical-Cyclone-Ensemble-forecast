import f90nml
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import numpy as np
import pandas as pd
import meteva
import meteva.base as meb      # this module is  for IO、data transformation and so on
import meteva.method as mem    # this module contains verification algorithm
import meteva.product as mpd   # this module contains veification tools 
import numpy as np          
import datetime                      
import copy
import pandas as pd
#matplotlib inline
#load_ext autoreload
import meteva.base as meb
import numpy as np
from matplotlib.colors import BoundaryNorm

nml = f90nml.read('/public1/home/zhanghan181/perl5/addcnop-haishen/parameter-ncl')
print(nml)
CASE=nml['domains']['CASE']
NDIX = nml['domains']['NDIX']  
NDJX = nml['domains']['NDJX']
NNDKX = nml['domains']['NNDKX']  
BDY = nml['domains']['BDY'] 
ilonstart = nml['domains']['ilonstart']  
ilonend = nml['domains']['ilonend'] 
jlatstart = nml['domains']['jlatstart']  
jlatend = nml['domains']['jlatend']
init_time = nml['compu_perts']['init_time'] 
opti_time = nml['compu_perts']['opti_time'] 
init_norm = nml['compu_perts']['init_norm']  
fin_norm = nml['compu_perts']['fin_norm']
DEL = nml['compu_perts']['DEL']
mincnop = nml['compu_perts']['mincnop']  
maxcnop = nml['compu_perts']['maxcnop']
sfc = nml['compu_perts']['sfc']  
mindel   = nml['compu_perts']['mindel']
maxdel   = nml['compu_perts']['maxdel']
min_opti_time    =   nml['compu_perts']['min_opti_time']
max_opti_time    =   nml['compu_perts']['max_opti_time']
mininival = nml['compu_spg2']['mininival']
maxinival = nml['compu_spg2']['maxinival']
EPS = nml['compu_spg2']['EPS']
MAXIT = nml['compu_spg2']['MAXIT']
MMAX = nml['compu_spg2']['MMAX']
# 平均误差, 在 +-0位置增加了强烈的对比，以凸显误差的符号
colors_grid = ["#D0DEEA", "#B4D3E9", "#6FB0D7", "#3787C0", "#105BA4", "#07306B", "#07306B"]
cmap,clev = meb.def_cmap_clevs(meb.cmaps.me,vmin = -12,vmax = 12)
print(clev)
meb.tool.color_tools.show_cmap_clev(cmap,clev)
for nc in range(mindel,maxdel):
     delta = nc*0.60
     #format(delta, '.2f')
     #("%.2f" % delta)
     data  = pd.read_csv("/public1/home/zhanghan181/perl5/addcnop-haishen/energy-evo_"+str('%.2f' % delta)+"-fin3norm-init2norm-HAISHEN-6h.txt",header=None, delim_whitespace=True)
     data2 = np.loadtxt("/public1/home/zhanghan181/perl5/addcnop-haishen/energy-evo_"+str('%.2f' % delta)+"-fin3norm-init2norm-HAISHEN-12h.txt",comments='*')#,header=None, delim_whitespace=True)
     data3 = np.loadtxt("/public1/home/zhanghan181/perl5/addcnop-haishen/energy-evo_"+str('%.2f' % delta)+"-fin3norm-init2norm-HAISHEN-24h.txt",comments='*')#,header=None, delim_whitespace=True)
     data4 = np.loadtxt("/public1/home/zhanghan181/perl5/addcnop-haishen/energy-evo_"+str('%.2f' % delta)+"-fin3norm-init2norm-HAISHEN-36h.txt",comments='*')#,header=None, delim_whitespace=True)

     data_6h =np.array(data)
     data_12h=np.array(data2)
     data_24h=np.array(data3)
     data_36h=np.array(data4)
     print(data_12h.shape)
     print(data.shape)
     error = np.random.randn(30)
     #print(data_12h)
     #print(data2)
     numcnop = np.arange(30)
     itime   = np.arange(17)

     #np.array(data_12h)
     #pd.DataFrame(data_12h,columns=['CNOP1','CNOP2','CNOP3','CNOP4'])
    #  plt.errorbar(numcnop,data_24h[4,:],fmt='bo-',yerr=0.3,xerr=0.03)
    #  plt.xlim(0,30)
    #  plt.show()

    #  plt.errorbar(numcnop,data_12h[15,:],fmt='bo-',yerr=0.3,xerr=0.03)
    #  plt.xlim(0,30)
    #  plt.show()
     
     #plt.gca().set_prop_cycle(plt.cycler('color',plt.cm.Spectral(np.linspace(0, 1, 15))))
     fig = plt.figure(nc)
     ax1 = fig.add_subplot(2,2,1)#(1,4,1)
     plt.gca().set_prop_cycle(plt.cycler('color',plt.cm.jet(np.linspace(0, 1, 15))))
     #ax2 = fig.add_subplot(2,2,2)
     for numsv in range(1,30,2):
         df1 = plt.plot(itime ,data_6h[:,numsv] ,marker='o',markersize=3,linewidth=0.5,label = 'CNOP'+str(numsv))  
     plt.ylim(0,140)  # 限定纵轴的范围  
     plt.xlim(0,17)  
     ax2 = fig.add_subplot(2,2,2)#(1,4,2)  
     plt.gca().set_prop_cycle(plt.cycler('color',plt.cm.jet(np.linspace(0, 1, 15))))  
     for numcnop2 in range(1,30,2):  
         df2 = plt.plot(itime,data_12h[:,numcnop2],marker='o',markersize=3,linewidth=0.5,label = 'CNOP'+str(numcnop2))
     plt.ylim(0,140)  # 限定纵轴的范围
     plt.xlim(0,17)
     ax3 = fig.add_subplot(2,2,3)#(1,4,3)
     plt.gca().set_prop_cycle(plt.cycler('color',plt.cm.jet(np.linspace(0, 1, 15))))
     for numcnop2 in range(1,30,2):
         df3 = plt.plot(itime,data_24h[:,numcnop2],marker='o',markersize=3,linewidth=0.5,label = 'CNOP'+str(numcnop2))
     plt.ylim(0,140)  # 限定纵轴的范围
     plt.xlim(0,17)
     ax4 = fig.add_subplot(2,2,4)#(1,4,4)
     plt.gca().set_prop_cycle(plt.cycler('color',plt.cm.jet(np.linspace(0, 1, 15))))
     for numcnop2 in range(1,30,2):
         df4 = plt.plot(itime,data_36h[:,numcnop2],marker='o',markersize=3,linewidth=0.5,label = 'CNOP'+str(numcnop2))
     plt.ylim(0,140)  # 限定纵轴的范围
     plt.xlim(0,17)
     #ax4.legend(loc=2, bbox_to_anchor=(-4.0,-0.2),borderaxespad = 0.,ncol=5)  ##设置ax4中legend的位置，将其放在图外
     ax4.legend(loc=2, bbox_to_anchor=(-1.55,-0.2),borderaxespad = 0.,ncol=5)  ##设置ax4中legend的位置，将其放在图外
     plt.savefig('/public1/home/zhanghan181/perl5/addcnop-haishen/evl-cnop-'+str(delta)+'.eps', format='eps', dpi=1000, bbox_inches='tight')
     
     data  = pd.read_csv("/public1/home/zhanghan181/perl5/addcnop-haishen/energy-evo-sv-_"+str('%.2f' % delta)+"-fin3norm-init2norm-HAISHEN-6h.txt",header=None, delim_whitespace=True)
     data2 = np.loadtxt("/public1/home/zhanghan181/perl5/addcnop-haishen/energy-evo-sv-_"+str('%.2f' % delta)+"-fin3norm-init2norm-HAISHEN-12h.txt",comments='*')#,header=None, delim_whitespace=True)
     data3 = np.loadtxt("/public1/home/zhanghan181/perl5/addcnop-haishen/energy-evo-sv-_"+str('%.2f' % delta)+"-fin3norm-init2norm-HAISHEN-24h.txt",comments='*')#,header=None, delim_whitespace=True)
     data4 = np.loadtxt("/public1/home/zhanghan181/perl5/addcnop-haishen/energy-evo-sv-_"+str('%.2f' % delta)+"-fin3norm-init2norm-HAISHEN-36h.txt",comments='*')#,header=None, delim_whitespace=True)
     
     sv_6h =np.array(data)
     sv_12h=np.array(data2)
     sv_24h=np.array(data3)
     sv_36h=np.array(data4)

     fig1 = plt1.figure(maxdel+nc)
     ax5 = fig1.add_subplot(2,2,1)#(1,4,1)
     plt1.gca().set_prop_cycle(plt.cycler('color',plt.cm.jet(np.linspace(0, 1, 15))))
     #ax2 = fig.add_subplot(2,2,2)
     for numsv in range(1,30,2):
         df1 = plt.plot(itime ,sv_6h[:,numsv] ,marker='o',markersize=3,linewidth=0.5,label = 'SV'+str(numsv))  
     plt1.ylim(0,140)  # 限定纵轴的范围  
     plt1.xlim(0,17)  
     ax6 = fig1.add_subplot(2,2,2)#(1,4,2)  
     plt1.gca().set_prop_cycle(plt.cycler('color',plt.cm.jet(np.linspace(0, 1, 15))))  
     for numcnop2 in range(1,30,2):  
         df2 = plt.plot(itime,sv_12h[:,numcnop2],marker='o',markersize=3,linewidth=0.5,label = 'SV'+str(numcnop2))
     plt1.ylim(0,140)  # 限定纵轴的范围
     plt1.xlim(0,17)
     ax7 = fig1.add_subplot(2,2,3)#(1,4,3)
     plt1.gca().set_prop_cycle(plt.cycler('color',plt.cm.jet(np.linspace(0, 1, 15))))
     for numcnop2 in range(1,30,2):
         df3 = plt.plot(itime,sv_24h[:,numcnop2],marker='o',markersize=3,linewidth=0.5,label = 'SV'+str(numcnop2))
     plt1.ylim(0,140)  # 限定纵轴的范围
     plt1.xlim(0,17)
     ax8 = fig1.add_subplot(2,2,4)#(1,4,4)
     plt1.gca().set_prop_cycle(plt.cycler('color',plt.cm.jet(np.linspace(0, 1, 15))))
     for numcnop2 in range(1,30,2):
         df4 = plt.plot(itime,sv_36h[:,numcnop2],marker='o',markersize=3,linewidth=0.5,label = 'SV'+str(numcnop2))
     plt1.ylim(0,140)  # 限定纵轴的范围
     plt1.xlim(0,17)
     #ax4.legend(loc=2, bbox_to_anchor=(-4.0,-0.2),borderaxespad = 0.,ncol=5)  ##设置ax4中legend的位置，将其放在图外
     ax8.legend(loc=2, bbox_to_anchor=(-1.30,-0.2),borderaxespad = 0.,ncol=5)  ##设置ax4中legend的位置，将其放在图外
     plt1.savefig('/public1/home/zhanghan181/perl5/addcnop-haishen/evl-sv-'+str(delta)+'.eps', format='eps', dpi=1000, bbox_inches='tight')

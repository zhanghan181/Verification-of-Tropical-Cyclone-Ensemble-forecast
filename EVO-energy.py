import f90nml
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
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


CASE4     = np.array(['Matsa', 'Fung-wong','Maria','Haishen'])
wmaxtime  = np.array([21, 17, 21, 17])

for case in range(0,5,3): 
    print(case)
    print(CASE4[case])
    for nc in range(mindel,maxdel+1): 
        delta = nc*DEL
        print(delta)
    for opti in range(min_opti_time,max_opti_time+1):
        if opti == 1: 
            opti_time=6 
            data = np.loadtxt("/public1/home/zhanghan181/perl5/ensemble-members/EVO-energy/EVO-energy_cnop_"+str('%.2f' % delta)+"-fin3norm-init2norm-"+str(CASE4[case])+"-"+str(opti_time)+"h.txt",comments='*')#,header=None, delim_whitespace=True)
            data_6h =np.array(data)
            print(data_6h.shape)
        if opti == 2: 
            opti_time=12 
            data = np.loadtxt("/public1/home/zhanghan181/perl5/ensemble-members/EVO-energy/EVO-energy_cnop_"+str('%.2f' % delta)+"-fin3norm-init2norm-"+str(CASE4[case])+"-"+str(opti_time)+"h.txt",comments='*')#,header=None, delim_whitespace=True)
            data_12h =np.array(data)
            print(data_12h.shape)
        if opti == 3:
            opti_time=24
            data = np.loadtxt("/public1/home/zhanghan181/perl5/ensemble-members/EVO-energy/EVO-energy_cnop_"+str('%.2f' % delta)+"-fin3norm-init2norm-"+str(CASE4[case])+"-"+str(opti_time)+"h.txt",comments='*')#,header=None, delim_whitespace=True)
            data_24h =np.array(data)
            print(data_24h.shape)
        if opti == 4:
            opti_time=36 
            data = np.loadtxt("/public1/home/zhanghan181/perl5/ensemble-members/EVO-energy/EVO-energy_cnop_"+str('%.2f' % delta)+"-fin3norm-init2norm-"+str(CASE4[case])+"-"+str(opti_time)+"h.txt",comments='*')#,header=None, delim_whitespace=True)
            data_36h =np.array(data)
            #print(data_36h)
            print(data_36h.shape)
    print(opti_time)
    numcnop = np.arange(maxcnop)
    itime   = np.arange(wmaxtime[case])
    print(numcnop)
    print(itime)
    max_energy=int(max(map(max,data_36h))/10+1)*10
    print(max_energy)
    fig = plt.figure(case)
    ax1 = fig.add_subplot(2,2,1)#(1,4,1)
    plt.gca().set_prop_cycle(plt.cycler('color',plt.cm.jet_r(np.linspace(0, 1, 15))))
    for numcnop2 in range(1,maxcnop+1,2):
        df1 = plt.plot(itime ,data_6h[0:wmaxtime[case],numcnop2] ,marker='o',markersize=3, linewidth=0.5, label = 'CNOP'+str(numcnop2))
    plt.ylim(0,max_energy)  # 限定纵轴的范围
    plt.xlim(0,wmaxtime[case]-1)

    ax2 = fig.add_subplot(2,2,2)#(1,4,2)
    plt.gca().set_prop_cycle(plt.cycler('color',plt.cm.jet_r(np.linspace(0, 1, 15))))
    for numcnop2 in range(1,maxcnop+1,2):
        df2 = plt.plot(itime,data_12h[0:wmaxtime[case],numcnop2],marker='o',markersize=3,linewidth=0.5,label = 'CNOP'+str(numcnop2))
    plt.ylim(0,max_energy)  # 限定纵轴的范围
    plt.xlim(0,wmaxtime[case]-1)

    ax3 = fig.add_subplot(2,2,3)#(1,4,3)
    plt.gca().set_prop_cycle(plt.cycler('color',plt.cm.jet_r(np.linspace(0, 1, 15))))
    for numcnop2 in range(1,maxcnop+1,2):
        df3 = plt.plot(itime,data_24h[0:wmaxtime[case],numcnop2],marker='o',markersize=3,linewidth=0.5,label = 'CNOP'+str(numcnop2))
    plt.ylim(0,max_energy)  # 限定纵轴的范围
    plt.xlim(0,wmaxtime[case]-1)

    ax4 = fig.add_subplot(2,2,4)#(1,4,4)
    plt.gca().set_prop_cycle(plt.cycler('color',plt.cm.jet_r(np.linspace(0, 1, 15))))
    for numcnop2 in range(1,maxcnop+1,2):
        df4 = plt.plot(itime,data_36h[0:wmaxtime[case],numcnop2],marker='o',markersize=3,linewidth=0.5,label = 'CNOP'+str(numcnop2))
    plt.ylim(0,max_energy)  # 限定纵轴的范围
    plt.xlim(0,wmaxtime[case]-1)

    ax4.legend(loc=2, bbox_to_anchor=(-1.55,-0.2),borderaxespad = 0.,ncol=5)  ##设置ax4中legend的位置，将其放在图外
    plt.savefig("/public1/home/zhanghan181/perl5/ensemble-members/EVO-energy/plot-EVO-energy_cnop_"+str('%.2f' % delta)+"-fin3norm-init2norm-"+str(CASE4[case])+".eps", format='eps', dpi=1000, bbox_inches='tight')            
# for case in range(3,5,3): 
#     print(case)
#     print(CASE4[case])
#     for nc in range(mindel,maxdel+1): 
#         delta = nc*DEL
#         print(delta)
for case in range(0,5,3): 
    print(case)
    print(CASE4[case])
    for nc in range(mindel,maxdel+1): 
        delta = nc*DEL
        print(delta)
    for opti in range(min_opti_time,max_opti_time+1):
        if opti == 1: 
            opti_time=6 
            data = np.loadtxt("/public1/home/zhanghan181/perl5/ensemble-members/EVO-energy/EVO-energy_ncnop_"+str('%.2f' % delta)+"-fin3norm-init2norm-"+str(CASE4[case])+"-"+str(opti_time)+"h.txt",comments='*')#,header=None, delim_whitespace=True)
            data_6h =np.array(data)
            print(data_6h.shape)
        if opti == 2: 
            opti_time=12 
            data = np.loadtxt("/public1/home/zhanghan181/perl5/ensemble-members/EVO-energy/EVO-energy_ncnop_"+str('%.2f' % delta)+"-fin3norm-init2norm-"+str(CASE4[case])+"-"+str(opti_time)+"h.txt",comments='*')#,header=None, delim_whitespace=True)
            data_12h =np.array(data)
            print(data_12h.shape)
        if opti == 3:
            opti_time=24
            data = np.loadtxt("/public1/home/zhanghan181/perl5/ensemble-members/EVO-energy/EVO-energy_ncnop_"+str('%.2f' % delta)+"-fin3norm-init2norm-"+str(CASE4[case])+"-"+str(opti_time)+"h.txt",comments='*')#,header=None, delim_whitespace=True)
            data_24h =np.array(data)
            print(data_24h.shape)
        if opti == 4:
            opti_time=36 
            data = np.loadtxt("/public1/home/zhanghan181/perl5/ensemble-members/EVO-energy/EVO-energy_ncnop_"+str('%.2f' % delta)+"-fin3norm-init2norm-"+str(CASE4[case])+"-"+str(opti_time)+"h.txt",comments='*')#,header=None, delim_whitespace=True)
            data_36h =np.array(data)
            #print(data_36h)
            print(data_36h.shape)
    print(opti_time)
    numcnop = np.arange(maxcnop)
    itime   = np.arange(wmaxtime[case])
    print(numcnop)
    print(itime)
    max_energy=int(max(map(max,data_36h))/10+1)*10
    print(max_energy)
    fig = plt.figure(case+5)
    ax1 = fig.add_subplot(2,2,1)#(1,4,1)
    plt.gca().set_prop_cycle(plt.cycler('color',plt.cm.jet_r(np.linspace(0, 1, 15))))
    for numcnop2 in range(1,maxcnop+1,2):
        df1 = plt.plot(itime ,data_6h[0:wmaxtime[case],numcnop2] ,marker='o',markersize=3, linewidth=0.5, label ='- CNOP'+str(numcnop2))
    plt.ylim(0,max_energy)  # 限定纵轴的范围
    plt.xlim(0,wmaxtime[case]-1)

    ax2 = fig.add_subplot(2,2,2)#(1,4,2)
    plt.gca().set_prop_cycle(plt.cycler('color',plt.cm.jet_r(np.linspace(0, 1, 15))))
    for numcnop2 in range(1,maxcnop+1,2):
        df2 = plt.plot(itime,data_12h[0:wmaxtime[case],numcnop2],marker='o',markersize=3,linewidth=0.5,label = '- CNOP'+str(numcnop2))
    plt.ylim(0,max_energy)  # 限定纵轴的范围
    plt.xlim(0,wmaxtime[case]-1)

    ax3 = fig.add_subplot(2,2,3)#(1,4,3)
    plt.gca().set_prop_cycle(plt.cycler('color',plt.cm.jet_r(np.linspace(0, 1, 15))))
    for numcnop2 in range(1,maxcnop+1,2):
        df3 = plt.plot(itime,data_24h[0:wmaxtime[case],numcnop2],marker='o',markersize=3,linewidth=0.5,label = '- CNOP'+str(numcnop2))
    plt.ylim(0,max_energy)  # 限定纵轴的范围
    plt.xlim(0,wmaxtime[case]-1)

    ax4 = fig.add_subplot(2,2,4)#(1,4,4)
    plt.gca().set_prop_cycle(plt.cycler('color',plt.cm.jet_r(np.linspace(0, 1, 15))))
    for numcnop2 in range(1,maxcnop+1,2):
        df4 = plt.plot(itime,data_36h[0:wmaxtime[case],numcnop2],marker='o',markersize=3,linewidth=0.5,label = '- CNOP'+str(numcnop2))
    plt.ylim(0,max_energy)  # 限定纵轴的范围
    plt.xlim(0,wmaxtime[case]-1)

    ax4.legend(loc=2, bbox_to_anchor=(-1.55,-0.2),borderaxespad = 0.,ncol=5)  ##设置ax4中legend的位置，将其放在图外
    plt.savefig("/public1/home/zhanghan181/perl5/ensemble-members/EVO-energy/plot-EVO-energy_ncnop_"+str('%.2f' % delta)+"-fin3norm-init2norm-"+str(CASE4[case])+".eps", format='eps', dpi=1000, bbox_inches='tight')            

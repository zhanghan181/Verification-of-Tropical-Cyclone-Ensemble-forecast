begin
filename = "parameter"
CASE = keyword_values( filename, "CASE", "string")  
NDIX = keyword_values( filename, "NDIX", "integer")  
NDJX = keyword_values( filename, "NDJX", "integer") 
NNDKX = keyword_values( filename, "NNDKX", "integer")  
BDY = keyword_values( filename, "BDY", "integer") 
ilonstart = keyword_values( filename, "ilonstart", "integer")  
ilonend = keyword_values( filename, "ilonend", "integer") 
jlatstart = keyword_values( filename, "jlatstart", "integer")  
jlatend = keyword_values( filename, "jlatend", "integer")
opti_time = keyword_values( filename, "opti_time", "integer") 
init_norm = keyword_values( filename, "init_norm", "integer")  
init_time = keyword_values( filename, "init_time", "integer")  
fin_norm = keyword_values( filename, "fin_norm", "integer")
DEL = keyword_values( filename, "DEL", "float")
mincnop = keyword_values( filename, "mincnop", "integer")  
maxcnop = keyword_values( filename, "maxcnop", "integer")
sfc = keyword_values( filename, "sfc", "integer")  
mininival = keyword_values( filename, "mininival", "integer")
maxinival = keyword_values( filename, "maxinival", "integer")
 mindel   = keyword_values( filename, "mindel", "integer")
 maxdel   = keyword_values( filename, "maxdel", "integer")
 min_opti_time    =   keyword_values( filename, "min_opti_time", "integer")
 max_opti_time    =   keyword_values( filename, "max_opti_time", "integer")
EPS = keyword_values( filename, "EPS", "float")
MAXIT = keyword_values( filename, "MAXIT", "integer")
MMAX = keyword_values( filename, "MMAX", "integer")
NDKX = 7*NNDKX+2
energyn  = new((/3,21/),double)
energy_2 = new((/21,maxcnop/),double)
sumslp   = new((/21/),double)
meanslp  = new((/21,2/),double)
verilon  = new((/5/),double)
verilat  = new((/5/),double)
W        = new((/7/),float)
;*********parameters-1************
CASE4     = (/"Matsa","Fung-wong","Maria","Haishen"/)
init_time4= (/"2005-08-03_12:00:00","2014-09-20_00:00:00","2018-07-06_12:00:00","2020-09-04_00:00:00"/)
it=0
cp        = 1004.6
Tr        = 273.15
wq        = 1
L         = 2510400
pressure  = 500
pressure1 = 850
Rd        = 287.04
 W(0)=1.0
 W(1)=1.0
 W(2)=cp/Tr
 W(3)=L*wq/(cp*Tr)
 ;W(5)=1/Rd/Tr
 W(5)=0
 W(6)=0
;**********************************

;**********************************
do opti=min_opti_time,max_opti_time
     if(opti.eq.1) then
       opti_time=6 
     end if
     if(opti.eq.2) then
       opti_time=12 
     end if
     if(opti.eq.3) then
       opti_time=24 
     end if
     if(opti.eq.4) then
       opti_time=36 
     end if
do nc=mindel,maxdel
     delta=nc*DEL  
     DEL1=sprintf("%.2f",delta)
     print(DEL1)

do case=0,3,3

do k=mincnop,maxcnop
kcnop = k
a = addfile("/public1/home/zhanghan181/perl5/ensemble-members/wrfout-control/wrfout_d01_"+init_time4(case),"r")
b = addfile("/public1/home/zhanghan181/perl5/ensemble-members/"+CASE4(case)+"/Twrfout_"+DEL1+"-"+k+"-fin3norm-init2norm-"+CASE4(case)+"-"+opti_time+"hd01_"+init_time4(case),"r")
;*********************************
pdfname   = "EVO-energy-"+DEL1+"-"+kcnop+"-fin"+fin_norm+"norm-init"+init_norm+"norm-"+CASE4(case)+"-"+opti_time+"h"
wks       = gsn_open_wks("pdf",pdfname)
cmap      = read_colormap_file("rainbow");BlueWhiteOrangeRed;precip3_16lev
cmap(0,:) = (/0,0,0,0/)    ; make first color fully transparent
plots     = new((/41,21/),graphic)
eTCline   = new((/41,21,4/),graphic)
;*********************************
times    = wrf_user_getvar(b,"times",-1) 
wmaxtime = dimsizes(times)
max_u0  =new(17,double)
WMax   = new(41,double)
imin   = new(41,integer)
jmin   = new(41,integer)
iimin   = new(41,integer)
jjmin   = new(41,integer)
tcklon  = new(41,double)
tcklat  = new(41,double)
bTCtext = new((/41/),graphic)
print(wmaxtime)
do it=0,wmaxtime-1
it2=it+0
res = True
QV   = wrf_user_getvar(a,"QVAPOR",it)             
MU1  = wrf_user_getvar(a,"MU",it)
slp1 = wrf_user_getvar(a,"slp",it)
u101 = wrf_user_getvar(a,"U10",it)                       ; Get U10
v101 = wrf_user_getvar(a,"V10",it)                       ; Get V10
u1   = wrf_user_getvar(a,"ua",it)        ; u averaged to mass points
v1   = wrf_user_getvar(a,"va",it)        ; v averaged to mass points
t1   = wrf_user_getvar(a,"T",it)
w1   = wrf_user_getvar(a,"W",it) 
PH1   = wrf_user_getvar(a,"PH",it)

u2   = wrf_user_getvar(b,"ua",it2)        ; u averaged to mass points
v2   = wrf_user_getvar(b,"va",it2)        ; v averaged to mass points
w2   = wrf_user_getvar(b,"W",it2) 
t2   = wrf_user_getvar(b,"T",it2)   
QV2  = wrf_user_getvar(b,"QVAPOR",it2)
MU2  = wrf_user_getvar(b,"MU",it2)
PH2  = wrf_user_getvar(b,"PH",it2)

hgt  = wrf_user_getvar(a,"HGT",it)
slp2 = wrf_user_getvar(b,"slp",it2)
u102 = wrf_user_getvar(b,"U10",it2)                       ; Get U10
v102 = wrf_user_getvar(b,"V10",it2)                       ; Get V10


p    = wrf_user_getvar(a, "pressure",it)
z    = wrf_user_getvar(b,"z",it)
slp  = wrf_user_getvar(b,"slp",it2)
t    = t2-t1
QV   = QV2-QV
MU   = MU2-MU1
PH   = PH2-PH1
u    = u2-u1
v    = v2-v1
ww   = w2-w1

energy1                  = ww*ww
energy1(0:NNDKX-1,:,:)   = u*u+v*v+W(2)*t*t+W(3)*QV*QV+W(5)*PH(0:NNDKX-1,:,:)*PH(0:NNDKX-1,:,:)+W(6)*ww(0:NNDKX-1,:,:)*ww(0:NNDKX-1,:,:); total moist energy in grid 
energy1(NNDKX,:,:)       = W(5)*PH(NNDKX,:,:)*PH(NNDKX,:,:)+W(6)*ww(NNDKX,:,:)*ww(NNDKX,:,:)
energy_1                 = dim_sum_n_Wrap(energy1, 0)
printVarSummary(energy1)
energy_2(it,kcnop-1)     = avg(energy1(:,:,:))
energy2                  = u*u+v*v+W(2)*t*t
energy3                  = u*u+v*v
energy_3                 = dim_sum_n_Wrap(energy3, 0)
energy1_2                = dim_sum_n_Wrap(energy1, 1)
energy1_3                = dim_sum_n_Wrap(energy1_2, 1)

t_plane  = wrf_user_intrp3d( t,p,"h",pressure,0.,False)
u_plane  = wrf_user_intrp3d( u,p,"h",pressure1,0.,False)
v_plane  = wrf_user_intrp3d( v,p,"h",pressure1,0.,False)
qv_plane = wrf_user_intrp3d(QV,p,"h",pressure,0.,False)
z_plane  = wrf_user_intrp3d( z,p,"h",pressure,0.,False)

u0       = energy_1
ua1      = u_plane
va1      = v_plane

maxu0    = 3*max(abs(u0))/4
maxu0    = toint(maxu0)
speed    = sqrt(ua1^2+va1^2)
maxspeed = max(abs(speed))
maxspeed = toint(maxspeed)
if (maxspeed.eq.0) then
maxspeed=maxspeed+1
else
maxspeed=maxspeed
end if
if (maxu0.eq.0) then
maxu0=maxu0+1
else
maxu0=maxu0
end if
printMinMax(u0,0)
printMinMax(speed,0) 

res@NoHeaderFooter = True
res@gsnMaximize    = False
res@gsnDraw        = False 
res@gsnFrame       = False
res@NoTitles       = True
mpres              = res
mpres              = wrf_map_resources(a,mpres)
mpres@mpGridLineThicknessF  = 0.5
mpres@mpGridLineDashPattern = 2
mpres@mpGridSpacingF        = 10
mpres@mpGridLineThicknessF  = 0.5
mpres@mpGridLineDashPattern = 2
mpres@mpGridSpacingF        = 10
mpres@Footer =  False
mpres@ZoomIn = False        ; Tell wrf_map_resources we want to zoom in.
mpres@Xstart = jmin(0)-30 ; Set these four special WRF resources
mpres@Xend   = jmin(0)+30    ; required for zooming.
mpres@Ystart = imin(0)-30
mpres@Yend   = imin(0)+30


pltres = res
pltres@PanelPlot = True
pltres@NoTitles = True        ; Turn off top left title just above plot
u_res=res
u_res@cnFillOn=True
u_res@cnFillPalette        = cmap 
u_res@cnLevelSelectionMode = "ManualLevels"
u_res@cnMaxLevelValF =maxu0
u_res@cnMinLevelValF =maxu0/15
u_res@cnLevelSpacingF =maxu0/15
u_res@lbLabelBarOn=True
u_res@cnLineLabelsOn = False

contour_u=wrf_contour(a,wks,u0,u_res)
           
opts = res      
opts@cnFillOn = False 
opts@cnLinesOn = True
opts@ContourParameters = (/ 980,1000,5 /)
;opts@ContourParameters = (/ 9150,9800,50 /) 
opts@cnLineLabelBackgroundColor="White"
opts@cnLowLabelPerimColor="White"
opts@Footer = False
opts@gsnSpreadColorEnd=-3
opts@cnInfoLabelOrthogonalPosF = -0.115
opts@cnLineColor = "red"
opts@cnHighLabelsOn = False
opts@cnLowLabelsOn = True
opts@cnLineLabelsOn = True
opts@gsnContourLineThicknessesScale = 1.0
opts@tiMainOn=True
opts@tiMainFontHeightF=0

contour_z = wrf_contour(a,wks,slp,opts)

wopts=res
wopts@FieldTitle = "Winds"             ; Overwrite the field title
wopts@vcRefAnnoOn=True
wopts@vcRefMagnitudeF= 15
wopts@vcRefLengthF =0.020 
wopts@vcMinDistanceF          = 0.02 
wopts@NumVectors = 30                    ; Density of wind barbs
wopts@vcRefLengthF = 0.02 ;
wopts@vcGlyphStyle="CurlyVector";"CurlyVector";"WindBard"
wopts@vcFillArrowHeadXF=0.4;箭头
wopts@vcFillArrowHeadYF=0.1
wopts@vcFillArrowHeadInteriorXF = 0.25
wopts@vcFillArrowEdgeThicknessF = 4
wopts@vcFillArrowEdgeColor="white"
wopts@vcFillArrowFillColor="black"
wopts@vcRefAnnoString1On= True
wopts@vcRefAnnoString2On= False
wopts@vcRefAnnoOrthogonalPosF=-0.2
wopts@vcRefAnnoPerimOn=False

vector = wrf_vector(a,wks,ua1,va1,wopts)

gsres = True
gsres@gsMarkerSizeF=0.08
gsres@gsMarkerIndex = 15
gsres@gsMarkerColor  = "GREEN"
plots(kcnop,it)=wrf_map_overlays(a,wks,(/vector,contour_u,contour_z/),pltres,mpres)
control                   = asciiread("/public1/home/zhanghan181/perl5/addcnop-masta/3v-0.06-addcnop-plus/exp0/Matsaexp_0.txt", (/1,4/), "double");60km
;bTCtext(it)               = gsn_add_polymarker(wks, plots(kcnop,it), tcklon(it),tcklat(it), gsres)


; eres=True
; eres@gsLineThicknessF      = 3.0
; eres@gsLineColor           = "black"
; verilon(0)=a->XLAT(1,sen_minlat, sen_minlon);sen_minlat:sen_maxlat,sen_minlon:sen_maxlon
; verilon(2)=a->XLAT(1,sen_maxlat, sen_maxlon)
; verilon(3)=a->XLAT(1,sen_maxlat, sen_minlon)
; verilon(1)=a->XLAT(1,sen_minlat, sen_maxlon)
; verilon(4)=a->XLAT(1,sen_minlat, sen_minlon)

; verilat(0)=a->XLONG(1,sen_minlat, sen_minlon)
; verilat(2)=a->XLONG(1,sen_maxlat, sen_maxlon)
; verilat(3)=a->XLONG(1,sen_maxlat, sen_minlon)
; verilat(1)=a->XLONG(1,sen_minlat, en_maxlon)
; verilat(4)=a->XLONG(1,sen_minlat, sen_minlon)

; eTCline(kcnop,it,0)=  gsn_add_polyline(wks, plots(kcnop,it),verilat(0:4),verilon(0:4),  eres)
end do
; ;---------------------------------------------------------------
pnlres=True
pnlres@gsnPanelMainString  = "Higos"
pnlres@gsnPanelMainFont ="helvetica"
pnlres@gsnPanelMainFontHeightF=0.02
pnlres@gsnPanelScalePlotIndex = 2
pnlres@gsnPanelCenter = True
pnlres@gsnPanelDebug = True
pnlres@gsnMaximize  = True
pnlres@amJust="TopLeft"
pnlres@gsnPanelFigureStringsBackgroundFillColor="white"
pnlres@gsnPanelFigureStringsFontHeightF = 0.005
pnlres@gsnPanelFigureStringsPerimOn=True
pnlres@gsnPanelLabelBar = False
;pnlres@cnFillOpacityF = 0.5
pnlres@gsnPanelBottom = 0.0
pnlres@gsnPanelLeft = 0.0
pnlres@gsnPanelRight = 1.0
pnlres@gsnPanelRowSpec = True
pnlres@gsnPanelTop = 1.0
pnlres@gsnPanelXWhiteSpacePercent = 2.
pnlres@gsnPanelYWhiteSpacePercent = 5.
pnlres@gsnPanelFigureStrings=(/"00","12h","24h","36h"/)
gsn_panel(wks,(/plots(kcnop,0),plots(kcnop,2),plots(kcnop,4),plots(kcnop,6)/),(/2,2/),pnlres)
pnlres@gsnPanelFigureStrings=(/"sen_maxlath","60h","72h","84h"/)
gsn_panel(wks,(/plots(kcnop,8),plots(kcnop,10),plots(kcnop,12),plots(kcnop,14)/),(/2,2/),pnlres)
pnlres@gsnPanelFigureStrings=(/"sen_maxlonh","108h","120h","126h"/)
gsn_panel(wks,(/plots(kcnop,16),plots(kcnop,18),plots(kcnop,20)/),(/2,2/),pnlres)
end do
  fmtf     = "41f30.16"
  opt      = True
  opt@fout = "EVO-energy_cnop_"+DEL1+"-fin3norm-init2norm-"+CASE4(case)+"-"+opti_time+"h.txt"
  write_matrix(energy_2,fmtf,opt)
delete(times)
do k=mincnop,maxcnop
kcnop = k
a = addfile("/public1/home/zhanghan181/perl5/ensemble-members/wrfout-control/wrfout_d01_"+init_time4(case),"r")
b = addfile("/public1/home/zhanghan181/perl5/ensemble-members/"+CASE4(case)+"/Tncnop_"+DEL1+"-"+k+"-fin3norm-init2norm-"+CASE4(case)+"-"+opti_time+"hd01_"+init_time4(case),"r")
;*********************************
pdfname   = "EVO-energy-ncnop-"+DEL1+"-"+kcnop+"-fin"+fin_norm+"norm-init"+init_norm+"norm-"+CASE4(case)+"-"+opti_time+"h"
wks       = gsn_open_wks("pdf",pdfname)
cmap      = read_colormap_file("rainbow");BlueWhiteOrangeRed;precip3_16lev
cmap(0,:) = (/0,0,0,0/)    ; make first color fully transparent
plots     = new((/41,21/),graphic)
eTCline   = new((/41,21,4/),graphic)
;*********************************
times    = wrf_user_getvar(b,"times",-1) 
wmaxtime = dimsizes(times)
max_u0  =new(17,double)
WMax   = new(41,double)
imin   = new(41,integer)
jmin   = new(41,integer)
iimin   = new(41,integer)
jjmin   = new(41,integer)
tcklon  = new(41,double)
tcklat  = new(41,double)
bTCtext = new((/41/),graphic)
print(wmaxtime)
do it=0,wmaxtime-1
it2=it+0
res = True
QV   = wrf_user_getvar(a,"QVAPOR",it)             
MU1  = wrf_user_getvar(a,"MU",it)
slp1 = wrf_user_getvar(a,"slp",it)
u101 = wrf_user_getvar(a,"U10",it)                       ; Get U10
v101 = wrf_user_getvar(a,"V10",it)                       ; Get V10
u1   = wrf_user_getvar(a,"ua",it)        ; u averaged to mass points
v1   = wrf_user_getvar(a,"va",it)        ; v averaged to mass points
t1   = wrf_user_getvar(a,"T",it)
w1   = wrf_user_getvar(a,"W",it) 
PH1   = wrf_user_getvar(a,"PH",it)

u2   = wrf_user_getvar(b,"ua",it2)        ; u averaged to mass points
v2   = wrf_user_getvar(b,"va",it2)        ; v averaged to mass points
w2   = wrf_user_getvar(b,"W",it2) 
t2   = wrf_user_getvar(b,"T",it2)   
QV2  = wrf_user_getvar(b,"QVAPOR",it2)
MU2  = wrf_user_getvar(b,"MU",it2)
PH2  = wrf_user_getvar(b,"PH",it2)

hgt  = wrf_user_getvar(a,"HGT",it)
slp2 = wrf_user_getvar(b,"slp",it2)
u102 = wrf_user_getvar(b,"U10",it2)                       ; Get U10
v102 = wrf_user_getvar(b,"V10",it2)                       ; Get V10


p    = wrf_user_getvar(a, "pressure",it)
z    = wrf_user_getvar(b,"z",it)
slp  = wrf_user_getvar(b,"slp",it2)
t    = t2-t1
QV   = QV2-QV
MU   = MU2-MU1
PH   = PH2-PH1
u    = u2-u1
v    = v2-v1
ww   = w2-w1

energy1                  = ww*ww
energy1(0:NNDKX-1,:,:)   = u*u+v*v+W(2)*t*t+W(3)*QV*QV+W(5)*PH(0:NNDKX-1,:,:)*PH(0:NNDKX-1,:,:)+W(6)*ww(0:NNDKX-1,:,:)*ww(0:NNDKX-1,:,:); total moist energy in grid 
energy1(NNDKX,:,:)       = W(5)*PH(NNDKX,:,:)*PH(NNDKX,:,:)+W(6)*ww(NNDKX,:,:)*ww(NNDKX,:,:)
energy_1                 = dim_sum_n_Wrap(energy1, 0)
printVarSummary(energy1)
energy_2(it,kcnop-1)     = avg(energy1(:,:,:))
energy2                  = u*u+v*v+W(2)*t*t
energy3                  = u*u+v*v
energy_3                 = dim_sum_n_Wrap(energy3, 0)
energy1_2                = dim_sum_n_Wrap(energy1, 1)
energy1_3                = dim_sum_n_Wrap(energy1_2, 1)

t_plane  = wrf_user_intrp3d( t,p,"h",pressure,0.,False)
u_plane  = wrf_user_intrp3d( u,p,"h",pressure1,0.,False)
v_plane  = wrf_user_intrp3d( v,p,"h",pressure1,0.,False)
qv_plane = wrf_user_intrp3d(QV,p,"h",pressure,0.,False)
z_plane  = wrf_user_intrp3d( z,p,"h",pressure,0.,False)

u0       = energy_1
ua1      = u_plane
va1      = v_plane

maxu0    = 3*max(abs(u0))/4
maxu0    = toint(maxu0)
speed    = sqrt(ua1^2+va1^2)
maxspeed = max(abs(speed))
maxspeed = toint(maxspeed)
if (maxspeed.eq.0) then
maxspeed=maxspeed+1
else
maxspeed=maxspeed
end if
if (maxu0.eq.0) then
maxu0=maxu0+1
else
maxu0=maxu0
end if
printMinMax(u0,0)
printMinMax(speed,0) 

res@NoHeaderFooter = True
res@gsnMaximize    = False
res@gsnDraw        = False 
res@gsnFrame       = False
res@NoTitles       = True
mpres              = res
mpres              = wrf_map_resources(a,mpres)
mpres@mpGridLineThicknessF  = 0.5
mpres@mpGridLineDashPattern = 2
mpres@mpGridSpacingF        = 10
mpres@mpGridLineThicknessF  = 0.5
mpres@mpGridLineDashPattern = 2
mpres@mpGridSpacingF        = 10
mpres@Footer =  False
mpres@ZoomIn = False        ; Tell wrf_map_resources we want to zoom in.
mpres@Xstart = jmin(0)-30 ; Set these four special WRF resources
mpres@Xend   = jmin(0)+30    ; required for zooming.
mpres@Ystart = imin(0)-30
mpres@Yend   = imin(0)+30


pltres = res
pltres@PanelPlot = True
pltres@NoTitles = True        ; Turn off top left title just above plot
u_res=res
u_res@cnFillOn=True
u_res@cnFillPalette        = cmap 
u_res@cnLevelSelectionMode = "ManualLevels"
u_res@cnMaxLevelValF =maxu0
u_res@cnMinLevelValF =maxu0/15
u_res@cnLevelSpacingF =maxu0/15
u_res@lbLabelBarOn=True
u_res@cnLineLabelsOn = False

contour_u=wrf_contour(a,wks,u0,u_res)
           
opts = res      
opts@cnFillOn = False 
opts@cnLinesOn = True
opts@ContourParameters = (/ 980,1000,5 /)
;opts@ContourParameters = (/ 9150,9800,50 /) 
opts@cnLineLabelBackgroundColor="White"
opts@cnLowLabelPerimColor="White"
opts@Footer = False
opts@gsnSpreadColorEnd=-3
opts@cnInfoLabelOrthogonalPosF = -0.115
opts@cnLineColor = "red"
opts@cnHighLabelsOn = False
opts@cnLowLabelsOn = True
opts@cnLineLabelsOn = True
opts@gsnContourLineThicknessesScale = 1.0
opts@tiMainOn=True
opts@tiMainFontHeightF=0

contour_z = wrf_contour(a,wks,slp,opts)

wopts=res
wopts@FieldTitle = "Winds"             ; Overwrite the field title
wopts@vcRefAnnoOn=True
wopts@vcRefMagnitudeF= 15
wopts@vcRefLengthF =0.020 
wopts@vcMinDistanceF          = 0.02 
wopts@NumVectors = 30                    ; Density of wind barbs
wopts@vcRefLengthF = 0.02 ;
wopts@vcGlyphStyle="CurlyVector";"CurlyVector";"WindBard"
wopts@vcFillArrowHeadXF=0.4;箭头
wopts@vcFillArrowHeadYF=0.1
wopts@vcFillArrowHeadInteriorXF = 0.25
wopts@vcFillArrowEdgeThicknessF = 4
wopts@vcFillArrowEdgeColor="white"
wopts@vcFillArrowFillColor="black"
wopts@vcRefAnnoString1On= True
wopts@vcRefAnnoString2On= False
wopts@vcRefAnnoOrthogonalPosF=-0.2
wopts@vcRefAnnoPerimOn=False

vector = wrf_vector(a,wks,ua1,va1,wopts)

gsres = True
gsres@gsMarkerSizeF=0.08
gsres@gsMarkerIndex = 15
gsres@gsMarkerColor  = "GREEN"
plots(kcnop,it)=wrf_map_overlays(a,wks,(/vector,contour_u,contour_z/),pltres,mpres)
control                   = asciiread("/public1/home/zhanghan181/perl5/addcnop-masta/3v-0.06-addcnop-plus/exp0/Matsaexp_0.txt", (/1,4/), "double");60km
;bTCtext(it)               = gsn_add_polymarker(wks, plots(kcnop,it), tcklon(it),tcklat(it), gsres)


; eres=True
; eres@gsLineThicknessF      = 3.0
; eres@gsLineColor           = "black"
; verilon(0)=a->XLAT(1,sen_minlat, sen_minlon);sen_minlat:sen_maxlat,sen_minlon:sen_maxlon
; verilon(2)=a->XLAT(1,sen_maxlat, sen_maxlon)
; verilon(3)=a->XLAT(1,sen_maxlat, sen_minlon)
; verilon(1)=a->XLAT(1,sen_minlat, sen_maxlon)
; verilon(4)=a->XLAT(1,sen_minlat, sen_minlon)

; verilat(0)=a->XLONG(1,sen_minlat, sen_minlon)
; verilat(2)=a->XLONG(1,sen_maxlat, sen_maxlon)
; verilat(3)=a->XLONG(1,sen_maxlat, sen_minlon)
; verilat(1)=a->XLONG(1,sen_minlat, en_maxlon)
; verilat(4)=a->XLONG(1,sen_minlat, sen_minlon)

; eTCline(kcnop,it,0)=  gsn_add_polyline(wks, plots(kcnop,it),verilat(0:4),verilon(0:4),  eres)
end do
; ;---------------------------------------------------------------
pnlres=True
pnlres@gsnPanelMainString  = "Higos"
pnlres@gsnPanelMainFont ="helvetica"
pnlres@gsnPanelMainFontHeightF=0.02
pnlres@gsnPanelScalePlotIndex = 2
pnlres@gsnPanelCenter = True
pnlres@gsnPanelDebug = True
pnlres@gsnMaximize  = True
pnlres@amJust="TopLeft"
pnlres@gsnPanelFigureStringsBackgroundFillColor="white"
pnlres@gsnPanelFigureStringsFontHeightF = 0.005
pnlres@gsnPanelFigureStringsPerimOn=True
pnlres@gsnPanelLabelBar = False
;pnlres@cnFillOpacityF = 0.5
pnlres@gsnPanelBottom = 0.0
pnlres@gsnPanelLeft = 0.0
pnlres@gsnPanelRight = 1.0
pnlres@gsnPanelRowSpec = True
pnlres@gsnPanelTop = 1.0
pnlres@gsnPanelXWhiteSpacePercent = 2.
pnlres@gsnPanelYWhiteSpacePercent = 5.
pnlres@gsnPanelFigureStrings=(/"00","12h","24h","36h"/)
gsn_panel(wks,(/plots(kcnop,0),plots(kcnop,2),plots(kcnop,4),plots(kcnop,6)/),(/2,2/),pnlres)
pnlres@gsnPanelFigureStrings=(/"sen_maxlath","60h","72h","84h"/)
gsn_panel(wks,(/plots(kcnop,8),plots(kcnop,10),plots(kcnop,12),plots(kcnop,14)/),(/2,2/),pnlres)
pnlres@gsnPanelFigureStrings=(/"sen_maxlonh","108h","120h","126h"/)
gsn_panel(wks,(/plots(kcnop,16),plots(kcnop,18),plots(kcnop,20)/),(/2,2/),pnlres)
end do
  fmtf     = "41f30.16"
  opt      = True
  opt@fout = "EVO-energy_ncnop_"+DEL1+"-fin3norm-init2norm-"+CASE4(case)+"-"+opti_time+"h.txt"
  write_matrix(energy_2,fmtf,opt)
delete (times)
end do
end do
end do
end

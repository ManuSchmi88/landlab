#Import necessesary Python and Landlab Modules
import numpy as np
from landlab import RasterModelGrid
from landlab import CLOSED_BOUNDARY, FIXED_VALUE_BOUNDARY
from landlab.components import FlowRouter
from landlab.components import LinearDiffuser
#from landlab.components import FastscapeEroder
from landlab.components import StreamPowerEroder
from landlab.components import drainage_density
from landlab import imshow_grid
from landlab.io.netcdf import write_netcdf
from landlab.io.netcdf import read_netcdf
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import time

#Build up Grid and define variables

#-----------DOMAIN---------------#

ncols = 301
nrows = 301
dx    = 100

#------------TIME----------------#

#runtime
total_T1 = 1e5  #yrs.
#timestep
dt = 100        #yrs
#number of timesteps and fillfactor for ffmpeg printout
nt = total_T1/dt
#time-vector, mostly for plotting purposes
timeVec = np.arange(0,total_T1,dt)
#Set the interval at which we create output. Warning. High output frequency
#will slow down the script s ignificantly
oi = 5000
no = total_T1/oi
zp = len(str(int(no)))

#------------UPLIFT--------------#

uplift_rate = 1e-4 #m/yr = 0.0001m/yr = 1mm/yr
uplift_per_step = uplift_rate * dt

#-------------EROSION------------#

Ksp = 1e-7
msp  = 0.6
nsp  = 1.0
ldib   = 1e-5
#time
elapsed_time = 0

crit_area = 1e5

#---------Vegetation-------------#

AqDens = 1000.0 #Density of Water [Kg/m^3]
grav   = 9.81   #Gravitational Acceleration [N/Kg]
n_soil = 0.025  #Mannings roughness for soil [-]
n_VRef = 0.2    #Mannings Reference Roughness for Vegi [-]
v_ref  = 1.0    #Reference Vegetation Density
w      = 1.    #Some scaling factor for vegetation [-?]

#---------VARIABLE INITIAION-----#

dhdtA    = [] #Vector containing dhdt values for each node per timestep
meandhdt = [] #contains mean elevation change per timestep
meanE    = [] #contains the mean "erosion" rate out of Massbalance
mean_hill_E = [] #contains mean hillslope erosion rate
mean_riv_E  = [] #contains mean river erosion rate
mean_dd = [] #contains mean drainage density
mean_K_riv  = [] #contains mean K-value for spl
mean_K_diff = [] #contains mean K-value for ld
mean_slope  = [] #mean slope within model-area
max_slope   = [] #maximum slope within model area
min_slope   = [] #minimum slope within model area
mean_elev   = [] #mean elevation within model area
max_elev    = [] #maximum elevation within model area
min_elev    = [] #minimum elevation within model area

print("Finished variable initiation.")


#------MODELGRID, CONDITIONS-----#
#NETCDF-INPUT Reader (comment if not used)
#mg = read_netcdf('This could be your inputfile.nc')
#INITIALIZE LANDLAB COMPONENTGRID
mg = RasterModelGrid((nrows,ncols),dx)
z  = mg.add_ones('node','topographic__elevation')
ir = np.random.rand(z.size)/1000 #builds up initial roughness
z += ir #adds initial roughness to grid

#SET UP BOUNDARY CONDITIONS
for edge in (mg.nodes_at_left_edge,mg.nodes_at_right_edge, mg.nodes_at_top_edge):
    mg.status_at_node[edge] = CLOSED_BOUNDARY
for edge in (mg.nodes_at_bottom_edge):
    mg.status_at_node[edge] = FIXED_VALUE_BOUNDARY

print("Finished setting up Grid and establishing boundary conditions")

#Set up vegetation__density field
vegi_perc = mg.zeros('node',dtype=float)
#vegi_trace_x = mg.x_of_node / 2
#more_vegi = np.where(mg.x_of_node >= vegi_trace_x)
#less_vegi = np.where(mg.x_of_node < vegi_trace_x)
#vegi_perc[less_vegi] += 0.4
#vegi_perc[more_vegi] += 0.4
vegi_test_timeseries = (np.sin(0.00015*timeVec)+1)/2

#Do the K-field vegetation-dependend calculations
#Calculations after Istanbulluoglu
vegi_perc += np.random.rand(z.size)/100
vegi_perc += vegi_test_timeseries[int(elapsed_time/dt)-1]
vegi_perc.clip(0.,1.)

#nSoil_to_15 = np.power(n_soil, 1.5)
#Ford = AqDens * grav * nSoil_to_15
n_v_frac = n_soil + (n_VRef*(vegi_perc/v_ref)) #self.vd = VARIABLE!
n_v_frac_to_w = np.power(n_v_frac, w)
Prefect = np.power(n_v_frac_to_w, 0.9)
Kv = Ksp * Ford/Prefect

#Set up K-field for StreamPowerEroder
Kfield = mg.zeros('node',dtype = float)
Kfield = Kv
#Kfield[np.where(mg.x_of_node >= vegi_trace_x)] = 5e-3

#Set up linear diffusivity field
lin_diff = mg.zeros('node', dtype = float)
lin_diff = ldib*np.exp(-vegi_perc)

print("Finished setting up the vegetation field and K and LD fields.")

#Create Threshold_sp field CURRENTLY NOT WORKING!
threshold_arr  = 2
#threshold_arr += 3e-5
#threshold_arr[np.where(mg.x_of_node >= 30000)] += 3e-5
#threshold_field = mg.add_field('node','threshold_sp',threshold_arr,noclobber = False)
#imshow_grid(mg,'threshold_sp')
#plt.title('Stream-Power Threshold')
#plt.savefig('Distribution of SP_Threshold',dpi=720)

#Initialize the erosional components
fr  = FlowRouter(mg)
ld  = LinearDiffuser(mg,linear_diffusivity=lin_diff)
#fc  = FastscapeEroder(mg,K_sp = Ksp1,m_sp=msp, n_sp=nsp, threshold_sp=threshold_arr)
sp  = StreamPowerEroder(mg,K_sp = Kfield,m_sp=msp, n_sp=nsp, threshold_sp = threshold_arr)
#-------------RUNTIME------------#

#Main Loop 1 (After first sucess is confirmed this is all moved in a class....)
t0 = time.time()
print("finished initiation of eroding components. starting loop...")

while elapsed_time < total_T1:

    #create copy of "old" topography
    z0 = mg.at_node['topographic__elevation'][mg.core_nodes].copy()

    #Call the erosion routines.
    fr.route_flow()
    ld.run_one_step(dt=dt)
    sp.run_one_step(dt=dt)
    mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_rate * dt #add uplift

    #calculate drainage_density
    channel_mask = mg.at_node['drainage_area'] > crit_area
    dd = drainage_density.DrainageDensity(mg, channel__mask = channel_mask)
    mean_dd.append(dd.calc_drainage_density())

    #Calculate dhdt and E
    dh = (mg.at_node['topographic__elevation'][mg.core_nodes] - z0)
    dhdt = dh/dt
    erosionMatrix = uplift_rate - dhdt
    meanE.append(np.mean(erosionMatrix))

    #Calculate river erosion rate, based on critical area threshold
    dh_riv = mg.at_node['topographic__elevation'][np.where(mg.at_node['drainage_area'] <= 100000)]\
        - z0[np.where(mg.at_node['drainage_area'] <= 100000)]
    dhdt_riv = dh_riv/dt
    mean_riv_E.append(np.mean(uplift_rate - dhdt_riv))

    #Calculate hillslope erosion rate
    dh_hill = mg.at_node['topographic__elevation'][np.where(mg.at_node['drainage_area'] <= 100000)]\
        - z0[np.where(mg.at_node['drainage_area'] <= 100000)]
    dhdt_hill = dh_hill/dt
    mean_hill_E.append(np.mean(uplift_rate - dhdt_hill))

    #update vegetation__density
    vegi_perc = np.random.rand(z.size)/100
    vegi_perc += vegi_test_timeseries[int(elapsed_time/dt)-1]

    #update lin_diff
    lin_diff = ldib*np.exp(-vegi_perc)
    #reinitialize diffuser
    ld = LinearDiffuser(mg,linear_diffusivity=lin_diff)

    #update K_sp
    n_v_frac = n_soil + (n_VRef*(vegi_perc/v_ref)) #self.vd = VARIABLE!
    n_v_frac_to_w = np.power(n_v_frac, w)
    Prefect = np.power(n_v_frac_to_w, 0.9)
    Kv = Ksp * Ford/Prefect
    Kfield = Kv
    #reinitialize StreamPowerEroder
    sp = StreamPowerEroder(mg, K_sp = Kfield, m_sp=msp, n_sp=nsp, sp_type = 'set_mn')

    #Calculate and save mean K-values
    #save mean_K_diff and mean_K_riv
    mean_K_riv.append(np.mean(Kfield))
    mean_K_diff.append(np.mean(lin_diff))

    #Calculate and save mean, max, min slopes
    mean_slope.append(np.mean(mg.at_node['topographic__steepest_slope'][mg.core_nodes]))
    max_slope.append(np.max(mg.at_node['topographic__steepest_slope'][mg.core_nodes]))
    min_slope.append(np.min(mg.at_node['topographic__steepest_slope'][mg.core_nodes]))

    #calculate and save mean, max, min elevation
    mean_elev.append(np.mean(mg.at_node['topographic__elevation'][mg.core_nodes]))
    max_elev.append(np.max(mg.at_node['topographic__elevation'][mg.core_nodes]))
    min_elev.append(np.min(mg.at_node['topographic__elevation'][mg.core_nodes]))

    #Run the output loop every oi-times
    if elapsed_time % oi  == 0:

        print('Elapsed Time:' , elapsed_time,'writing output!')
        ##Create DEM
        plt.figure()
        imshow_grid(mg,'topographic__elevation',grid_units=['m','m'],var_name = 'Elevation',cmap='terrain')
        plt.savefig('./DEM/DEM_'+str(int(elapsed_time/oi)).zfill(zp)+'.png')
        plt.close()
        ##Create Flow Accumulation Map
        plt.figure()
        imshow_grid(mg,fr.drainage_area,grid_units=['m','m'],var_name =
        'Drainage Area',cmap='bone')
        plt.savefig('./ACC/ACC_'+str(int(elapsed_time/oi)).zfill(zp)+'.png')
        plt.close()
        ##Create Slope - Area Map
        plt.figure()
        plt.loglog(mg.at_node['drainage_area'][np.where(mg.at_node['drainage_area'] > 0)],
           mg.at_node['topographic__steepest_slope'][np.where(mg.at_node['drainage_area'] > 0)],
           marker='.',linestyle='None')
        plt.xlabel('Area')
        plt.ylabel('Slope')
        plt.savefig('./SA/SA_'+str(int(elapsed_time/oi)).zfill(zp)+'.png')
        plt.close()
        ##Create NetCDF Output
        write_netcdf('./NC/output{}'.format(elapsed_time)+'__'+str(int(elapsed_time/oi)).zfill(zp)+'.nc',
                mg,format='NETCDF4')
        ##Create erosion_diffmaps
        plt.figure()
        imshow_grid(mg,erosionMatrix,grid_units=['m','m'],var_name='Erosion m/yr',cmap='jet')
        plt.savefig('./DHDT/eMap_'+str(int(elapsed_time/oi)).zfill(zp)+'.png')
        plt.close()

    elapsed_time += dt #update elapsed time
tE = time.time()
print()
print('End of  Main Loop. So far it took {}s to get here. No worries homeboy...'.format(tE-t0))

## OUTPUT OF EROSION RATES AND DIFFMAPS (BETA! NEEDS TO GO INTO SEPERATE CLASS
## TO KEEP RUNFILE NEAT AND SLEEK
#E-t:
fig, ax1 = plt.subplots(figsize = [11,7])
ax2 = ax1.twinx()
ax1.plot(timeVec, mean_hill_E, 'k', alpha = 0.6, linewidth = 2.5)
ax1.plot(timeVec, mean_riv_E, 'k--', alpha = 0.6, linewidth = 2.5)
#ax1.set_ylim([uplift_rate*0.9,uplift_rate*1.1])
ax1.plot(timeVec, mean_E, 'r', linewidth = 4.7)
ax2.plot(timeVec,100*vegi_test_timeseries,'g', linewidth = 4)
#ax2.set_ylim([0,100])
ax1.set_xlabel('years', fontsize = 22)
ax1.set_ylabel('Erosion rate', color='k', fontsize = 22)
ax2.set_ylabel('Vegetation cover [%]', color='k', fontsize = 22)
ax1.legend(['Hillslope Erosion','Fluvial Erosion', 'Total Erosion'], loc = 3, fontsize = 18)
ax2.legend(['Vegetation Cover'], loc = 4, fontsize = 18)
plt.savefig('./VegiEros_dualy.png',dpi = 720)

#Plot Vegi_erosion_rate
fig, axarr = plt.subplots(5, sharex = True, figsize = [11,14])
axarr[0].plot(timeVec, vegi_test_timeseries,'g', linewidth = 2.5)
axarr[0].set_title('Mean Surface Vegetation', fontsize = 12)
axarr[0].set_ylabel('Vegetation cover')
axarr[1].plot(timeVec, mean_elev, 'k', linewidth = 2.5)
axarr[1].plot(timeVec, max_elev, 'k--', linewidth = 2, alpha = 0.5)
axarr[1].plot(timeVec, min_elev, 'k--', linewidth = 2, alpha = 0.5)
axarr[1].set_title('Mean Elevation', fontsize = 12)
axarr[1].set_ylabel('Mean Elevation [m]')
#axarr[1].set_ylim([0,80])
axarr[2].plot(timeVec, np.degrees(np.arctan(mean_slope)), 'r', linewidth = 2.5)
axarr[2].plot(timeVec, np.degrees(np.arctan(max_slope)), 'r--', linewidth = 2.0, alpha = 0.5)
axarr[2].plot(timeVec, np.degrees(np.arctan(min_slope)), 'r--', linewidth = 2.0, alpha = 0.5)
#axarr[2].set_ylim([0,10])
axarr[2].set_title('Mean Slope', fontsize = 12)
axarr[2].set_ylabel('Mean Slope [deg]')
axarr[3].plot(timeVec,mean_dd, 'b', linewidth = 2.5)
axarr[3].set_title('Mean Drainage Density')
axarr[3].set_ylabel('Drainage Density')
axarr[4].plot(timeVec, mean_hill_E, 'g--', linewidth = 2.0, alpha = 0.5)
axarr[4].plot(timeVec, mean_riv_E, 'b--', linewidth = 2.0, alpha = 0.5)
axarr[4].plot(timeVec, mean_E, 'r--', linewidth = 2.2, alpha = 0.8)
axarr[4].legend(['Hillsl.', 'Rivers','Mean'])
axarr[4].set_title("Erosion rates")
axarr[4].set_ylabel('Erosion rate [m/yr]')
axarr[4].set_xlabel('Model Years', fontsize = 12)
plt.savefig('./Multiplot.png',dpi = 720)

#Normalized Plot
mean_E_norm = mean_E/np.max(mean_E)
mean_riv_E_norm = mean_riv_E/np.max(mean_riv_E)
mean_hill_E_norm = mean_hill_E/np.max(mean_hill_E)
mean_elev_norm = mean_elev/np.max(mean_elev)
mean_slope_norm = mean_slope/np.max(mean_slope)
mean_dd_norm = mean_dd/np.max(mean_dd)
fig, ax = plt.subplots(figsize = [11,7])
ax.plot(timeVec,vegi_test_timeseries, 'g--', linewidth = 5, alpha = 0.6)
#ax.plot(timeVec,mean_E_norm,'r--', linewidth = 5)
#ax.plot(timeVec,mean_elev_norm, 'k', linewidth = 5)
ax.plot(timeVec,mean_slope_norm, 'b', linewidth = 5, alpha = 0.7)
ax.plot(timeVec, mean_dd_norm, 'm', linewidth = 5)
#ax.legend(["Vegetation", "Erosion", "Elevation", "Slope", "Drainage Density"])
ax.legend(['Vegetation', 'Slope', 'Drainage Density'])

#Plot

print("FINALLY! TADA! IT IS DONE! LOOK AT ALL THE OUTPUT I MADE!!!!")

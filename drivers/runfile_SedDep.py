#Import necessesary Python and Landlab Modules
import numpy as np
from landlab import RasterModelGrid
from landlab import CLOSED_BOUNDARY, FIXED_VALUE_BOUNDARY
from landlab.components import FlowRouter
from landlab.components import LinearDiffuser
#from landlab.components import FastscapeEroder
from landlab.components import SedDepEroder
#from landlab.components import StreamPowerEroder
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
dx    = 100 #m

#------------TIME----------------#

#runtime
total_T1 = 1e5 #yrs.
#timestep
dt = 10        #yrs
#number of timesteps and fillfactor for ffmpeg printout
nt = total_T1/dt
#time-vector, mostly for plotting purposes
timeVec = np.arange(0,total_T1,dt)
#Set the interval at which we create output. Warning. High output frequency
#will slow down the script s ignificantly
oi = 10000
no = total_T1/oi
zp = len(str(int(no)))

#------------UPLIFT--------------#

uplift_rate = 5e-4 #m/yr
uplift_per_step = uplift_rate * dt

#-------------EROSION------------#

Ksp = 1e-7
msp  = 0.6
nsp  = 1.0
ldib   = 1e-5 #m^2/yr
#time
elapsed_time = 0

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
for edge in (mg.nodes_at_left_edge,mg.nodes_at_right_edge):
    mg.status_at_node[edge] = CLOSED_BOUNDARY
for edge in (mg.nodes_at_top_edge,mg.nodes_at_bottom_edge):
    mg.status_at_node[edge] = FIXED_VALUE_BOUNDARY

print("Finished setting up Grid and establishing boundary conditions")

#Set up vegetation__density field
vegi_perc = mg.zeros('node',dtype=float)
#vegi_trace_x = mg.x_of_node / 2
#more_vegi = np.where(mg.x_of_node >= vegi_trace_x)
#less_vegi = np.where(mg.x_of_node < vegi_trace_x)
vegi_perc[less_vegi] += 0.4
vegi_perc[more_vegi] += 0.4
#vegi_test_timeseries = (np.sin(0.00015*timeVec)+1)/2

#Do the K-field vegetation-dependend calculations
#Calculations after Istanbulluoglu
nSoil_to_15 = np.power(n_soil, 1.5)
Ford = AqDens * grav * nSoil_to_15
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
threshold_arr  = 0
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
#sp  = StreamPowerEroder(mg,K_sp = Kfield,m_sp=msp, n_sp=nsp, threshold_sp=threshold_arr)
sd = SedDepEroder(mg, K_sp=1.e-4, g=9.81, rock_density=2700,
                     sediment_density=2700, fluid_density=1000,
                     runoff_rate=1.,
                     sed_dependency_type='generalized_humped',
                     kappa_hump=13.683, nu_hump=1.13, phi_hump=4.24,
                     c_hump=0.00181, Qc='MPM', m_sp=0.5, n_sp=1.,
                     K_t=1.e-4, m_t=1.5, n_t=1., C_MPM=1., a_sp=1.,
                     b_sp=0.5, c_sp=1., k_w=2.5, k_Q=2.5e-7,
                     mannings_n=0.05, threshold_shear_stress=None,
                     Dchar=0.05, set_threshold_from_Dchar=True,
                     set_Dchar_from_threshold=False,
                     threshold_Shields=0.05,
                     slope_sensitive_threshold=False,
                     pseudoimplicit_repeats=5,
                     return_stream_properties=False)

#-------------RUNTIME------------#

#Main Loop 1 (After first sucess is confirmed this is all moved in a class....)
t0 = time.time()
print("finished initiation of eroding components. starting loop...")

while elapsed_time < total_T1:

    #create copy of "old" topography
    z0 = mg.at_node['topographic__elevation'].copy()

    #Call the erosion routines.
    fr.route_flow()
    ld.run_one_step(dt=dt)
    sp.run_one_step(dt=dt)
    mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_rate * dt #add uplift

    #calculate drainage_density
    channel_mask = mg.at_node['drainage_area'] > 10000
    dd = drainage_density.DrainageDensity(mg, channel__mask = channel_mask)
    mean_dd.append(dd.calc_drainage_density())

    #Calculate dhdt and E
    dh = (mg.at_node['topographic__elevation'] - z0)
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
    #sp = StreamPowerEroder(mg, K_sp = Kfield, m_sp=msp, n_sp=nsp, sp_type = 'set_mn')
    sd = SedDepEroder(mg, K_sp=Kfield, g=9.81, rock_density=2700,
                         sediment_density=2700, fluid_density=1000,
                         runoff_rate=1.,
                         sed_dependency_type='generalized_humped',
                         kappa_hump=13.683, nu_hump=1.13, phi_hump=4.24,
                         c_hump=0.00181, Qc='MPM', m_sp=0.5, n_sp=1.,
                         K_t=1.e-4, m_t=1.5, n_t=1., C_MPM=1., a_sp=1.,
                         b_sp=0.5, c_sp=1., k_w=2.5, k_Q=2.5e-7,
                         mannings_n=0.05, threshold_shear_stress=None,
                         Dchar=0.05, set_threshold_from_Dchar=True,
                         set_Dchar_from_threshold=False,
                         threshold_Shields=0.05,
                         slope_sensitive_threshold=False,
                         pseudoimplicit_repeats=5,
                         return_stream_properties=False)

    #Calculate and save mean K-values
    #save mean_K_diff and mean_K_riv
    mean_K_riv.append(np.mean(Kfield))
    mean_K_diff.append(np.mean(lin_diff))

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
plt.figure()
plt.plot(timeVec,meanE)
plt.ylabel('Erosion rate [m/yr]')
plt.xlabel('Runtime [yrs]')
plt.savefig('E-t-timeseries.png')

#Plot Vegi_erosion_rate
fig, ax1 = plt.subplots(figsize = [12,8])
ax2 = ax1.twinx()
ax1.plot(timeVec, vegi_test_timeseries, 'g--',linewidth = 2.2)
ax2.plot(timeVec, meanE, 'r-',linewidth = 2.2)
ax1.set_xlabel('years ')
ax1.set_ylabel('Vegetation Density %', color='g')
ax2.set_ylabel('Erosion Rate km/y', color='r')
plt.savefig('./VegiEros_dualy.png',dpi = 720)
plt.show()

#Plot

print("FINALLY! TADA! IT IS DONE! LOOK AT ALL THE OUTPUT I MADE!!!!")

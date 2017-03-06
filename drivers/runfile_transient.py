#Import necessesary Python and Landlab Modules
import numpy as np
from landlab import RasterModelGrid
from landlab import CLOSED_BOUNDARY, FIXED_VALUE_BOUNDARY
from landlab.components import FlowRouter
from landlab.components import LinearDiffuser
from landlab.components import FastscapeEroder
from landlab import imshow_grid
from landlab.io.netcdf import write_netcdf
from landlab.io.netcdf import read_netcdf
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import time

#Build up Grid and define variables

#-----------DOMAIN---------------#

ncols = 601
nrows = 601
dx    = 100

#------------TIME----------------#

#runtime T1
total_T1 = 5e6  #yrs.
#runtime T2
total_T2 = 6e6
#timestep
dt = 100        #yrs
#number of timesteps and fillfactor for ffmpeg printout
nt = total_T1/dt
#time-vector, mostly for plotting purposes
timeVec = np.arange(0,total_T2,dt)
#Set the interval at which we create output. Warning. High output frequency
#will slow down the script significantly
oi = 5000
no = total_T1/oi
zp = len(str(int(no)))

#------------UPLIFT--------------#

uplift_rate = 5e-4 #m/yr
uplift_per_step = uplift_rate * dt

#-------------EROSION------------#

Ksp1 = 1e-5
Ksp2 = 3e-5
msp  = 0.5
msp2 = 0.5
nsp  = 1.0
nsp2 = 1.0
ldi   = 1e-2
ldi2  = 3e-2
#time
elapsed_time = 0

#---------VARIABLE INITIAION-----#

dhdtA    = [] #Vector containing dhdt values for each node per timestep
meandhdt = [] #contains mean elevation change per timestep
meanE    = [] #contains the mean "erosion" rate out of Massbalance


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

#Create Threshold_sp field
threshold_arr  = mg.zeros('node',dtype=float)
threshold_arr += 3e-5
threshold_field = mg.add_field('node','threshold_sp',threshold_arr,noclobber = False)
imshow_grid(mg,'threshold_sp')
plt.title('Stream-Power Threshold')
plt.savefig('Distribution of SP_Threshold',dpi=720)

#Initialize the erosional components for the first runtime
fr  = FlowRouter(mg)
ld  = LinearDiffuser(mg, linear_diffusivity = ldi)
fc  = FastscapeEroder(mg,K_sp = Ksp1, m_sp = msp, n_sp=nsp, threshold_sp=threshold_arr)
#Initialize the erosional components for the second runtime
ld2 = LinearDiffuser(mg,linear_diffusivity=ldi2)
fc2 = FastscapeEroder(mg,K_sp = Ksp2, m_sp = msp2, n_sp = nsp2, threshold_sp = threshold_arr)

#-------------RUNTIME------------#
#----------FIRST LOOP------------#

#Main Loop 1 (After first sucess is confirmed this is all moved in a class....)
t0 = time.time()
while elapsed_time < total_T1:

    #create copy of "old" topography
    z0 = mg.at_node['topographic__elevation'].copy()

    #Call the erosion routines.
    fr.route_flow()
    ld.run_one_step(dt=dt)
    fc.run_one_step(dt=dt)
    mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_rate * dt #add uplift

    #Calculate dhdt and E
    dh = (mg.at_node['topographic__elevation'] - z0)
    dhdt = dh/dt
    erosionMatrix = uplift_rate - dhdt
    meanE.append(np.mean(erosionMatrix))

    #do some garbage collection
    del z0
    del dhdt

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
print('End of first loop. So far it took {}s to get here. No worries homeboy...'.format(tE-t0))
print('starting second loop. Prepare for WARP-SPEED SCOTTY! ENERGY!')

#-----------SECOND LOOP----------#
while elapsed_time < total_T2:

    #create copy of "old" topography
    z0 = mg.at_node['topographic__elevation'].copy()

    #Call the erosion routines.
    fr.route_flow()
    ld2.run_one_step(dt=dt)
    fc2.run_one_step(dt=dt)
    mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_rate * dt #add uplift

    #Calculate dhdt and E
    dh = (mg.at_node['topographic__elevation'] - z0)
    dhdt = dh/dt
    erosionMatrix = uplift_rate - dhdt
    meanE.append(np.mean(erosionMatrix))

    #do some garbage collection
    del z0
    del dhdt

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
print('End of last loop. But we are not at the end. The show must go on. '.format(tE-t0))

## OUTPUT OF EROSION RATES AND DIFFMAPS (BETA! NEEDS TO GO INTO SEPERATE CLASS
## TO KEEP RUNFILE NEAT AND SLEEK

#E-t:
plt.figure()
plt.plot(timeVec,meanE)
plt.ylabel('Erosion rate [m/yr]')
plt.xlabel('Runtime [yrs]')
plt.savefig('E-t-timeseries.png')

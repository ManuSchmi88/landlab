"""Landlab Driver for running Landscape Evolution Experiments with
    - Soil weathering
    - Soil diffusion
    - Detachment-limited river erosion
    - tectonic uplift
    - vegetation modulation of erosion effects

Created by: Manuel Schmid, University of Tuebingen, 07.04.2017
"""

## Import necessary Python and Landlab Modules
import numpy as np
from landlab import RasterModelGrid
from landlab import CLOSED_BOUNDARY, FIXED_VALUE_BOUNDARY
from landlab.components import FlowRouter
from landlab.components import DepthDependentDiffuser
from landlab.components import ExponentialWeatherer
from landlab.components import StreamPowerEroder
from landlab.components import LinearDiffuser
from landlab.components import FastscapeEroder
from landlab.components import DepressionFinderAndRouter
from landlab.components import drainage_density
from landlab.components import SteepnessFinder
from landlab import imshow_grid
from landlab.io.netcdf import write_netcdf
from landlab.io.netcdf import read_netcdf
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams['agg.path.chunksize'] = 200000000
import time

#---------------------------Parameter Definitions------------------------------#
##Model Grid:##
ncols = 101
nrows = 101
dx    = 100

##Model runtime:##
#Defines total model runtime
totalT = int(15.e6)

### Sin Modulation Vegetation ###
#Defines spinup-time before sin-modulation
ssT    = int(5.e6)
#Defines Amplitude of vegi sin-modulation
sinAmp = 0.1
#Defines Period of sin-modulation
sinPeriod = 1e5

### Step - function Vegetation ###
#vegi step-function time
sfT    = int(5.e6)
#vegi step-function amplitude
sfA = 0.

### Step - function Rainfall ###
#rainfall amplitude
baseRainfall = 3 
rfA = 0 
transientRainfallTimespan = int(totalT - ssT)


#Defines time-step
dt     = int(100)
#Number of total-timestep (nt) and spin-up timesteps (ssnt)
nt = int(totalT / dt)
ssnt = int(ssT / dt)
ssntSF = int(sfT / dt)
#time-vector (total and transient), used for plotting later
timeVec = np.arange(0, totalT, dt)
transTimeVec = np.arange(0, (totalT - ssT), dt)

##Uplift:##
#upliftRate
upliftRate = 2.1e-4 #m/yr
uplift_per_step = upliftRate * dt

##Fluvial Erosion:
##Fluvial erosion is implement as stream-power-law
##E = K * A^m * S^n
##For comparison look up usual Tucker, Whipple etc.
#ksp = 0.4
ksp = 7e-8 
msp = 0.5
nsp = 1 
thresholdSP = 4.e-4 

critArea = 1e6 #Minimum area for channel formation

##Hillslope Erosion:
##hillslope erosion is implement as linear diffusion which depends on soil depth
##for comparison look up Johnstone and Hilley 2014.
##Linear Diffusivity is modified after Istanbulluoglu 2015, and represented as as
##numpy-array. You need to define the "Base-diffusivity":
linDiffBase   = 2e-1  #m^2/yr
alphaDiff     = 0.3    #After Istanbulluoglu
soilTransportDepth = 1 #m
initalSoilDepth = 0    #m
#topographic_elevation - soil_depth
initialBedrockElevation = 0 #m

##Bedrock Weathering:
#weathering of bedrock is implemented as exponential weathering function after
#Ahnert 1976
maxSoilProductionRate = 0.00015 #m/yr
soilProductionDepth   = 0.2     #m

##Vegetation Implementation
##This is stuff that is needed to use the vegetation-modulation after
##Istanbulluoglu 2015
aqDens  = 1000.    #Density of water
grav    = 9.81     #Gravitation Acceleration
nSoil   = 0.01     #Mannings Number for bare soil [-]
nVRef   = 0.6      #Mannungs Number for Vegetation
vRef    = 1.      #Reference Vegetation for Maximum Mannings Roughness
w       = 1.       #Scaling Factor for vegetation influence

##Output production
##This Driver is set up to produce NetCDFs, .pngs, etc.
##For more detail look in the Plotting section of the main loop
##and comment stuff that you don't want produced
#Interval in which output is produced
outInt = 10000 #yrs
#Number of total produced outputs
no = totalT / outInt
#number of zeros for file_naming. Don't meddle with this.
zp = len(str(int(no)))

print("finished with parameter-initiation")
print("---------------------")
#---------------------------------Grid Setup-----------------------------------#
#This initiates a Modelgrid with dimensions nrows x ncols and spatial scaling of dx
mg = RasterModelGrid((nrows,ncols), dx)

#only uncomment this if there is a pre-existing topography you want to load. 
#right now this only works if the topo was saved in numpys .npy format.
try:
    topoSeed = np.load('topoSeed.npy')
    print('loaded topoSeed.npy')
except:
    print('There is no file containing a initial topography')

#Initate all the fields that are needed for calculations
mg.add_zeros('node','topographic__elevation')
mg.add_zeros('node','bedrock__elevation')
mg.at_node['bedrock__elevation'][:] = initialBedrockElevation
#checks if standart topo is used. if not creates own
if 'topoSeed' in locals():
    mg.at_node['topographic__elevation'] += topoSeed 
    #mg.at_node['bedrock__elevation'] += topoSeed 
    print('Using pre-existing topography from file topoSeed.npy')
else:
    mg.at_node['topographic__elevation'] += np.random.rand(mg.at_node.size)/10000 
    #mg.at_node['bedrock__elevation'] += np.random.rand(mg.at_node.size)/10000
    print('No pre-existing topography. Creating own random noise topo.')

mg.add_zeros('node','soil__depth')
mg.at_node['soil__depth'][:] = initalSoilDepth
mg.add_zeros('node','vegetation__density')

#Create boundary conditions of the model grid (either closed or fixed-head)
for edge in (mg.nodes_at_left_edge,mg.nodes_at_right_edge, mg.nodes_at_top_edge):
    mg.status_at_node[edge] = CLOSED_BOUNDARY
for edge in (mg.nodes_at_bottom_edge):
    mg.status_at_node[edge] = FIXED_VALUE_BOUNDARY

print("finished with setup of modelgrid")
print("---------------------")

##---------------------------------Vegi implementation--------------------------#
##Set up a timeseries for vegetation-densities
##This basically assumes that for a spin-up time (ssT) we have constant vegetation
##cover (vp) and after that we get change vegetation cover as a sin-function
vp = .1 
vegiTimeseries  = np.zeros(int(totalT / dt)) + vp

#This part sets the modification of the vegi-distribution. comment/uncomment
#for usage
#this modifies the vegiTimeseries array with a sinusoidal curve:
#vegiDens = sinAmp * sin(sinB * timeVec) + vp
sinB = (2*np.pi) / sinPeriod
vegiTimeseries[ssnt:] =  sinAmp * np.sin(sinB * transTimeVec) + vp
#this incorporates a vegi step-function at timestep sfT with amplitude sfA
#vegiTimeseries[ssntSF:] = vp - sfA
mg.at_node['vegetation__density'][:] = vp
#This maps the vegetation density on the nodes to the links between the nodes
vegiLinks = mg.map_mean_of_link_nodes_to_link('vegetation__density')

##These are the necesseray calculations for implementing the vegetation__density
##in the fluvial routines
nSoil_to_15 = np.power(nSoil, 1.5)
Ford = aqDens * grav * nSoil_to_15
n_v_frac = nSoil + (nVRef * ((mg.at_node['vegetation__density'] / vRef)**w)) #self.vd = VARIABLE!
#n_v_frac_to_w = np.power(n_v_frac, w)
#Prefect = np.power(n_v_frac_to_w, 0.9)
Prefect = np.power(n_v_frac, 0.9)
Kv = ksp * Ford/Prefect

##These are the calcultions to calculate the linear diffusivity based on vegis
linDiff = mg.zeros('node', dtype = float)
linDiff = linDiffBase * np.exp(-alphaDiff * vegiLinks)

print("finished setting up the vegetation fields and Kdiff and Kriv")
print("---------------------")

##---------------------------------Rain implementation--------------------------#
##Set up a Timeseries of rainfall values
#YOU NEED TO COMMENT/UNCOMMENT THE WHOLE SECTION IF YOU WANT TO SWITCH BETWEEN
#STEP CHANGE RAINFALL AND OSCILLATING RAINFALL

rainTimeseries = np.zeros(int(totalT / dt)) + baseRainfall
mg.add_zeros('node', 'rainvalue')
mg.at_node['rainvalue'][:] = int(baseRainfall)
##----Step-change modification---#
#rainTimeseries[ssntSF:] = baseRainfall - rfA 

##---OSCILLATING---#
#This function creates a asymetrical wave-form for rainfall input
def createAsymWave(baseLevel,
                   posAmp,
                   negAmp,
                   period,
                   time,
                   dt):
    #This sets up the basic sin-waveform which underlies the algorith
    baseTime = np.arange(0,time,dt)
    print('created Timeline')
    baseFreq = (2 * np.pi) / period
    print('established Frequency')
    baseWave = np.sin(baseFreq * baseTime) + (baseLevel)
    print('created waveform')
    
    #now we create a wave with different amplitudes by vector addition
    #positive Part
    posInd = np.where(baseWave > baseLevel)
    posSinAmp = posAmp - baseLevel
    posSinWave = posSinAmp * np.sin(baseFreq * baseTime) + baseLevel
    baseWave[posInd] = posSinWave[posInd]
    #negative Part
    negInd = np.where(baseWave < baseLevel)
    negSinAmp = baseLevel - negAmp
    negSinWave = negSinAmp * np.sin(baseFreq * baseTime) + baseLevel
    baseWave[negInd] = negSinWave[negInd]

    return baseWave
    
def createSymWave(baseLevel,
                 ampMax,
                 ampMin,
                 period,
                 time,
                 dt):
    baseTime = np.arange(0,time,dt)
    print('created Timeline')
    baseFreq = (2 * np.pi) / period
    print('established Frequency')
    ampTotal = np.abs((ampMax - ampMin) / 2)
    newBase = np.abs((ampMax + ampMin) / 2)
    baseWave =  ampTotal * np.sin(baseFreq * baseTime) + newBase
    print('created waveform')
    
    return baseWave
  
rainTimeseries[ssntSF:] = createAsymWave(baseRainfall, 6, 0, sinPeriod,
        transientRainfallTimespan, dt)  

    
##---------------------------------Array initialization---------------------#
##This initializes all the arrays that are used to store data during the runtime
##of the model. this is mostly for plotting purposed and to create the .txt
##outputs. This potentially takes up a lot of space, so check if needed.
dhdtA       = [] #Vector containing dhdt values for each node per timestep
meandhdt    = [] #contains mean elevation change per timestep
mean_E      = [] #contains the mean "erosion" rate out of Massbalance
mean_hill_E = [] #contains mean hillslope erosion rate
mean_riv_E  = [] #contains mean river erosion rate
mean_dd     = [] #contains mean drainage density
mean_K_riv  = [] #contains mean K-value for spl
mean_K_diff = [] #contains mean K-value for ld
mean_slope  = [] #mean slope within model-area
max_slope   = [] #maximum slope within model area
min_slope   = [] #minimum slope within model area
mean_elev   = [] #mean elevation within model area
max_elev    = [] #maximum elevation within model area
min_elev    = [] #minimum elevation within model area
vegi_P_mean = [] #mostly for bugfixing because Manu is stupid fuckup without brain and life and fuck you
mean_SD     = [] #mean soil depth
mean_Ksn    = [] #mean channel steepness
max_Ksn     = [] #max channel steepness

##---------------------------------Component initialization---------------------#
#sp = StreamPowerEroder(mg,
#                       K_sp = Kv,
#                       m_sp = msp,
#                       n_sp = nsp,
#                       threshold_sp=thresholdSP)

fc = FastscapeEroder(mg,
                    K_sp = Kv,
                    m_sp = msp,
                    n_sp = nsp,
                    threshold_sp = 0,
                    rainfall_intensity = baseRainfall)

fr = FlowRouter(mg)

lm = DepressionFinderAndRouter(mg)

expw = ExponentialWeatherer(mg,
                            max_soil_production_rate = maxSoilProductionRate,
                            soil_production_decay_depth = soilProductionDepth)

dld  = DepthDependentDiffuser(mg,
                              linear_diffusivity = linDiff,
                              soil_transport_decay_depth = soilTransportDepth)

ld   = LinearDiffuser(mg, linear_diffusivity = linDiff)

sf = SteepnessFinder(mg,
                    min_drainage_area = 1e6)

print("finished with the initialization of the erosion components")   
print("---------------------")

##---------------------------------Main Loop------------------------------------#
t0 = time.time()
elapsed_time = 0
print("starting with main loop.")
print("---------------------")
#Create incremental counter for controlling progress of mainloop
counter = 0
#Create Limits for DHDT plot. Move this somewhere else later..
DHDTLowLim = upliftRate - (upliftRate * 1)
DHDTHighLim = upliftRate + (upliftRate * 1)

while elapsed_time < totalT:

    #create copy of "old" topography
    z0 = mg.at_node['topographic__elevation'].copy()

    #Call the erosion routines.
    #expw.run_one_step(dt=dt)
    #dld.run_one_step(dt=dt)
    ld.run_one_step(dt=dt)
    fr.run_one_step()
    lm.map_depressions()
    floodedNodes = np.where(lm.flood_status==3)[0]
    fc.run_one_step(dt=dt, flooded_nodes = floodedNodes)
    sf.calculate_steepnesses()
    mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_per_step #add uplift
    mg.at_node['bedrock__elevation'][mg.core_nodes] += uplift_per_step #add uplift

    #look for nodes where river incises below current soil thickness
    bad_nodes = mg.at_node['topographic__elevation'] < mg.at_node['bedrock__elevation']
    #redefine bedrock to current channel elevation
    mg.at_node['bedrock__elevation'][bad_nodes] = mg.at_node['topographic__elevation'][bad_nodes]

    #calculate drainage_density
    channel_mask = mg.at_node['drainage_area'] > critArea
    dd = drainage_density.DrainageDensity(mg, channel__mask = channel_mask)
    mean_dd.append(dd.calc_drainage_density())

    #Calculate dhdt and E
    dh = (mg.at_node['topographic__elevation'] - z0)
    dhdt = dh/dt
    erosionMatrix = upliftRate - dhdt
    mean_E.append(np.mean(erosionMatrix))

    #Calculate river erosion rate, based on critical area threshold
    dh_riv = mg.at_node['topographic__elevation'][np.where(mg.at_node['drainage_area'] > critArea)]\
        - z0[np.where(mg.at_node['drainage_area'] > critArea)]
    dhdt_riv = dh_riv/dt
    mean_riv_E.append(np.mean(upliftRate - dhdt_riv))

    #Calculate hillslope erosion rate
    dh_hill = mg.at_node['topographic__elevation'][np.where(mg.at_node['drainage_area'] <= critArea)]\
        - z0[np.where(mg.at_node['drainage_area'] <= critArea)]
    dhdt_hill = dh_hill/dt
    mean_hill_E.append(np.mean(upliftRate - dhdt_hill))

    #update vegetation__density
    mg.at_node['vegetation__density'][:] = vegiTimeseries[int(elapsed_time/dt)-1]
    vegiLinks = mg.map_mean_of_link_nodes_to_link('vegetation__density')

    #update linDiff
    linDiff = linDiffBase*np.exp(-alphaDiff * vegiLinks)
    #reinitialize diffuser
#   dld  = DepthDependentDiffuser(mg,
#                                  linear_diffusivity = linDiff,
#                                  soil_transport_decay_depth = soilTransportDepth)
#

    ld   = LinearDiffuser(mg, linear_diffusivity = linDiff) 

    #update K_sp
    n_v_frac = nSoil + (nVRef * (mg.at_node['vegetation__density'] / vRef)) #self.vd = VARIABLE!
    n_v_frac_to_w = np.power(n_v_frac, w)
    Prefect = np.power(n_v_frac_to_w, 0.9)
    Kv = ksp * Ford/Prefect

    #update Rainfallvalues
    rainValue = rainTimeseries[int(elapsed_time/dt)-1]
    mg.at_node['rainvalue'][:] = rainValue

    fc = FastscapeEroder(mg,
                    K_sp = Kv,
                    m_sp = msp,
                    n_sp = nsp,
                    threshold_sp = thresholdSP,
                    rainfall_intensity = rainValue )

    #Calculate and save mean K-values
    #save mean_K_diff and mean_K_riv
    mean_K_riv.append(np.mean(Kv))
    mean_K_diff.append(np.mean(linDiff))

    #Calculate and save mean, max, min slopes
    mean_slope.append(np.mean(mg.at_node['topographic__steepest_slope'][mg.core_nodes]))
    max_slope.append(np.max(mg.at_node['topographic__steepest_slope'][mg.core_nodes]))
    min_slope.append(np.min(mg.at_node['topographic__steepest_slope'][mg.core_nodes]))

    #calculate and save mean, max, min elevation
    mean_elev.append(np.mean(mg.at_node['topographic__elevation'][mg.core_nodes]))
    max_elev.append(np.max(mg.at_node['topographic__elevation'][mg.core_nodes]))
    min_elev.append(np.min(mg.at_node['topographic__elevation'][mg.core_nodes]))

    #SoilDepth
    mean_SD.append(np.mean(mg.at_node['soil__depth'][mg.core_nodes]))
    
    #Mean Ksn Value
    _ksndump = mg.at_node['channel__steepness_index'][mg.core_nodes]
    mean_Ksn.append(np.mean(_ksndump[np.nonzero(_ksndump)]))
    max_Ksn.append(np.max(_ksndump[np.nonzero(_ksndump)]))

    counter += 1
    #print(counter)

    #Run the output loop every outInt-times
    if elapsed_time % outInt  == 0:

        print('Elapsed Time:' , elapsed_time,', writing output!')
        ##Create DEM
        plt.figure()
        imshow_grid(mg,'topographic__elevation',grid_units=['m','m'],var_name = 'Elevation',cmap='terrain')
        plt.savefig('./DEM/DEM_'+str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        plt.close()
        ##Create Flow Accumulation Map
        plt.figure()
        imshow_grid(mg,fr.drainage_area,grid_units=['m','m'],var_name =
        'Drainage Area',cmap='bone')
        plt.savefig('./ACC/ACC_'+str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        plt.close()
        ##Create Slope - Area Map
        plt.figure()
        plt.loglog(mg.at_node['drainage_area'][np.where(mg.at_node['drainage_area'] > 0)],
           mg.at_node['topographic__steepest_slope'][np.where(mg.at_node['drainage_area'] > 0)],
           marker='.',linestyle='None')
        plt.xlabel('Area')
        plt.ylabel('Slope')
        plt.savefig('./SA/SA_'+str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        plt.close()
        ##Create NetCDF Output
        write_netcdf('./NC/output{}'.format(elapsed_time)+'__'+str(int(elapsed_time/outInt)).zfill(zp)+'.nc',
                mg,format='NETCDF4')
        ##Create erosion_diffmaps
        plt.figure()
        imshow_grid(mg,erosionMatrix,grid_units=['m','m'],var_name='Erosion m/yr',cmap='jet',limits=[DHDTLowLim,DHDTHighLim])
        plt.savefig('./DHDT/eMap_'+str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        plt.close()
        ##Create Ksn Maps
        plt.figure()
        imshow_grid(mg, 'channel__steepness_index', grid_units=['m','m'],
                var_name='ksn', cmap='jet')
        plt.savefig('./Ksn/ksnMap_'+str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        plt.close()
        #plt.figure()
        #imshow_grid(mg,'soil__depth',grid_units=['m','m'],var_name=
        #        'Elevation',cmap='terrain')
        #plt.savefig('./SoilDepth/SD_'+str(int(elapsed_time/outInt)).zfill(zp)+'png')
        #plt.close()

    elapsed_time += dt #update elapsed time
tE = time.time()
print()
print('End of  Main Loop. So far it took {}s to get here. No worries homeboy...'.format(tE-t0))


##---------------------------------Plotting-------------------------------------#

## OUTPUT OF EROSION RATES AND DIFFMAPS (BETA! NEEDS TO GO INTO SEPERATE CLASS
## TO KEEP RUNFILE NEAT AND SLEEK
#E-t:
#fig, ax1 = plt.subplots(figsize = [11,7])
#ax2 = ax1.twinx()
#ax1.plot(timeVec, mean_hill_E, 'k', alpha = 0.6, linewidth = 2.5)
#ax1.plot(timeVec, mean_riv_E, 'k--', alpha = 0.6, linewidth = 2.5)
##ax1.set_ylim([upliftRate*0.9,upliftRate*1.1])
#ax1.plot(timeVec, mean_E, 'r', linewidth = 4.7)
#ax2.plot(timeVec,100*vegiTimeseries,'g', linewidth = 4)
##ax2.set_ylim([0,100])
#ax1.set_xlabel('years', fontsize = 22)
#ax1.set_ylabel('Erosion rate', color='k', fontsize = 22)
#ax2.set_ylabel('Vegetation cover [%]', color='k', fontsize = 22)
#ax1.legend(['Hillslope Erosion','Fluvial Erosion', 'Total Erosion'], loc = 3, fontsize = 18)
#ax2.legend(['Vegetation Cover'], loc = 4, fontsize = 18)
#plt.savefig('./VegiEros_dualy.png',dpi = 720)
#plt.close()

#Plot Vegi_erosion_rate
fig, axarr = plt.subplots(6, sharex = True, figsize = [11,14])
axarr[0].plot(timeVec, vegiTimeseries,'g', linewidth = 2.5)
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
axarr[5].plot(timeVec, rainTimeseries, 'k', linewidth = 2.5)
axarr[5].set_title("Rainfall")
axarr[5].set_ylabel("Rain Value")
axarr[5].set_xlabel("Model Years", fontsize = 12)
plt.savefig('./Multiplot_absolut.png',dpi = 720)
plt.close()

#Multiplot with normalized differentations
#vegi_timeseries_diff = np.diff(vegiTimeseries)/(np.max(np.diff(vegiTimeseries)))
#mean_elev_diff = np.diff(mean_elev)/(np.max(np.abs((np.diff(mean_elev)))))
#mean_slope_diff = np.diff(mean_slope)/(np.max(np.abs(np.diff(mean_slope))))
#mean_hill_E_diff = np.diff(mean_hill_E)/(np.max(np.abs(np.diff(mean_hill_E))))
#mean_riv_E_diff = np.diff(mean_riv_E)/(np.max(np.abs(np.diff(mean_riv_E))))
#mean_E_diff = np.diff(mean_E)/np.max(np.abs(np.diff(mean_E)))
#mean_dd_diff = np.diff(mean_dd)/np.max(np.abs(np.diff(mean_dd)))
#timeVec_diff = np.delete(timeVec, -1)

#fig, axarr = plt.subplots(5, sharex = True, figsize = [11,14])
#axarr[0].plot(timeVec_diff, vegi_timeseries_diff,'g', linewidth = 2.5)
#axarr[0].plot(timeVec,vegiTimeseries,'g--',alpha=0.5)
#axarr[0].set_title('Change In Mean Surface Vegetation', fontsize = 12)
#axarr[0].set_ylabel('Vegetation cover change')
#axarr[1].plot(timeVec_diff, mean_elev_diff, 'k', linewidth = 2.5)
#axarr[1].set_title('Change In Mean Elevation', fontsize = 12)
#axarr[1].set_ylabel('dh/dt')
#axarr[2].plot(timeVec_diff, mean_slope_diff, 'r', linewidth = 2.5)
#axarr[2].set_title('Change In Mean Slope', fontsize = 12)
#axarr[2].set_ylabel('dS/dt')
#axarr[3].plot(timeVec_diff,mean_dd_diff, 'b', linewidth = 2.5)
#axarr[3].set_title('Change In Mean Drainage Density')
#axarr[3].set_ylabel('d(dd)/dt')
##axarr[4].plot(timeVec_diff, mean_hill_E_diff, 'g--', linewidth = 2.0, alpha = 0.5)
##axarr[4].plot(timeVec_diff, mean_riv_E_diff, 'b--', linewidth = 2.0, alpha = 0.5)
#axarr[4].plot(timeVec_diff, mean_E_diff, 'r--', linewidth = 2.2, alpha = 0.8)
##axarr[4].legend(['Hillsl.', 'Rivers','Mean'])
#axarr[4].set_title("Change In Erosion Rates")
#axarr[4].set_ylabel('dE/dt')
#axarr[4].set_xlabel('Model Years', fontsize = 12)
#plt.savefig('./Multiplot_diff.png',dpi = 720)
#plt.close()

#Save the most useful output arrays as CSV file for later plotting
np.savetxt('./CSVOutput/MeanSlope.csv', mean_slope)
np.savetxt('./CSVOutput/MaxSlope.csv', max_slope)
np.savetxt('./CSVOutput/MeanElev.csv', mean_elev)
np.savetxt('./CSVOutput/MaxElev.csv', max_elev)
np.savetxt('./CSVOutput/MeanRiverErosion.csv', mean_riv_E)
np.savetxt('./CSVOutput/MeanHillslopeErosion.csv', mean_hill_E)
np.savetxt('./CSVOutput/MeanErosion.csv', mean_E)
np.savetxt('./CSVOutput/VegetationDensity.csv', vegiTimeseries)
np.savetxt('./CSVOutput/Timeseries.csv', timeVec)
np.savetxt('./CSVOutput/Vegi_bugfix.csv', vegi_P_mean)
np.savetxt('./CSVOutput/MeanSoilthick.csv', mean_SD)
np.savetxt('./CSVOutput/MeanRivK.csv', mean_K_riv)
np.savetxt('./CSVOutput/MeanHillK.csv', mean_K_diff)
np.savetxt('./CSVOutput/MeanKsn.csv', mean_Ksn)
np.savetxt('./CSVOutput/MaxKsn.csv', max_Ksn)
np.savetxt('./CSVOutput/RainTimeseries.csv', rainTimeseries)

#bugfixing because manu is a fucking dumb asshole
plt.plot(vegi_P_mean)
plt.savefig('./vegi_P_bugfix.png', dpi = 720)
plt.close()
print("FINALLY! TADA! IT IS DONE! LOOK AT ALL THE OUTPUT I MADE!!!!")

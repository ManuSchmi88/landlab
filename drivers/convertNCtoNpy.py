""" 
This script will take a landlab netcdf file as input and convert the 
'topographic__elevation' parameter to a numpy array .npy which can be
loaded in the standart landlab script.
"""

import numpy as np
import matplotlib.pyplot as plt
from landlab import RasterModelGrid
from landlab.io.netcdf import write_netcdf
from landlab.io.netcdf import read_netcdf
from landlab import imshow_grid


fileNc = 'output19990000__1999.nc'


def loadNetCDF(filename):
    global conGrid
    try:
        conGrid = read_netcdf(filename)
    except:
        print('No Netcdf')

def convertNetCdf():
    _buffGrid = conGrid.at_node['topographic__elevation'][:]
    np.save('convertTopography', _buffGrid)

def makeExamplePlot():
    plt.figure()
    imshow_grid(conGrid, 'topographic__elevation')
    plt.savefig('convertTopo.png')
    plt.close()

def main():

    loadNetCDF(fileNc)
    convertNetCdf()
    makeExamplePlot()
    print('####### FINISHED CONVERSION #########')


main()

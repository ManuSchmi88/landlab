""" Creates a standarized topography and saves it in the numpy-inherent array
format. Therefore needs numpy to work."""

## Import necessary Python and Landlab Modules
import numpy as np
from matplotlib import pyplot as plt
from landlab import RasterModelGrid
from landlab import imshow_grid

ncols = 101
nrows = 101
dx = 100

#creates landlab grid out of ncols and nrows to get the right amount of nodes
#spacing. could also do completely with numpy but thats easier...
mg = RasterModelGrid((nrows, ncols), dx)
#creates random array with size of mg-grid
topoSeed = np.random.rand(mg.at_node.size)/100
#saves topoSeed as numpy array
np.save('topoSeed',topoSeed)
plt.figure()
imshow_grid(mg, topoSeed)
plt.savefig('initalTopo.png')
plt.close()

print('------------------------------')
print('Saved a output-topography to file topoSeed.npy')
print('Saved a picture of the topography to initialTopo.png')
print('------------------------------')

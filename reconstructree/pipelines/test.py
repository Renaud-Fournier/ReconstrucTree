from reconstructree import *


pointset = load_txt('/home/fournierr/Documents/Stage CIRAD/data/Winter trees/X0036-1-1-WW.txt')

bbx = getbbx(pointset)

voxelsize = 0.1

tensor = totensor(pointset, bbx, voxelsize)

print(shape(tensor))

newpointset = topointset(tensor, voxelsize, bbx)

newbbx = getbbx(newpointset)

print(list(bbx))
print(list(newbbx))

#savepointset('/home/fournierr/Documents/Stage CIRAD/data/pointset.txt', pointset)


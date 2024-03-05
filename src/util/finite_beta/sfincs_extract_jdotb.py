#!/usr/bin/env python3

import pickle
import numpy as np

pfile = open('sfincsScan.dat', 'rb')
data = pickle.load(pfile)
pfile.close()

# Convert FSABjHat to Tesla Amperes / m^2:
factor = 437695 * 1e20 * 1.602177e-19

#print(data)

lookFor = 'FSABjHat'
foundIt = False
for j in range(len(data['ylabels'])):
   oldLabel = data['ylabels'][j]
   if oldLabel.find(lookFor) == 0:
      print("Found FSABjHat in sfincs output at index",j)
      foundIt = True
      break
if not foundIt:
   print("Error! No quantity with name beginning "+lookFor+" could be found in the sfincsScan.dat file.")
   exit(1)

index = j

assert data['ylabels'][index] == 'FSABjHat'

print('s:')
print(data['xdata'][index])
print('jdotb [SI units]:')
print(data['ydata'][index] * factor)

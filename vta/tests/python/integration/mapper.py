"""
Author: Vinod Ganesan, Gokulan Ravi
Last Updated: 25th April, 2020

A simple python based DNN mapper tool that maps a given DNN layer description to a set of
SHAKTI ISA instructions

TODO: Extend this to parse directly from PyTorch network definition
"""

import numpy as np
import configparser
import math

### Reading the Config File
config = configparser.ConfigParser()
config.read('systolic.config')

## Systolic Configurations
nRows = int(config['SYSTOLIC_DIMS']['nRows'])
nCols = int(config['SYSTOLIC_DIMS']['nCols'])
bufferGranularity = int(config['SYSTOLIC_DIMS']['bufferGranularity'])
inpBufferSize = int(config['SYSTOLIC_DIMS']['inpBufferSize'])*bufferGranularity
accumBufferSize = int(config['SYSTOLIC_DIMS']['accumBufferSize'])*bufferGranularity
weightBufferSize = int(config['SYSTOLIC_DIMS']['weightBufferSize'])*bufferGranularity

## Network Configurations
opType = config['DNN_DIMS']['opType']
inp_dims = np.array([int(item) for item in config['DNN_DIMS']['inp_dimensions'].split('x')])
filter_dims = np.array([int(item) for item in config['DNN_DIMS']['filter_dimensions'].split('x')])
bitWidth = config['DNN_DIMS']['bitWidth']
stride = int(config['DNN_DIMS']['stride'])
padding = int(config['DNN_DIMS']['padding'])

N = inp_dims[0]
C = inp_dims[1]
H = inp_dims[2]
W = inp_dims[3]

M = filter_dims[0]
R = filter_dims[2]
S = filter_dims[3]

if not inp_dims[1] == filter_dims[1]:
  print("Filter Channels and Input Channels don't match -- Please Fix")
  exit(0)

## Miscellaneous 
filename = open("isa.txt", "w")
""" 
Algorithm to determine the mapping

1. xxxxxxx
"""


## TODO: Insert Load_Param instruction -- figure out where to insert them
def spit_ISA_conv(num_filter_fold_4d, num_input_fold_4d, inp_slice, filter_slice, tot_systolic_folds, instr_type):
  Ns = inp_slice[0]
  Hs = inp_slice[2]
  Ws = inp_slice[3]

  Ms = filter_slice[0]
  Cs = filter_slice[1]
  Rs = filter_slice[2]
  Ss = filter_slice[3]
  temp = 0
  out_size = 0
  nprev = 0 
  hprev = 0
  wprev = 0
  cprev = 0

  for n in range(Ns):
    ifmap_size = Cs * Hs * Ws
    I = int(math.ceil(ifmap_size / inpBufferSize))
    for i in range(I):
      Hnew = inpBufferSize // (Cs * Ws)
      print("hnew", Hnew)
      minH = i * Hnew
      maxH = min((i+1)*Hnew, Hs)
      filename.write("LOAD_INPUT: %d:%d, %d:%d, %d:%d, %d:%d\n" % (n, n, minH, maxH-1, 0, Ws-1, 0, Cs-1))

      ofmap_size = Hnew * Ws

      num_filters = accumBufferSize // ofmap_size
      num_filters = num_filters - (num_filters % nCols)

      if num_filters < nCols:
        num_filters = nCols


      num_filter_channels_to_load = weightBufferSize // (num_filters * Rs * Ss)
      if num_filter_channels_to_load < nRows:
        num_filter_channels_to_load = nRows
        num_filters = weightBufferSize // (num_filter_channels_to_load * Rs * Ss) 
        num_filters = num_filters - (num_filters % nCols)

      if (weightBufferSize / num_filters * num_filter_channels_to_load) <= Rs * Ss:
        num_pixels = weightBufferSize // (num_filters * num_filter_channels_to_load)
      else:
        num_pixels = Rs * Ss

      print(num_filters, num_filter_channels_to_load, num_pixels)

      M = int(Ms / num_filters)
      for m1 in range(M):
        for m2 in range(int(math.ceil(num_filters / nCols))):
          for mc in range(int(math.ceil(Cs / num_filter_channels_to_load))):
            for mrs in range(int(math.ceil(Rs * Ss / num_pixels))):
              filename.write("LOAD_WEIGHT: %d:%d, %d:%d, %d:%d\n" % 
                (mrs * num_pixels, (mrs+1) * num_pixels - 1,
                mc * num_filter_channels_to_load, (mc+1) * num_filter_channels_to_load - 1,
                m1 * num_filters + m2 * nCols, m1 * num_filters + (m2+1) * nCols - 1
                ))
              filename.write("COMPUTE_SYSTOLIC\n")
 
          filename.write("STORE_OUTPUT: %d:%d, %d:%d, %d:%d, %d:%d\n" %
            (n, n, minH, maxH-1, 0, Ws-1, m1 * num_filters + m2 * nCols, m1 * num_filters + (m2+1) * nCols-1)) 

#TODO Take care of condition when the input is not in the buffer -- simple but needs to be handled 
def spit_ISA_relu_maxpool2d(is_buffer, num_input_fold_4d, inp_size):
  size = 0 
  nprev = 0
  hprev = 0
  wprev = 0
  cprev = 0
  for n in range(num_input_fold_4d[0]):
    for h in range(num_input_fold_4d[2]):
      for w in range(num_input_fold_4d[3]):
        for c in range(num_input_fold_4d[1]):
          if not is_buffer:
            filename.write("LOAD_INPUT(%d:%d, %d:%d, %d:%d, %d:%d) \n"%(n*Ns, (n+1)*Ns-1, h*Hs, (h+1)*Hs-1, w*Ws, (w+1)*Ws-1, c*Cs, (c+1)*Cs-1))
            size += inp_size
            if 2*size > accumBufferSize:
              filename.write("STORE_OUTPUT(%d:%d, %d:%d, %d:%d, %d:%d) \n"%(nprev*Ns, (n+1)*Ns-1, hprev*Hs, (h+1)*Hs-1, wprev*Ws, (w+1)*Ws-1, cprev*Cs, (c+1)*Cs-1))
              nprev = n
              hprev = h
              wprev = w
              cprev = c
          filename.write("COMPUTE_VECTORALU \n")



##TODO: Not handling batch size > 1
## Assumptions - Assuming H = W in NCHW and R = S in MCRS.. TODO change
def mapping_setup():

  ## A channel of the filter (C'< C, C' = nRows) is mapped to the rows 
  ## Multiple channels across filters (M' < M, M' = nCols) is mapped across the columns
  ## For one systolic Fold, M' x C' x 1 x 1 filters are stationary in the array for a M x C x R x S filter dimension
  ## M/M' x C/C' x R x S is the number of systolic folds to calculate convolution over all M -> CxRxS filters
  ## This fold is computed here -- 4D value for iterators in the future, num_fold gives the final value
  ## Note: This type of mapping works very badly if your channel_size is very small (First layer - 224x224x3 -- Have very poor utilization)
  filter_map_fold_dims = np.array([nCols, nRows, 1, 1])  
  num_filter_fold_4d = np.ceil(np.divide(filter_dims, filter_map_fold_dims))
  num_filter_fold_4d = num_filter_fold_4d.astype(int)
  num_filter_fold = np.prod(num_filter_fold_4d)
  print("Filter fold: " + str(num_filter_fold_4d))

  ## However the num_fold shown above is not enough if the input feature map size is greater than what the input buffer can hold
  ## In which case we need to calculate a similar value for inputs
  ## This is computed here -- filter_folds*inputs_folds give you the total number of schedule

  avail_HpWp = math.ceil(inpBufferSize / nRows)  ## nRows should be mapped to enable systolic flow -- hence size/nRows is inevitable
  avail_HpWp = min(np.ceil(math.sqrt(avail_HpWp)), H)

  inp_NCpHW = np.array([N, C, H, W])
  inp_NCpHpWp = np.array([N, nRows, avail_HpWp, avail_HpWp])
  inp_size = np.prod(inp_NCpHpWp)
  num_input_fold_4d = np.ceil(np.divide(inp_NCpHW, inp_NCpHpWp))
  num_input_fold_4d = num_input_fold_4d.astype(int)
  num_input_fold = np.prod(num_input_fold_4d)

  print("Input fold: " + str(num_input_fold_4d))
  tot_systolic_folds = num_filter_fold * num_input_fold

  ### TODO modify this code to directly generate ISA binary instead of Pseudo-ops
  if opType == 'Conv':
    spit_ISA_conv(num_filter_fold_4d, num_input_fold_4d, inp_NCpHW, filter_dims, tot_systolic_folds, 'GEMM')
  elif opType == 'ReLU' or opType == 'MaxPool2D':
    spit_ISA_relu_maxpool2d(True, num_input_fold_4d, inp_size)



def main():
  print(nRows, nCols, bufferGranularity, inpBufferSize, accumBufferSize, weightBufferSize)
  print(inp_dims, filter_dims)
  mapping_setup()



if __name__ == "__main__":
  main()

import numpy as np 



'''
Surya
Assumption:
Since Weight Stationary Implementation requires more psum accumulation, the output buffer size needs to be more. 
Refer : Scale Sim, Eyeriss
Input Layout: NHWC
Weight Layout: RSCM

Fixing a Loop Order: R -> S -> C -> M -> N -> H -> W
Tilable: H, W, C, M

Approach:
1. Load Weights to Systolic Array
    Opitmal Approach: Fix C' and M' to the systolic dimensions so that all except the last fold is fully utilized.
    Exploration: 
        Invalid: Declare all configurations having C' * M' greater than Weight buffer size as Invalid.
        Valid: But will be underutilized.

2. Load Input Tensor
    C' is Fixed in previous step.
    Exploration: 
        Invalid: Declare all configurations having C' * W' * H' greater than Input buffer size as Invalid.
        Valid: ~  

Concern: Partial Sum Logic
'''


def gen_load_instr(DRAM=0,SRAM=0,Z_SIZE = 0,Z_STRIDE = 1,Y_SIZE = 0, Y_STRIDE = 1, X_SIZE = 0, RESET = 0):

    insn = "Load, " + " DRAM  = " + str(DRAM) + " SRAM  = " + str(SRAM) + " Z_SIZE  = " + str(Z_SIZE) + " Z_STRIDE  = " + str(Z_STRIDE) + " Y_SIZE  = " + str(Y_SIZE) + " Y_STRIDE  = " + str(Y_STRIDE) + " X_SIZE = " + str(X_SIZE) + " RESET = " + str(RESET)   

    return insn

def gen_compute_instr(  input  = 0,
                        output = 0,
                        weight = 0,
                        H = 0,
                        W = 0,
                        Stride = [1,1],
                        Pad = [0,0,0,0],
                        preload = 0):

    insn = "Compute " + " input  = " + str(input) + " output  = " + str(output) + " weight  = " + str(weight) + " H  = " + str(H) + " W  = " + str(W) + " Stride  = [" + str(Stride[0]) + "," + str(Stride[1]) + \
           "] Pad = [ " + str(Pad[0]) + "," + str(Pad[1]) + "," + str(Pad[2]) + "," + str(Pad[3]) + "]"   + " preload = " + str(preload)   

    return insn


def systolic_fold_conv(ConvParams,TileParams,):
    N,C,H,W = ConvParams[0]
    R,S,M   = ConvParams[1]
    E,F     = ConvParams[2]
    Sx,Sy   = ConvParams[3]
    P_left,P_right,P_top,P_bottom   = ConvParams[4]

    C1,C2   = TileParams[0]
    M1,M2   = TileParams[1]
    E1,E2   = TileParams[2]
    F1,F2   = TileParams[3]
    
    #Now we have to deal with the inputs

    # We have the following variables here : i,j, time
    # Calculate the area of effect of the filter ^ Output tile ^ channel : Let this tile be of size H_eff*W_eff
    # In this tile, We have, input at clock cycle t , in the ith row should be row major flattened input
    # of all pixels with (x > Sx*r, y >  Sy*s), With offset of 1/column as the columns go along


    #Now, Where to store the outputs:

    # If you have chosen row major previously, your outputs will also be in row major order, U will have to accumulate them 
    # in a buffer

    #Things to do:
    '''
    What the solution might look like(Initial Draw up)
    Load input : 
        Find expression for which slice of the input to load, using the parameters above
    Load weights in the systolic:
        Find expression for which slice of the input to load, using the parameters above 
    Compute Systolic.:
        Find expression for number of times this needs to happen, and can we trigger a future load appropriately here(Hence generating the instruction dependencies)
    Store output. :
        Which section to store needs to e computed
    '''

    #Key Points
    '''
    Input Channels map to rows of systolic
    Output Channels map to columns of systolic
    Weight stationary Implementation, fixes ordering for some loops
    '''

    #ASSUMPTION : OUTPUT FITS ON BUFFER
    #The next 2 loops are interchangable
    for r in range(R):
        for s in range(S):
            #These 2 loops also can be interchanged
            for c1 in range(C1):
                for m1 in range(M1):
                    # in the systolic element i(row) ,j(column)
                    # Load Weight[r,s, c1*C2 + i, m1*M2 + j] #RSCM

                    for e1 in range(E1):
                        for f1 in range(F1):
                            #Load input [
                            #               N,
                            #               c1*C2<C<(c1+1)*C2,
                            #               (e1*E2) + r*Sx < H < (e1*E2+E2) + r*Sx,
                            #               (f1*F2) + s*Sy < W < (f1*F2+F2) + s*Sy  
                            #           ]
                            
                            #Compute
                            #   H'  = e1?
                            #   W'  = f1?s
                            #   strideX,strideY  = 1,1 ?
                            #   Padding from params directly
                            #   
                            # I think these work, but I dont think compute instruction 
                            # was designed with this in mind

                            x = 0 
    #Flush Outputs 


            # Load Weights Here, using the following logic :
            # column j of systolic has the jth output filter
            # row i of the systolic will have the weights corresponding to the ith input channel
            # so in the element i,j, you have Weight(j , i, r , s) (in MCRS fashion, (Not sure if this is the convention we have to follow))


    return ""


def mapper(ConvParams, TileParams,op):
    #TEMPORARY CODE Begin
    assert op == "conv"
    #TEMPORARY CODE End
    N    = ConvParams[0][0]
    C1,C2   = TileParams[0]
    M1,M2   = TileParams[1]
    E1,E2   = TileParams[2]
    F1,F2   = TileParams[3]

    #Add read from file functionality here

    NUM_ROWS = 64
    NUM_COLS = 64


    WEIGHT_BUFFER_SIZE = 1e6

    INPUT_BUFFER_SIZE = 1e6

    OUTPUT_BUFFER_SIZE = 1e6
    
    if(C2 > NUM_ROWS):
        print("C2 too high")
        return "", 0
    if(M2 > NUM_COLS):
        print("M2 too high")
        return "", 0

    if(WEIGHT_BUFFER_SIZE < C2*M2):
        print("Weight buffers too small")
        return "",0

    if(INPUT_BUFFER_SIZE < N * C2 * E2 * F2):
        print("Input Buffer too small")
        return "", 0
    
    if(OUTPUT_BUFFER_SIZE < N * E1*E2 * F1*F2 * M1*M2):
        print("Output Buffer too small")
        return "",0
        
    lower = ""
    #TODO Check illegal configurations

    #Lets just see 1 tile first :
    '''
    for n in range(N):
        for m in range(M1):
            for e in range(E1):
                for f in range(F1):
                    #Output = 0 here
                    for c in range(C1):
                        tile_pos = [n,c,m,e,f]
    '''
    lower += systolic_fold_conv(ConvParams,TileParams)

    validity = 1

                    #Flush outputs here







    return lower,validity
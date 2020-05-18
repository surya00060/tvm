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

def systolic_fold_conv(ConvParams,TileParams,tile_pos,):
    N,C,H,W = ConvParams[0]
    R,S,M   = ConvParams[1]
    E,F     = ConvParams[2]
    Sx,Sy   = ConvParams[3]
    Px,Py   = ConvParams[4]

    C1,C2   = TileParams[0]
    M1,M2   = TileParams[1]
    E1,E2   = TileParams[2]
    F1,F2   = TileParams[3]

    n,c,m,e,f = tile_pos


    
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

    #The next 2 loops are interchangable
    for r in range(R):
        for s in range(S):
            # Load Weights Here, using the following logic :
            # column j of systolic has the jth output filter
            # row i of the systolic will have the weights corresponding to the ith input channel
            # so in the element i,j, you have Weight(j , i, r , s) (in MCRS fashion, (Not sure if this is the convention we have to follow))


            #Now we have to deal with the inputs

            # We have the following variables here : i,j, time
            # Calculate the area of effect of the filter ^ Output tile ^ channel : Let this tile be of size H_eff*W_eff
            # In this tile, We have, input at clock cycle t , in the ith row should be row major flattened input
            # of all pixels with (x > Sx*r, y >  Sy*s), With offset of 1/column as the columns go along


            #Now, Where to store the outputs:
            
            # If you have chosen row major previously, your outputs will also be in row major order, U will have to accumulate them 
            # in a buffer




            dummy =0 # Code to avoid syntax error
    

    return ""


def mapper(ConvParams, TileParams,op):
    #TEMPORARY CODE Begin
    assert op = "conv"
    #TEMPORARY CODE End
    N    = ConvParams[0,0]
    C1,C2   = TileParams[0]
    M1,M2   = TileParams[1]
    E1,E2   = TileParams[2]
    F1,F2   = TileParams[3]


    lower = ""
    #TODO Check illegal configurations

    #Lets just see 1 tile first :
    for n in range(N):
        for m in range(M1):
            for e in range(E1):
                for f in range(F1):
                    #Output = 0 here
                    for c in range(C1):
                        tile_pos = [n,c,m,e,f]
                        lower += systolic_fold_conv(ConvParams,TileParams,tile_pos)

                    #Flush outputs here







    return lower
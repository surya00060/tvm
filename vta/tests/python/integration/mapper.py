import numpy as np 

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
    Weight stationary Implementation, fixes ordering to some e
    '''
    


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
        for c in range(C1):
            for m in range(M1):
                for e in range(E1):
                    for f in range(F1):
                        tile_pos = [n,c,m,e,f]
                        lower += systolic_fold_conv(ConvParams,TileParams,tile_pos)







    return lower
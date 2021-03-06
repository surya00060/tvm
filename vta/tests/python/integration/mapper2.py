import numpy as np

def gen_load_instr(DRAM=0,SRAM=0,Z_SIZE = 0,Z_STRIDE = 1,Y_SIZE = 0, Y_STRIDE = 1, X_SIZE = 0, RESET = 0):

    # insn = "Load, " + " DRAM  = " + str(DRAM) + " SRAM  = " + str(SRAM) + " Z_SIZE  = " + str(Z_SIZE) + " Z_STRIDE  = " + str(Z_STRIDE) + " Y_SIZE  = " + str(Y_SIZE) + " Y_STRIDE  = " + str(Y_STRIDE) + " X_SIZE = " + str(X_SIZE) + " RESET = " + str(RESET)  + "\n" 
    insn = {'type': 'load',
            'pop_prev_dep':0,
            'pop_next_dep':0,
            'push_prev_dep':0,
            'push_next_dep':0,
            'DRAM':DRAM,
            'SRAM':SRAM,
            'Z_SIZE': Z_SIZE,
            'Z_STRIDE': Z_STRIDE,
            'Y_SIZE': Y_SIZE,
            'Y_STRIDE': Y_STRIDE,  
            'X_SIZE': X_SIZE,
            'RESET':RESET,
    }
    return insn

def gen_store_instr(DRAM=0,SRAM=0,Z_SIZE = 0,Z_STRIDE = 1,Y_SIZE = 0, Y_STRIDE = 1, X_SIZE = 0, RESET = 0):

    # insn = "Store, " + " DRAM  = " + str(DRAM) + " SRAM  = " + str(SRAM) + " Z_SIZE  = " + str(Z_SIZE) + " Z_STRIDE  = " + str(Z_STRIDE) + " Y_SIZE  = " + str(Y_SIZE) + " Y_STRIDE  = " + str(Y_STRIDE) + " X_SIZE = " + str(X_SIZE) + " RESET = " + str(RESET)  + "\n" 
    insn = {'type': 'store',
            'pop_prev_dep':1,
            'pop_next_dep':0,
            'push_prev_dep':0,
            'push_next_dep':0,
            'DRAM':DRAM,
            'SRAM':SRAM,
            'Z_SIZE': Z_SIZE,
            'Z_STRIDE': Z_STRIDE,
            'Y_SIZE': Y_SIZE,
            'Y_STRIDE': Y_STRIDE,  
            'X_SIZE': X_SIZE,
            'RESET':RESET,
    }
    return insn
    
def gen_compute_instr(  input  = 0,
                        output = 0,
                        weight = 0,
                        H = 0,
                        W = 0,
                        Stride = [1,1],
                        Pad = [0,0,0,0],
                        preload = 0):

    # insn = "Compute " + " input  = " + str(input) + " output  = " + str(output) + " weight  = " + str(weight) + " H  = " + str(H) + " W  = " + str(W) + " Stride  = [" + str(Stride[0]) + "," + str(Stride[1]) + \
    #       "] Pad = [ " + str(Pad[0]) + "," + str(Pad[1]) + "," + str(Pad[2]) + "," + str(Pad[3]) + "]"   + " preload = " + str(preload)   + "\n"
    # print('compute')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    insn = {'type': 'compute',
            'pop_prev_dep':1,
            'pop_next_dep':0,
            'push_prev_dep':0,
            'push_next_dep':0,
            'input': input,
            'output':output,
            'weight':weight,
            'H':H,
            'W':W,
            'Stride': Stride,
            'Pad':Pad,
            'preload':preload,
    }
    return insn



def mapper_conv2d(ConvParams, TileParams):
    print(TileParams, ConvParams)
    N,C,H,W             = ConvParams[0]
    R,S,M               = ConvParams[1]
    E,F                 = ConvParams[2]
    Sx,Sy               = ConvParams[3]
    Pxl,Pxr,Pyt,Pyb     = ConvParams[4]


    C1,C2   = TileParams[0]
    M1,M2   = TileParams[1]
    E1,E2   = TileParams[2]
    F1,F2   = TileParams[3]

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
    
    #Calculate the max number of R,S That can be fed into Weight buffer

    '''
    #introduce multiple loads in next iteration, based on duration of 1 load instr
    max_RS = WEIGHT_BUFFER_SIZE/(C2*M2)

    curr_rs_ptr  = 0 

    '''
    
    curr_layer_DRAM_weight_ptr = 0
    curr_layer_DRAM_input_ptr = 0
    curr_layer_DRAM_output_ptr = 0

    SRAM_weight_base = 0

    SRAM_input_base = 0

    SRAM_output_base = 0

    code = []
    for n in range(N):
        for e1 in range(E1):
            for f1 in range(F1):
                for m1 in range(M1):
                #Flush Output Buffer
                    for c1 in range(C1):
                        for r in range(R):
                            for s in range(S):
                                #RSCM
                                curr_weight_ptr = curr_layer_DRAM_weight_ptr + r * S * C * M + s * C * M + (c1*C2)*M + m1* M2   

                                load_weight  = gen_load_instr(DRAM = curr_weight_ptr, SRAM = SRAM_weight_base, Z_SIZE = M2, Y_SIZE= C2, X_SIZE = 1)
                                code.append(load_weight)


                                curr_input_ptr = curr_layer_DRAM_input_ptr + n*C*H*W + c1*C2*H*W + ((e1*E2) + r*Sx)*W + (f1*F2) + s*Sy
                                load_input = gen_load_instr(DRAM = curr_input_ptr, SRAM = SRAM_input_base,Z_SIZE=C2,Y_SIZE=E2,X_SIZE=F2) #Not sure is E2 and F2 need to be reordered here

                                code.append(load_input)

                                compute_instr = gen_compute_instr(input=SRAM_input_base,output = SRAM_output_base,weight = SRAM_weight_base,H=E2,W =F2,Stride=[1,1],Pad=ConvParams[4],preload = (c1==0)) #This means load 0s/flush output
                                code.append(compute_instr)
                
                curr_output_ptr = curr_layer_DRAM_output_ptr + n*M*E*F + m1*M2*E*F + e1*E2*F + f1*F2
                store_output = gen_store_instr(DRAM=curr_output_ptr,SRAM = SRAM_output_base,Z_SIZE=M2,Y_SIZE=E2,X_SIZE=F2) #Not sure is E2 and F2 need to be reordered here

                code.append(store_output)

    #Depedency Flags
    #Corrently added for one layer
    for i in range((len(code)-1)):
        if code[i]['type'] == 'load' and code[i+1]['type'] == 'compute':
            code[i]['push_next_dep'] = 1
            code[i+1]['pop_prev_dep'] = 1 #Obvious
        
        elif code[i]['type'] == 'compute' and code[i+1]['type'] == 'store':
            code[i]['push_next_dep'] = 1
            code[i+1]['pop_prev_dep'] = 1 #Obvious

    return code,1

                            #    Load input [
                            #               N,
                            #               c1*C2<C<(c1+1)*C2,
                            #               (e1*E2) + r*Sx < H < (e1*E2+E2) + r*Sx,
                            #               (f1*F2) + s*Sy < W < (f1*F2+F2) + s*Sy  
                            #           ]



def mapper(ConvParams,TileParams,op):
    if(op == "conv"):
        return mapper_conv2d(ConvParams,TileParams)


def mapper_dense(DenseParams,TileParams,op):
    assert op == "dense"
    N,M,K  = DenseParams
    N1,N2 = TileParams[0]
    M1,M2 = TileParams[1]
    K1,K2 = TileParams[2]

    NUM_ROWS = 64
    NUM_COLS = 64

    WEIGHT_BUFFER_SIZE = 1e6

    INPUT_BUFFER_SIZE = 1e6

    OUTPUT_BUFFER_SIZE = 1e6

    code  = []

    if(M2 > NUM_COLS):
        return [], 0
    if(K2 > NUM_ROWS):
        return [], 0
    if(M2*N2 > OUTPUT_BUFFER_SIZE):
        return [],0
    if(N2*K2 > INPUT_BUFFER_SIZE):
        return [],0

     
    curr_layer_DRAM_weight_ptr = 0
    curr_layer_DRAM_input_ptr = 0
    curr_layer_DRAM_output_ptr = 0

    SRAM_weight_base = 0

    SRAM_input_base = 0

    SRAM_output_base = 0

    for n1 in range(N1):
        for m1 in range(M1):
            for k1 in range(K1):
                # Load inputs
                # Assuming N,CHW

                curr_input_ptr = curr_layer_DRAM_input_ptr + n1*N2*K + k1*K2
                load_input = gen_load_instr(DRAM=curr_input_ptr,SRAM=SRAM_input_base,Z_SIZE= K2,Y_SIZE = N2, X_SIZE=1) # Not sure about X,Y,Z Here
                code.append(load_input)
                # Load Weights
                # Assuming CHW, M (Similar to RSCM) 
                curr_weight_ptr = curr_layer_DRAM_weight_ptr + k1*K2*M + m1*M2
                load_weight = gen_load_instr(DRAM=curr_weight_ptr,SRAM=SRAM_weight_base,Z_SIZE= M2,Y_SIZE = K2, X_SIZE=1) # Not sure about X,Y,Z Here
                code.append(load_input)

                compute_instr = gen_compute_instr(input=SRAM_input_base,output = SRAM_output_base,weight = SRAM_weight_base,H=1,W =N2,Stride=[1,1],Pad = [0,0,0,0],preload = (m1==0)) 
                code.append(compute_instr)# May have to change H,W in the previous line, depending on the parser. 


            # Store Outputs N, M
            curr_output_ptr = curr_layer_DRAM_output_ptr + n1*N2*M + m1*M2  
            store_output = gen_store_instr(DRAM=curr_output_ptr,SRAM = SRAM_output_base,Z_SIZE=M2,Y_SIZE=N2,X_SIZE=1) # Not sure about X,Y,Z Here
            code.append(store_output)

                # COmpute systolic
    for i in range((len(code)-1)):
        if code[i]['type'] == 'load' and code[i+1]['type'] == 'compute':
            code[i]['push_next_dep'] = 1
            code[i+1]['pop_prev_dep'] = 1 #Obvious
        
        elif code[i]['type'] == 'compute' and code[i+1]['type'] == 'store':
            code[i]['push_next_dep'] = 1
            code[i+1]['pop_prev_dep'] = 1 #Obvious

    return code,1
                 






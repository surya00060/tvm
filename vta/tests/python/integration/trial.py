import os
import numpy as np

import tvm
from tvm import te
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_runtime as runtime

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple

import vta
from vta import program_fpga, reconfig_runtime
import vta.testing
from vta.testing import simulator

from mapper2 import *

import re

import tvm.relay.backend.graph_runtime_codegen

env = vta.get_env()

target = "llvm"

batch_size = 1
dtype = "float32"
model_name = "resnet-18"
log_file = "%s.log" % model_name
graph_opt_sch_file = "%s_graph_opt.log" % model_name


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3,stride = 1,padding = 1)
        self.fc1 = nn.Linear(65536, 10)
        self.maxpool = nn.AdaptiveMaxPool2d(32)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = x.view(-1, 65536)
        x = F.softmax(self.fc1(x),dim = 1)
        return x

    
def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape =  [1, 64, 64, 64]
    output_shape = (batch_size, 10)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'squeezenet_v1.1':
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        model = Net().eval()
        input_shape = [1, 64, 64, 64]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()
        input_name = 'input0'
        shape_list = [(input_name, input_data.shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model,
                                          shape_list)        
        net = mod["main"]
        

        net = relay.Function(net.params, net.body, None, net.type_params, net.attrs)
        #mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape,net


def parse_conv(t):
    #print(t)
    N,C,H,W = t.args[0][1]

    M,_,R,S = t.args[1][1]

    Sx,Sy = t.args[2]

    pad_t, pad_l, pad_r, pad_b = t.args[3]

    assert pad_r == pad_l
    assert pad_b == pad_t
    #print(t.workload())
    print(t.args)
    E = ((H+ 2*pad_t - R)//Sx) + 1
    F = ((W+ 2*pad_l - S)//Sy) + 1

    return N,C,H,W,R,S,M,E,F,Sx,Sy,pad_l,pad_r,pad_t,pad_b

def parse_dense(t):
    print(t.args[1][1])

    N,_ = t.args[0][1]
    M,K = t.args[1][1]

    return N,M,K



class RelayTensor:
    def __init__(self,name,size,dependencies = None,op = None,dram_addr = 0):
        self.name = name
        self.size = size

        self.dram_addr = dram_addr

        self.dependencies =dependencies
        self.op = op

    
def parse_inputs(inp):
    elem = inp.split(":")
    print(elem)
    name = elem[0]
    reg = re.compile("([^:]*)")
    x = reg.match(elem[1])
    print("X : ",x.start())
    size = []
    size.append(x) 
    return elem[0] 


'''
def generate_graph(mod,params,data_shape,out_shape):
    #Break the relay code down into a list
    ir = str(mod["main"]).split('\n')
    print(ir[0])

    #Step 1 generate a list of inputs, and generate their addresses also
    
    nodes = []

    inputs = (ir[0].split("%"))
    for i in inputs:
        if(i == 'fn ('):
            continue
        nodes.append(parse_inputs(i))
        #print(i)

    


    input_to_net  =  ir[0]



        

    

    return 0,0,0,0,0
'''



def convert_to_ops(mod,params,data_shape,out_shape):
    nodes = []
    node_index_when_last_used = []

    #First lets add the input nodes 

    task_conv = autotvm.task.extract_from_program(mod["main"], target=target,
                                        params=params,
                                        ops=(relay.op.get("nn.conv2d"),))

    

    print(params.keys())

    for k in params.keys():
        nodes.append(RelayTensor(k,params[k].shape))

    nodes.append(RelayTensor("input0",data_shape))

    ir = mod["main"]

    ir = str(ir).split("\n")
    conv_index = 0

    for line in ir:
        if('nn.conv2d' in line):
            task_conv[conv_index] =  0


    '''
    for p in params:
        print(p)
    '''
    return 






def tune_and_evaluate():
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, data_shape, out_shape,net = get_network("mxnet",1)

    print(mod["main"])
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"),))

    #print(tasks)

    env = vta.get_env()
    '''
    for t in tasks:
        with env.target: 
            N,C,H,W,R,S,M,E,F,Sx,Sy,Pxl,Pxr,Pyt,Pyb = parse_conv(t)
            print("N : ",N)
            print("C : ",C)
            print("H : ",H)
            print("W : ",W)

            print("R : ",R)
            print("S : ",S)
            print("M : ",M)

            print("E : ",E)
            print("F : ",F)
            print("Sx : ",Sx)
            print("Sy : ",Sy)
            print("Pxl : ",Pxl)
            print("Pxr : ",Pxr)
            print("Pyl : ",Pyt)
            print("Pyr : ",Pyb)

            NCHW = [N,C,H,W]
            RSM  = [R,S,M]
            EF   = [E,F]
            S    = [Sx,Sy]
            P    = [Pxl,Pxr,Pyt,Pyb]

            ConvParams = [NCHW, RSM , EF , S , P]

            print(t.config_space.__len__())
            #print(t)
            for i in range(t.config_space.__len__() - 10,t.config_space.__len__()):
                d = t.config_space.get(i).to_json_dict()['entity']
                print(d)
                tile_C = [C//d[0][-1][-1],d[0][-1][-1]]
                tile_M = [M//d[1][-1][-1],d[1][-1][-1]]
                tile_E = [E//d[2][-1][-1],d[2][-1][-1]]
                tile_F = [F//d[3][-1][-1],d[3][-1][-1]]
                
                print("tile_C : ",tile_C)
                print("tile_M : ",tile_M)
                print("tile_E : ",tile_E)
                print("tile_F : ",tile_F)
                
                TileParams = [tile_C,tile_E,tile_F,tile_M]
                lowered,validity = mapper(ConvParams,TileParams,"conv")
                if(validity == 0): 
                    continue  

            
            exit()
            s, args = t.instantiate(t.config_space.get(4))
            #print(s)
            print(args[0].shape)
            print(args[1])
            print(args[2])
            


            #print(tvm.lower(s, args, simple_mode=True))
            
    '''
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.dense"),))

    print(tasks)

    for t in tasks:
        with env.target:
            N,M,K = (parse_dense(t))
            print("N :" , N)        # N ==== TILE Y
            print("M :" , M)        # M ==== TILE X
            print("K :" , K)        # K ==== TILE K
            print(t.config_space)

            for i in range(t.config_space.__len__()):
                d = t.config_space.get(i).to_json_dict()['entity']
                print(d[0])
                
                tile_N = [N//d[0][-1][-1],d[0][-1][-1]]
                tile_M = [M//d[1][-1][-1],d[1][-1][-1]]
                tile_K = [K//d[2][-1][-1],d[2][-1][-1]]

                tile_params = [tile_N,tile_M,tile_K]
                dense_params = [N,M,K]
                lowered,validity = mapper_dense(dense_params,tile_params,"dense")


                if(validity == 0):
                    continue
                


    exit()


    print(mod["main"])

    # run tuning tasks
    print(tasks)

    #tune_kernels(tasks, **tuning_opt)
    #tune_graph(mod["main"], data_shape, log_file, graph_opt_sch_file)

    print(tasks)
    '''
    # compile kernels with graph-level best records
    with autotvm.apply_graph_best(graph_opt_sch_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(mod, target=target, params=params)
            print(lib.imported_modules.get_source())

            

        # upload parameters to device
        ctx = tvm.cpu()
        data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
        module = runtime.create(graph, lib, ctx)
        module.set_input(input_name, data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=100, repeat=3)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
                (np.mean(prof_res), np.std(prof_res)))
    '''

# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

tune_and_evaluate()



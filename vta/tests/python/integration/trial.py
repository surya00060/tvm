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
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.fc1 = nn.Linear(32 * 222 * 222, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 1577088)
        x = F.softmax(self.fc1(x),dim = 1)
        return x

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

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
        input_shape = [1, 3, 224, 224]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()
        input_name = 'input0'
        shape_list = [(input_name, input_data.shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model,
                                          shape_list)        
        net = mod["main"]
        

        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape



def tune(t):
    Workload = namedtuple("Conv2DWorkload",
                      ['batch', 'height', 'width', 'in_filter', 'out_filter',
                       'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])
    
    print(t)


def tune_and_evaluate():
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, data_shape, out_shape = get_network("mxnet",1)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"),))


    for t in tasks:
        with env.target_vta_cpu : 
            print(t.config_space)
            print(t.__class__.__name__)
            s, args = t.instantiate(t.config_space.get(1))
            print(s)
            print(args)
            print(tvm.lower(s, args, simple_mode=True))
        
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



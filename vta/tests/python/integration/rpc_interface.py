import logging
import sys
import json
import os

import pytest
import numpy as np
from collections import namedtuple

import tvm
from tvm import te
from tvm import relay
from tvm import autotvm
from tvm.contrib import util
from tvm.contrib.pickle_memoize import memoize
import topi
import topi.testing
import vta
from vta import program_fpga, reconfig_runtime
import vta.testing
from vta.testing import simulator


def build_run(s,):
    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    kernel = te.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)
    bias = te.placeholder(bias_shape, name="bias", dtype=env.acc_dtype)
    padding = relay.nn.get_pad_tuple2d((wl.hpad, wl.wpad))
    
    if "vta" in target.keys:
        mod = vta.build(s, [data, kernel, bias, res],
                        target=target,
                        target_host=env.target_host,
                        name="conv2d")
    else:
        mod = tvm.build(s, [data, kernel, bias, res],
                        target=target,
                        target_host=env.target_host,
                        name="conv2d")
    temp = util.tempdir()
    mod.save(temp.relpath("conv2d.o"))
    remote.upload(temp.relpath("conv2d.o"))
    f = remote.load_module("conv2d.o")
    ctx = remote.context(str(target))

    res_np = np.zeros(topi.util.get_const_tuple(res.shape)).astype(res.dtype)
    data_arr = tvm.nd.array(data_np, ctx)
    kernel_arr = tvm.nd.array(kernel_np, ctx)
    bias_arr = tvm.nd.array(bias_np, ctx)
    res_arr = tvm.nd.array(res_np, ctx)
    time_f = f.time_evaluator("conv2d", ctx, number=samples)

    # In vta sim mode, collect simulator runtime statistics
    stats = {}
    cost = None
    if env.TARGET in ["sim", "tsim"]:
        # Check if we're in local RPC mode (allows us to rebuild the
        # runtime on the fly when varying the VTA designs)
        local_rpc = int(os.environ.get("VTA_LOCAL_SIM_RPC", "0"))
        print(local_rpc)
        if local_rpc:
            print('Local RPC')
            if env.TARGET == "sim":
                remote.get_function("vta.simulator.profiler_clear")()
            else:
                remote.get_function("vta.tsim.profiler_clear")()
            cost = time_f(data_arr, kernel_arr, bias_arr, res_arr)
            if env.TARGET == "sim":
                stats = json.loads(remote.get_function("vta.simulator.profiler_status")())
            else:
                stats = json.loads(remote.get_function("vta.tsim.profiler_status")())
        else:
            simulator.clear_stats()
            cost = time_f(data_arr, kernel_arr, bias_arr, res_arr)
            stats = simulator.stats()
            print(cost, stats)
    else:
        cost = time_f(data_arr, kernel_arr, bias_arr, res_arr)
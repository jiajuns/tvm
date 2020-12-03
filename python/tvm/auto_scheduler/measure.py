# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Distributed measurement infrastructure to measure the runtime costs of tensor programs.

These functions are responsible for building the tvm module, uploading it to
remote devices, recording the running time costs, and checking the correctness of the output.

We separate the measurement into two steps: build and run.
A builder builds the executable binary files and a runner runs the binary files to
get the measurement results. The flow of data structures is

  .               `ProgramBuilder`                 `ProgramRunner`
  `MeasureInput` -----------------> `BuildResult` ----------------> `MeasureResult`

We implement these in python to utilize python's multiprocessing and error handling.
"""

import os
import time
import shutil
import tempfile
import multiprocessing

import tvm._ffi
from tvm.runtime import Object, module, ndarray
from tvm.driver import build_module
from tvm.ir import transform
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
from tvm.contrib import tar, ndk

from . import _ffi_api
from .loop_state import StateObject
from .utils import (
    call_func_with_timeout,
    check_remote,
    get_const_tuple,
    make_traceback_info,
    request_remote,
)
from .compute_dag import ComputeDAG
from .search_task import SearchTask
from .workload_registry import (
    serialize_workload_registry_entry,
    deserialize_workload_registry_entry,
)


# The time cost for measurements with errors
# We use 1e10 instead of sys.float_info.max for better readability in log
MAX_FLOAT = 1e10


@tvm._ffi.register_object("auto_scheduler.MeasureCallback")
class MeasureCallback(Object):
    """ The base class of measurement callback functions. """


@tvm._ffi.register_object("auto_scheduler.MeasureInput")
class MeasureInput(Object):
    """Store the input of a measurement.

    Parameters
    ----------
    task : SearchTask
        The SearchTask of this measurement.
    state : Union[State, StateObject]
        The State to be measured.
    """

    def __init__(self, task, state):
        state = state if isinstance(state, StateObject) else state.state_object
        self.__init_handle_by_constructor__(_ffi_api.MeasureInput, task, state)

    def serialize(self):
        """Custom serialization to workaround MeasureInput not exposing all its
        members to the TVM ffi interface.

        Note that we do not implement __getstate__ as it does not seem to work
        with initialization of the workload registry (maybe because of
        initialization order?).
        """
        return [
            _ffi_api.SerializeMeasureInput(self),
            serialize_workload_registry_entry(self.task.workload_key),
        ]

    @staticmethod
    def deserialize(data):
        inp = _ffi_api.DeserializeMeasureInput(data[0])
        deserialize_workload_registry_entry(data[1])
        return recover_measure_input(inp)


@tvm._ffi.register_object("auto_scheduler.BuildResult")
class BuildResult(Object):
    """Store the result of a build.

    Parameters
    ----------
    filename : Optional[str]
        The filename of built binary file.
    args : List[Tensor]
        The arguments.
    error_no : int
        The error code.
    error_msg : Optional[str]
        The error message if there is any error.
    time_cost : float
        The time cost of build.
    """

    def __init__(self, filename, args, error_no, error_msg, time_cost):
        filename = filename if filename else ""
        error_msg = error_msg if error_msg else ""

        self.__init_handle_by_constructor__(
            _ffi_api.BuildResult, filename, args, error_no, error_msg, time_cost
        )


@tvm._ffi.register_object("auto_scheduler.MeasureResult")
class MeasureResult(Object):
    """Store the results of a measurement.

    Parameters
    ----------
    costs : List[float]
        The time costs of execution.
    error_no : int
        The error code.
    error_msg : Optional[str]
        The error message if there is any error.
    all_cost : float
        The time cost of build and run.
    timestamp : float
        The time stamps of this measurement.
    """

    def __init__(self, costs, error_no, error_msg, all_cost, timestamp):
        error_msg = error_msg if error_msg else ""

        self.__init_handle_by_constructor__(
            _ffi_api.MeasureResult, costs, error_no, error_msg, all_cost, timestamp
        )


def recover_measure_input(inp, rebuild_state=False):
    """
    Recover a deserialized MeasureInput by rebuilding the missing fields.
    1. Rebuid the compute_dag in inp.task
    2. (Optional) Rebuild the stages in inp.state

    Parameters
    ----------
    inp: MeasureInput
        The deserialized MeasureInput
    rebuild_state: bool = False
        Whether rebuild the stages in MeasureInput.State

    Returns
    -------
    new_input: MeasureInput
        The fully recovered MeasureInput with all fields rebuilt.
    """
    task = inp.task
    new_task = SearchTask(
        ComputeDAG(task.workload_key),
        task.workload_key,
        task.target,
        task.target_host,
        task.hardware_params,
    )

    if rebuild_state:
        new_state = new_task.compute_dag.infer_bound_from_state(inp.state)
    else:
        new_state = inp.state

    return MeasureInput(new_task, new_state)


@tvm._ffi.register_object("auto_scheduler.ProgramBuilder")
class ProgramBuilder(Object):
    """ The base class of ProgramBuilders. """

    def build(self, measure_inputs, verbose=1):
        """Build programs and return results.

        Parameters
        ----------
        measure_inputs : List[MeasureInput]
            A List of MeasureInput.
        verbose: int = 1
            Verbosity level. 0 for silent, 1 to output information during program building.

        Returns
        -------
        res : List[BuildResult]
        """
        return _ffi_api.ProgramBuilderBuild(self, measure_inputs, verbose)


@tvm._ffi.register_object("auto_scheduler.ProgramRunner")
class ProgramRunner(Object):
    """ The base class of ProgramRunners. """

    def run(self, measure_inputs, build_results, verbose=1):
        """Run measurement and return results.

        Parameters
        ----------
        measure_inputs : List[MeasureInput]
            A List of MeasureInput.
        build_results : List[BuildResult]
            A List of BuildResult to be ran.
        verbose: int = 1
            Verbosity level. 0 for silent, 1 to output information during program running.

        Returns
        -------
        res : List[MeasureResult]
        """
        return _ffi_api.ProgramRunnerRun(self, measure_inputs, build_results, verbose)


@tvm._ffi.register_object("auto_scheduler.ProgramMeasurer")
class ProgramMeasurer(Object):
    """
    Measurer that measures the time costs of tvm programs
    This class combines ProgramBuilder and ProgramRunner, and provides a simpler API.

    Parameters
    ----------
    builder : ProgramBuilder
        The ProgramBuilder to build programs
    runner : ProgramRunner
        The ProgramRunner to measure programs.
    callbacks : List[MeasureCallback]
        Callbacks to be called after each measurement batch
    verbose : int
        The Verbosity level: 0 for silent, 1 to output information during program
    max_continuous_error : Optional[int]
        The number of allowed maximum continuous error before stop the tuning
    """

    def __init__(self, builder, runner, callbacks, verbose, max_continuous_error=None):
        max_continuous_error = max_continuous_error or -1  # -1 means using the default value
        self.__init_handle_by_constructor__(
            _ffi_api.ProgramMeasurer, builder, runner, callbacks, verbose, max_continuous_error
        )


@tvm._ffi.register_object("auto_scheduler.LocalBuilder")
class LocalBuilder(ProgramBuilder):
    """LocalBuilder use local CPU cores to build programs in parallel.

    Parameters
    ----------
    timeout : int = 15
        The timeout limit (in second) for each build thread.
        This is used in a wrapper of the multiprocessing.Process.join().
    n_parallel : int = multiprocessing.cpu_count()
        Number of threads used to build in parallel.
    build_func : str = 'default'
        The name of registered build function.
    """

    def __init__(self, timeout=15, n_parallel=multiprocessing.cpu_count(), build_func="default"):
        self.__init_handle_by_constructor__(_ffi_api.LocalBuilder, timeout, n_parallel, build_func)


@tvm._ffi.register_object("auto_scheduler.LocalRunner")
class LocalRunner(ProgramRunner):
    """LocalRunner that uses local CPU/GPU to measures the time cost of programs.

    Parameters
    ----------
    timeout : int = 10
        The timeout limit (in second) for each run.
        This is used in a wrapper of the multiprocessing.Process.join().
    number : int = 3
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int = 1
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first "1" is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms : int = 100
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval : float = 0.0
        The cool down interval between two measurements.
    enable_cpu_cache_flush: bool = False
        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator closer to
        its actual latency during end-to-end inference.
        To make this option effective, the argument `number` should also be set to 1.
        This is only has effect on CPU task.
    """

    def __init__(
        self,
        timeout=10,
        number=3,
        repeat=1,
        min_repeat_ms=100,
        cooldown_interval=0.0,
        enable_cpu_cache_flush=False,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.LocalRunner,
            timeout,
            number,
            repeat,
            min_repeat_ms,
            cooldown_interval,
            enable_cpu_cache_flush,
        )


@tvm._ffi.register_object("auto_scheduler.RPCRunner")
class RPCRunner(ProgramRunner):
    """RPCRunner that uses RPC call to measures the time cost of programs on remote devices.
    Or sometime we may need to use RPC even in local running to insulate the thread environment.
    (e.g. running CUDA programs)

    Parameters
    ----------
    key : str
        The key of the device registered in the RPC tracker.
    host : str
        The host address of the RPC Tracker.
    port : int
        The port of RPC Tracker.
    priority : int = 1
        The priority of this run request, larger is more prior.
    n_parallel : int = 1
        The number of tasks run in parallel.
    timeout : int = 10
        The timeout limit (in second) for each run.
        This is used in a wrapper of the multiprocessing.Process.join().
    number : int = 3
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int = 1
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first "1" is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms : int = 100
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval : float = 0.0
        The cool down interval between two measurements.
    enable_cpu_cache_flush: bool = False
        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator closer to
        its actual latency during end-to-end inference.
        To make this option effective, the argument `number` should also be set to 1.
        This is only has effect on CPU task.
    """

    def __init__(
        self,
        key,
        host,
        port,
        priority=1,
        n_parallel=1,
        timeout=10,
        number=3,
        repeat=1,
        min_repeat_ms=100,
        cooldown_interval=0.0,
        enable_cpu_cache_flush=False,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.RPCRunner,
            key,
            host,
            port,
            priority,
            n_parallel,
            timeout,
            number,
            repeat,
            min_repeat_ms,
            cooldown_interval,
            enable_cpu_cache_flush,
        )

        if check_remote(key, host, port, priority, timeout):
            print("Get devices for measurement successfully!")
        else:
            raise RuntimeError(
                "Cannot get remote devices from the tracker. "
                "Please check the status of tracker by "
                "'python -m tvm.exec.query_rpc_tracker --port [THE PORT YOU USE]' "
                "and make sure you have free devices on the queue status."
            )


class LocalRPCMeasureContext:
    """A context wrapper for running RPCRunner locally.
    This will launch a local RPC Tracker and local RPC Server.

    Parameters
    ----------
    priority : int = 1
        The priority of this run request, larger is more prior.
    n_parallel : int = 1
        The number of tasks run in parallel.
    timeout : int = 10
        The timeout limit (in second) for each run.
        This is used in a wrapper of the multiprocessing.Process.join().
    number : int = 3
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int = 1
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first "1" is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms : int = 0
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval : float = 0.0
        The cool down interval between two measurements.
    enable_cpu_cache_flush: bool = False
        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator closer to
        its actual latency during end-to-end inference.
        To make this option effective, the argument `number` should also be set to 1.
        This is only has effect on CPU task.
    """

    def __init__(
        self,
        priority=1,
        n_parallel=1,
        timeout=10,
        number=3,
        repeat=1,
        min_repeat_ms=0,
        cooldown_interval=0.0,
        enable_cpu_cache_flush=False,
    ):
        # pylint: disable=import-outside-toplevel
        from tvm.rpc.tracker import Tracker
        from tvm.rpc.server import Server

        ctx = tvm.context("cuda", 0)
        if ctx.exist:
            cuda_arch = "sm_" + "".join(ctx.compute_version.split("."))
            set_cuda_target_arch(cuda_arch)
        host = "0.0.0.0"
        self.tracker = Tracker(host, port=9000, port_end=10000, silent=True)
        device_key = "$local$device$%d" % self.tracker.port
        self.server = Server(
            host,
            port=self.tracker.port,
            port_end=10000,
            key=device_key,
            use_popen=True,
            silent=True,
            tracker_addr=(self.tracker.host, self.tracker.port),
        )
        self.runner = RPCRunner(
            device_key,
            host,
            self.tracker.port,
            priority,
            n_parallel,
            timeout,
            number,
            repeat,
            min_repeat_ms,
            cooldown_interval,
            enable_cpu_cache_flush,
        )
        # Wait for the processes to start
        time.sleep(0.5)

    def __del__(self):
        # Close the tracker and server before exit
        self.tracker.terminate()
        self.server.terminate()


class MeasureErrorNo(object):
    """ Error type for MeasureResult. """

    NO_ERROR = 0  # No error
    INSTANTIATION_ERROR = 1  # Errors happen when apply transform steps from init state
    COMPILE_HOST = 2  # Errors happen when compiling code on host (e.g., tvm.build)
    COMPILE_DEVICE = 3  # Errors happen when compiling code on device
    # (e.g. OpenCL JIT on the device)
    RUNTIME_DEVICE = 4  # Errors happen when run program on device
    WRONG_ANSWER = 5  # Answer is wrong when compared to a reference output
    BUILD_TIMEOUT = 6  # Timeout during compilation
    RUN_TIMEOUT = 7  # Timeout during run
    UNKNOWN_ERROR = 8  # Unknown error


def _timed_func(inp_serialized, build_func, verbose):
    tic = time.time()
    inp = MeasureInput.deserialize(inp_serialized)
    task = inp.task

    error_no = MeasureErrorNo.NO_ERROR
    error_msg = None
    args = []

    try:
        sch, args = task.compute_dag.apply_steps_from_state(
            inp.state, layout_rewrite=ComputeDAG.RewriteForPreTransformed
        )
    # pylint: disable=broad-except
    except Exception:
        error_no = MeasureErrorNo.INSTANTIATION_ERROR
        error_msg = make_traceback_info()

    if error_no == 0:
        dirname = tempfile.mkdtemp()
        filename = os.path.join(dirname, "tmp_func." + build_func.output_format)

        try:
            with transform.PassContext():
                func = build_module.build(
                    sch, args, target=task.target, target_host=task.target_host
                )
            func.export_library(filename, build_func)
        # pylint: disable=broad-except
        except Exception:
            error_no = MeasureErrorNo.COMPILE_HOST
            error_msg = make_traceback_info()
    else:
        filename = ""

    if verbose >= 1:
        if error_no == MeasureErrorNo.NO_ERROR:
            print(".", end="")
        else:
            print(".E", end="")  # Build error

    return filename, args, error_no, error_msg, time.time() - tic


def local_build_worker(args):
    """
    Build function of LocalBuilder to be ran in the Builder thread pool.

    Parameters
    ----------
    args: Tuple[MeasureInput, str, int, int]
        inputs, build-func, time, verbose args passed to local_builder_build

    Returns
    -------
    res : BuildResult
        The build result of this Builder thread.
    """
    inp, build_func, timeout, verbose = args
    if build_func == "default":
        build_func = tar.tar
    elif build_func == "ndk":
        build_func = ndk.create_shared
    else:
        raise ValueError("Invalid build_func" + build_func)

    res = call_func_with_timeout(timeout, _timed_func, args=(inp, build_func, verbose))
    if isinstance(res, TimeoutError):
        if verbose >= 1:
            print(".T", end="")  # Build timeout
        res = None, [], MeasureErrorNo.BUILD_TIMEOUT, None, timeout
    elif isinstance(res, Exception):
        if verbose >= 1:
            print(".E", end="")  # Build error
        res = None, [], MeasureErrorNo.COMPILE_HOST, str(res), timeout

    return res


@tvm._ffi.register_func("auto_scheduler.local_builder.build")
def local_builder_build(inputs, timeout, n_parallel, build_func="default", verbose=1):
    """
    Build function of LocalBuilder to build the MeasureInputs to runnable modules.

    Parameters
    ----------
    inputs : List[MeasureInput]
        The MeasureInputs to be built.
    timeout : int
        The timeout limit (in second) for each build thread.
        This is used in a wrapper of the multiprocessing.Process.join().
    n_parallel : int
        Number of threads used to build in parallel.
    build_func : str = 'default'
        The name of build function to process the built module.
    verbose: int = 1
        Verbosity level. 0 for silent, 1 to output information during program building.

    Returns
    -------
    res : List[BuildResult]
        The build results of these MeasureInputs.
    """
    # This pool is not doing computationally intensive work, so we can use threads
    pool = multiprocessing.pool.ThreadPool(n_parallel)
    tuple_res = pool.map(
        local_build_worker,
        [
            (
                i.serialize(),
                build_func,
                timeout,
                verbose,
            )
            for i in inputs
        ],
    )
    pool.terminate()
    pool.join()
    del pool

    results = []
    for res in tuple_res:
        results.append(BuildResult(*res))

    return results


def _timed_eval_func(
    inp_serialized,
    build_res,
    number,
    repeat,
    min_repeat_ms,
    cooldown_interval,
    enable_cpu_cache_flush,
    verbose,
):
    inp = MeasureInput.deserialize(inp_serialized)
    tic = time.time()
    error_no = 0
    error_msg = None
    try:
        func = module.load_module(build_res.filename)
        ctx = ndarray.context(str(inp.task.target), 0)
        # Limitation:
        # We can not get PackFunction directly in the remote mode as it is wrapped
        # under the std::function. We could lift the restriction later once we fold
        # the PackedFunc as an object. Currently, we pass function name to work
        # around it.
        f_prepare = "cache_flush_cpu_non_first_arg" if enable_cpu_cache_flush else ""
        time_f = func.time_evaluator(
            func.entry_name,
            ctx,
            number=number,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            f_preproc=f_prepare,
        )
    # pylint: disable=broad-except
    except Exception:
        costs = (MAX_FLOAT,)
        error_no = MeasureErrorNo.COMPILE_DEVICE
        error_msg = make_traceback_info()

    if error_no == 0:
        try:
            args = [ndarray.empty(get_const_tuple(x.shape), x.dtype, ctx) for x in build_res.args]
            random_fill = tvm.get_global_func("tvm.contrib.random.random_fill", True)
            assert random_fill, "Please make sure USE_RANDOM is ON in the config.cmake"
            for arg in args:
                random_fill(arg)
            ctx.sync()
            costs = time_f(*args).results
        # pylint: disable=broad-except
        except Exception:
            costs = (MAX_FLOAT,)
            error_no = MeasureErrorNo.RUNTIME_DEVICE
            error_msg = make_traceback_info()

    shutil.rmtree(os.path.dirname(build_res.filename))
    toc = time.time()
    time.sleep(cooldown_interval)

    if verbose >= 1:
        if error_no == MeasureErrorNo.NO_ERROR:
            print("*", end="")
        else:
            print("*E", end="")  # Run error
    return costs, error_no, error_msg, toc - tic + build_res.time_cost, toc


@tvm._ffi.register_func("auto_scheduler.local_runner.run")
def local_run(
    inputs,
    build_results,
    timeout=10,
    number=3,
    repeat=1,
    min_repeat_ms=0,
    cooldown_interval=0,
    enable_cpu_cache_flush=False,
    verbose=1,
):
    """
    Run function of LocalRunner to test the performance of the input BuildResults.

    Parameters
    ----------
    inputs : List[MeasureInput]
        The MeasureInputs to be measured.
    build_results : List[BuildResult]
        The BuildResults to be measured.
    timeout : int = 10
        The timeout limit (in second) for each run.
        This is used in a wrapper of the multiprocessing.Process.join().
    number : int = 3
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int = 1
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first "1" is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms : int = 0
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval : float = 0.0
        The cool down interval between two measurements.
    enable_cpu_cache_flush: bool = False
        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator closer to
        its actual latency during end-to-end inference.
        To make this option effective, the argument `number` should also be set to 1.
        This is only has effect on CPU task.
    verbose: int = 1
        Verbosity level. 0 for silent, 1 to output information during program measuring.

    Returns
    -------
    res : List[MeasureResult]
        The measure results of these MeasureInputs.
    """

    measure_results = []
    assert len(inputs) == len(build_results), "Measure input size should be equal to build results"
    for inp, build_res in zip(inputs, build_results):
        if build_res.error_no != 0:
            res = (
                (MAX_FLOAT,),
                build_res.error_no,
                build_res.error_msg,
                build_res.time_cost,
                time.time(),
            )
        else:
            res = call_func_with_timeout(
                timeout,
                _timed_eval_func,
                args=(
                    inp.serialize(),
                    build_res,
                    number,
                    repeat,
                    min_repeat_ms,
                    cooldown_interval,
                    enable_cpu_cache_flush,
                    verbose,
                ),
            )
            if isinstance(res, TimeoutError):
                if verbose >= 1:
                    print("*T", end="")  # Run timeout
                res = (
                    (MAX_FLOAT,),
                    MeasureErrorNo.RUN_TIMEOUT,
                    None,
                    build_res.time_cost + timeout,
                    time.time(),
                )
            elif isinstance(res, Exception):
                if verbose >= 1:
                    print("*E", end="")  # Run error
                res = (
                    (MAX_FLOAT,),
                    MeasureErrorNo.RUNTIME_DEVICE,
                    str(res),
                    build_res.time_cost + timeout,
                    time.time(),
                )

        measure_results.append(MeasureResult(*res))

    if verbose >= 1:
        print("")

    return measure_results


def _timed_rpc_run(
    inp_serialized,
    build_res,
    key,
    host,
    port,
    priority,
    timeout,
    number,
    repeat,
    min_repeat_ms,
    cooldown_interval,
    enable_cpu_cache_flush,
    verbose,
):
    inp = MeasureInput.deserialize(inp_serialized)
    tic = time.time()
    error_no = 0
    error_msg = None
    try:
        # upload built module
        remote = request_remote(key, host, port, priority, timeout)
        remote.upload(build_res.filename)
        func = remote.load_module(os.path.split(build_res.filename)[1])
        ctx = remote.context(str(inp.task.target), 0)
        # Limitation:
        # We can not get PackFunction directly in the remote mode as it is wrapped
        # under the std::function. We could lift the restriction later once we fold
        # the PackedFunc as an object. Currently, we pass function name to work
        # around it.
        f_prepare = "cache_flush_cpu_non_first_arg" if enable_cpu_cache_flush else ""
        time_f = func.time_evaluator(
            func.entry_name,
            ctx,
            number=number,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            f_preproc=f_prepare,
        )
    # pylint: disable=broad-except
    except Exception:
        costs = (MAX_FLOAT,)
        error_no = MeasureErrorNo.COMPILE_DEVICE
        error_msg = make_traceback_info()

    if error_no == 0:
        try:
            args = [ndarray.empty(get_const_tuple(x.shape), x.dtype, ctx) for x in build_res.args]
            try:
                random_fill = remote.get_function("tvm.contrib.random.random_fill")
            except AttributeError:
                raise AttributeError(
                    "Please make sure USE_RANDOM is ON in the config.cmake " "on the remote devices"
                )
            for arg in args:
                random_fill(arg)
            ctx.sync()

            costs = time_f(*args).results
            # clean up remote files
            remote.remove(build_res.filename)
            remote.remove(os.path.splitext(build_res.filename)[0] + ".so")
            remote.remove("")
        # pylint: disable=broad-except
        except Exception:
            costs = (MAX_FLOAT,)
            error_no = MeasureErrorNo.RUNTIME_DEVICE
            error_msg = make_traceback_info()

    shutil.rmtree(os.path.dirname(build_res.filename))
    toc = time.time()

    time.sleep(cooldown_interval)
    if verbose >= 1:
        if error_no == MeasureErrorNo.NO_ERROR:
            print("*", end="")
        else:
            print("*E", end="")  # Run error

    return costs, error_no, error_msg, toc - tic + build_res.time_cost, toc


def _rpc_run_worker(args):
    """Function to be ran in the RPCRunner thread pool.

    Parameters
    ----------
    args : Tuple[MeasureInput, BuildResult, ...]
        Single input and build result plus the rest of the arguments to `rpc_runner_run`.

    Returns
    -------
    res : MeasureResult
        The measure result of this Runner thread.
    """
    _, build_res, _, _, _, _, timeout, _, _, _, _, _, verbose = args
    if build_res.error_no != MeasureErrorNo.NO_ERROR:
        return (
            (MAX_FLOAT,),
            build_res.error_no,
            build_res.error_msg,
            build_res.time_cost,
            time.time(),
        )

    res = call_func_with_timeout(timeout, _timed_rpc_run, args=args)
    if isinstance(res, TimeoutError):
        if verbose >= 1:
            print("*T", end="")  # Run timeout
        res = (
            (MAX_FLOAT,),
            MeasureErrorNo.RUN_TIMEOUT,
            None,
            build_res.time_cost + timeout,
            time.time(),
        )
    elif isinstance(res, Exception):
        if verbose >= 1:
            print("*E", end="")  # Run error
        res = (
            (MAX_FLOAT,),
            MeasureErrorNo.RUNTIME_DEVICE,
            str(res),
            build_res.time_cost + timeout,
            time.time(),
        )

    return res


@tvm._ffi.register_func("auto_scheduler.rpc_runner.run")
def rpc_runner_run(
    inputs,
    build_results,
    key,
    host,
    port,
    priority=1,
    n_parallel=1,
    timeout=10,
    number=3,
    repeat=1,
    min_repeat_ms=0,
    cooldown_interval=0.0,
    enable_cpu_cache_flush=False,
    verbose=1,
):
    """Run function of RPCRunner to test the performance of the input BuildResults.

    Parameters
    ----------
    inputs : List[MeasureInput]
        The MeasureInputs to be measured.
    build_results : List[BuildResult]
        The BuildResults to be measured.
    key : str
        The key of the device registered in the RPC tracker.
    host : str
        The host address of the RPC Tracker.
    port : int
        The port of RPC Tracker.
    priority : int = 1
        The priority of this run request, larger is more prior.
    n_parallel : int = 1
        The number of tasks run in parallel.
    timeout : int = 10
        The timeout limit (in second) for each run.
        This is used in a wrapper of the multiprocessing.Process.join().
    number : int = 3
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int = 1
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first "1" is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms : int = 0
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval : float = 0.0
        The cool down interval between two measurements.
    enable_cpu_cache_flush: bool = False
        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator closer to
        its actual latency during end-to-end inference.
        To make this option effective, the argument `number` should also be set to 1.
        This is only has effect on CPU task.
    verbose: int = 1
        Verbosity level. 0 for silent, 1 to output information during program measuring.

    Returns
    -------
    res : List[MeasureResult]
        The measure results of these MeasureInputs.
    """
    assert len(inputs) == len(build_results), "Measure input size should be equal to build results"
    # This pool is not doing computationally intensive work, so we can use threads
    pool = multiprocessing.pool.ThreadPool(n_parallel)
    tuple_res = pool.map(
        _rpc_run_worker,
        [
            (
                inp.serialize(),
                build_res,
                key,
                host,
                port,
                priority,
                timeout,
                number,
                repeat,
                min_repeat_ms,
                cooldown_interval,
                enable_cpu_cache_flush,
                verbose,
            )
            for inp, build_res in zip(inputs, build_results)
        ],
    )
    pool.terminate()
    pool.join()
    del pool

    results = []
    for res in tuple_res:
        results.append(MeasureResult(*res))

    if verbose >= 1:
        print("")

    return results

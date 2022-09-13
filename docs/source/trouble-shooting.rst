Trouble Shooting
================

.. rubric:: Overview

If you experience troubles with running PySAGES take a look at these known problems.
Here we explain issues that appear during the use of PySAGES and how you can troubleshoot them.

We hope these are helpful, but if you do not find a solution to your problem, consider reaching out to us by opening an issue on `GitHub <https://github.com/SSAGESLabs/PySAGES/issues>`_.

* My GPU simulation is significantly slower with PySAGES than just with the backend.
   * Make sure JAX can run on the GPU. Check `here <https://github.com/google/jax#pip-installation-gpu-cuda>`_ of how to install JAX with CUDA support. If your script output contains the line :code:`:WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)` JAX is not properly running on the GPU.

   * Make sure that for each simulation task (aka GPU or MPI rank) at least 4 CPUs are visible to the process. JAX requires this to run extra processes for JIT compiling. Otherwise, the GPU execution can be interrupted by JAX slowing computation. In a `slurm` environment the option :code:`--cpus-per-gpu=4` can be passed to `sbatch` scripts and `srun`. With OpenMPI you can specify :code:`mpiexec -np 1 --map-by slot:pe=4 python sages.py`.

   * Some collective variables can be expensive in the calculation in general or in gradient calculation, this can lead to slow simulations and or long JIT compilation times. For troubleshooting try running with `pysages.methods.Unbiased`, which does not use gradient calculations. Or try a different collective variable that is known to be simple, like `pysages.colvars.Component`.

* A PySAGES function cannot be launched and it errors with explaining that a function cannot be dispatched.
    * We are using `plum <https://github.com/wesselb/plum>`_ to dispatch functions with different arguments. Similar to C++ function overloading this happens by comparing the types (and number) of arguments to implemented functions. So make sure that your arguments are of the correct type. A common source of error is passing a numpy array, where a list is expected, or a float where an integer is expected. Plum does not try to cast your arguments into the correct types automatically.

     
* My HOOMD-blue 2.X simulation crashes with segfault at the beginning of the simulations.
    * Re-initializing the simulation context in the :code:`generate_context` ends with a segfault.::

        context = hoomd.context.SimulationContext()
        with context:
          hoomd.context.initialize()
          ...
  	  
      Should be replaced by::
	
	 hoomd.context.initialize()
	 context = hoomd.context.SimulationContext()
	 with context:
	   ...

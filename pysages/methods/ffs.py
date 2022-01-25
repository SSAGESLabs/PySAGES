# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from collections import namedtuple
from jax import scipy
from pysages.grids import build_indexer
from typing import Callable, Mapping
#import copy
#from copy import deepcopy, copy
from .core import SamplingMethod, generalize  # pylint: disable=relative-beyond-top-level
from pysages.utils import JaxArray
from pysages.utils import copy
from pysages.utils import ToCPU
import jax.numpy as np
from typing import NamedTuple
from pysages.backends import ContextWrapper
from pysages.collective_variables.core import build
from pysages.utils import identity

# ======= #
#   FFS   #
# ======= #
class FFSState(NamedTuple):
    bias: JaxArray
    xi: JaxArray

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)

#FFS
class FFS(SamplingMethod):
    snapshot_flags = {"positions", "indices"}
    Current_Window =0.0
    Prob_window= 0.0
    Window0 = []
    def build(self, snapshot, helpers):
        self.helpers=helpers
        return _ffs(self,snapshot,helpers)
    #run method for FFS
    def run(
            self, context_generator: Callable, timesteps: int,dt: float,win_i: float,win_l: float,Nw: int,sampling_steps_basin: int, Nmax_replicas: int,callback: Callable = None,
        context_args: Mapping = dict(), **kwargs
    ):
        """
        FFS implementation of running a single simulation/replica with the direct forward flux sampling method.

        Arguments
        ---------
        context_generator: Callable
            User defined function that sets up a simulation context with the backend.
            Must return an instance of `hoomd.context.SimulationContext` for HOOMD-blue
            and `simtk.openmm.Simulation` for OpenMM. The function gets `context_args`
            unpacked for additional user arguments.

        timesteps: int
            Number of timesteps the simulation is running.

        dt: float
            timestep of the simulation
        win_i: float
            initial window for the system
        win_l: float
            last window to be calculated in ffs
        Nw: int
            number of equally spaced windows
        sampling_steps_basin: int
            period for sampling configurations in the basin
        Nmax_replicas: int
            number of stored configuration for each window

        callback: Optional[Callable]
            Allows for user defined actions into the simulation workflow of the method.
            `kwargs` gets passed to the backend `run` function.
        """
        context = context_generator(**context_args)
        wrapped_context = ContextWrapper(context, self, callback)
        with wrapped_context:
            run=wrapped_context.run
            sampler=wrapped_context.sampler
            xi=sampler.state.xi.block_until_ready()
            windows=np.linspace(win_i, win_l, num=Nw)
            Check=Check_input(windows,xi)
            restore=self.helpers.restore
            helpers=self.helpers
            cv=self.cv
            reference_snapshot=copy(sampler.snapshot)
            if not Check:
                print('Bad initial configuration'+'\n')
                exit()
#Initial runs to sample from basin A
            Ini_snapshots=Basin_sampling(Nmax_replicas,sampling_steps_basin,windows,run,sampler,reference_snapshot,restore,helpers,cv)
#Calculate initial flow
            Phi_a,Ini_0=Initial_flow(Nmax_replicas,dt,windows,Ini_snapshots,run,sampler,restore,helpers,cv)
            write_to_file(Phi_a)
            INI=Ini_0
            Hist=np.zeros(len(windows))
            
            Hist=Hist.at[0].set(Phi_a)
#Calculate conditional probability for each window
            for k in range(1,len(windows)):
                if k==1:
                    old_snaps=INI
                prob, w1_snapshots=running_window(windows,k,old_snaps,run,sampler,restore,helpers,cv)
                write_to_file(prob)
                Hist=Hist.at[k].set(prob)
                old_snaps=increase_snaps(w1_snapshots,INI)
                print('size of snapshots= '+str(len(old_snaps))+'\n')
            K_t=np.prod(Hist)
            write_to_file('#Flux_Constant')
            write_to_file(K_t)
            return

def _ffs(method,snapshot,helpers):
    cv=method.cv
    dt=snapshot.dt
    natoms = np.size(snapshot.positions, 0)

    #initialize method
    def initialize():
        bias=np.zeros((natoms, 3))
        xi,_=cv(helpers.query(snapshot))
        return FFSState(bias,xi)
    def update(state, data):
        xi, _ = cv(data)
        bias = state.bias

        return FFSState(bias,xi)
    return snapshot, initialize, generalize(update,helpers)

#write the scalar values to the results file
def write_to_file(value):
    f=open('results.dat','a+')
    f.write(str(value)+'\n')
    f.close()
    return
#since snapshots are depleted each window, this function restore the list to the initial value
#this only works with stochastic integrators like BD or langevin, for other, velocity resampling is needed
def increase_snaps(windows,initial_w):
    if len(windows)>0:
        ratio=len(initial_w)//len(windows)
        windows=windows*ratio
    return windows
#collective variable as function
#check if the initial configuration is good
def Check_input(grid,xi):
    test=xi
    Win_A=grid[0]
    print(test)
    if test<Win_A:
        print("Good initial configuration\n")
        print(test)
        return True
    else:
        return False
#sampling basing configurations for initial flux calculations
def Basin_sampling(Max_num_snapshots,sampling_time,grid,run,sampler,reference_snapshot,restore,helpers,cv):
    count=0
    basin_snapshots=[]
    snap_restore=reference_snapshot
    win_A=grid[0]
    xi=sampler.state.xi.block_until_ready()
    print("Starting basin sampling\n")
    while len(basin_snapshots)<int(Max_num_snapshots):
        run(sampling_time)
        xi=sampler.state.xi.block_until_ready()
        if np.all(xi<win_A):
            snap=copy(sampler.snapshot)
            basin_snapshots.append(snap)
            print("Storing basing configuration with cv value:\n")
            print(xi)
#            print(snap.positions)
        else:
            restore(sampler.snapshot,snap_restore)
            xi,_=cv(helpers.query(sampler.snapshot))
            print("Restoring basing configuration since system leave basin with cv value:\n")
            print(xi)
    print('Finish sampling Basin with '+str(Max_num_snapshots)+' snapshots\n')
    return basin_snapshots
#initial flow selecting snapshots from list generated with basin_sampling
def Initial_flow(Num_window0,timestep,grid,initial_snapshots,run,sampler,restore,helpers,cv):
    success=0
    time_count=0.0
    window0_snaps=[]
    win_A=grid[0]
    for i in range(0,Num_window0):
        print('Initial stored configuration:'+str(i)+'\n')
        snapshot_r=initial_snapshots[i]
        restore(sampler.snapshot,snapshot_r)
        xi, _=cv(helpers.query(sampler.snapshot))
        print(xi)
#        print(initial_snapshots[0].positions)
#        print(snapshot_r.positions)
        has_reachedA=False
        while not has_reachedA:
            #this can be used not every timestep
            run(1)
            time_count+=timestep
            xi=sampler.state.xi.block_until_ready()
            if np.all(xi>=win_A):
                success+=1;
                has_reachedA=True;
                if len(window0_snaps)<=Num_window0:
                    snap=copy(sampler.snapshot)
                    window0_snaps.append(snap)
                break
    print('Finish Initial Flow with '+str(success)+' succeses over '+str(time_count)+' time\n')
    Phi_a=float(success)/(time_count)
    return Phi_a,window0_snaps    


def running_window(grid,step,old_snapshots,run,sampler,restore,helpers,cv):
    success=0
    new_snapshots=[]
    win_A=grid[0]
    win_value=grid[int(step)]
    has_conf_stored=False
    Running=True
    for i in range(0,len(old_snapshots)):
        snapshot_r=old_snapshots[i]
        restore(sampler.snapshot,snapshot_r)
        xi,_=cv(helpers.query(sampler.snapshot))
        print('Stored configuration:'+str(i)+' of window:'+str(step)+'\n')
        print(xi)
        #limit running time to avoid zombie trajectories, but it can be replaced
        Running=True
        while Running:
            run(1)
            xi=sampler.state.xi.block_until_ready()
            if np.all(xi<win_A):
                Running=False
            if np.all(xi>=win_value):
                snap=copy(sampler.snapshot)
                new_snapshots.append(snap)
                success+=1
                Running=False
                if not has_conf_stored:
#                    write_trajectory(step)
                    has_conf_stored=True
    if success==0:
        return print('Error in window'+' '+str(step)+'\n')
    if len(new_snapshots)>0:
        prob_local=float(success)/float(len(old_snapshots))
        print('Finish window '+str(step)+' with '+str(len(new_snapshots))+' snapshots\n')
        return prob_local,new_snapshots 
    

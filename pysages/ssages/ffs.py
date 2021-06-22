# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020: SSAGES Team (see LICENSE.md)

import jax.numpy as np

from abc import ABC, abstractmethod
from collections import namedtuple
from jax import jit, scipy
from jax.numpy import linalg
from plum import dispatch
from pysages.ssages.cvs import build
from pysages.nn.models import mlp
from pysages.nn.objectives import PartialRBObjective
from pysages.nn.optimizers import LevenbergMaquardtBayes
from pysages.nn.training import trainer

from .grids import get_index


# ================================== #
#        Forward Flux Sampling       #
# ================================== #
#When called, the function should start a cycle
#Correct_initial_structure=initialize(system,cv,grid,ids)
#if (Correct_initial_structure):
    #Calculate Phi_A
#    for i in range(0,N0_run):
#        system.run(1)
#        ξ, Jξ=cv(rs,indices(ids))
#        Crossed=CrossedtoA(ξ,grid)
#        if Crossed:
#            Store_snapshot(system,0)
#    Phi_A=len(stored_snapshots)/(system.dt*N0_run)
#Then, to calculate direct forward flux
#    for i in windows:
#        Success=0 
#        for j in range(0,len(stored_snapshots[i])):
#            system=init.read(stored_snapshots[i][j])
#            for k in range(0,N_run):
#                system.run(1)
#                ξ, Jξ=cv(rs,indices(ids))
#                Crossed=CrossingInterface(ξ,grid,i)
#                if Crossed>0:
#                    Store_snapshot(system,i+1)
#                    Success+=1
#                    break
#                elif Crossed<0:
#                    break
#        if len(stored_snapshots[i])==0:
#            print("Something bad happened")
#            break

        #Compute Conditional probability P(i+1|i)
#        hist[i]=Success/(len(stored_snapshots[i]))
    #calculate flux
#    Flux=Phi_A*np.prod(hist)





class FFSState(
    namedtuple(
        "State",
        ("Window", "hist", "Success","Phi_A", "L", "L_"),
    )
):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)

#FFS needs a 1D grid in Collective Variable to define the path
class FFS(GriddedOrderParameter):
    def __call__(self, snapshot, helpers):
        #Number of parallel replicas per window
        M = np.asarray(self.kwargs.get('M', 100))
        return ffs(snapshot, self.cv, self.grid, Nmax_replicas, N0_steps,N_steps)


def ffs(snapshot, cv, grid, N_replicas, N_steps):
    dt = snapshot.dt
    dims = grid.shape.size
    L_A = grid[0]
    N_windows=len(grid)
    L_B=grid[N_windows-1]
 #Function to determine if the initial snapshot is good   
    def initialize(snapshot,cv,grid,ids):
        ξ, Jξ=cv(rs,indices(ids))
    #if the initial snapshot is below the first window it is ok
        if ξ<grid[1]:
            return True
        else:
            return False
#Function that detects when has returned to origin the simulation
    def CrossedtoA(current_window,grid):
        window_A=grid[0]
        if current_window<= window_A:
            return True
        else:
            return False
#Function that detects when has crossed the interface
    def CrossingInterface(current_window,grid,i):
        interface=grid[i];
        limit=grid[0];
        #Crossed the interface and then the running should stop
        if current_window >= interface:
            return 1;
            #has returned to origin and the running should stop
        elif current_window <= limit:
            return -1;
            #continue
        else:
            return 0;
    def Initial_flow(Num_window0,dt,win_A,initial_snapshots,system):
        success=0
        time_count=0.0
        window0_snaps=[]
        for i in range(0,Num_window0):
            randin=random.randint(0,len(initial_snapshots))
            snapshot=initial_snapshots[randin]
            #how pysages takes snapshots from system?
            system.restore_snapshot(snapshot)
            has_reachedA=False
            while not has_reachedA:
                run(200)
                time_count+=timestep
                snap = system.take_snapshot()
                ξ, Jξ=cv(rs,indices(ids))
                if ξ>=win_A:
                    success+=1;
                    has_reachedA=True;
                    if len(window0_snaps)<=Num_window0:
                        window0_snaps.append(snap)
        Phi_a=float(success)/(time_count)
        return Phi_a,window0_snaps
    def running_window(Nw_steps,win_A,win_value,old_snapshots,system):
        success=0
        new_snapshots=[]
        for i in range(0,len(old_snapshots)):
            snapshot=old_snapshots[i]
            #sages snapshots can be used to restore the simulation?
            system.restore_snapshot(snapshot)
            for l in range(0,Nw_steps):
                run(1)
                snap=system.take_snapshot()
                ξ, Jξ=cv(rs,indices(ids))
                if ξ<win_A:
                    break
                if ξ>=win_value:
                    new_snapshots.append(snapshot)
                    success+=1
                    break
        if success==0:
            return print('Error in window'+' '+str(win_value)+'\n')
        if len(new_snapshots)>0:
            prob_local=float(success)/float(len(old_snapshots))
        return prob_local,new_snapshots






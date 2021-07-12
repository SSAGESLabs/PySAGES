# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020: SSAGES Team (see LICENSE.md)

from collections import namedtuple
from jax import scipy
from pysages.ssages.grids import build_indexer

from .core import GriddedSamplingMethod, generalize  # pylint: disable=relative-beyond-top-level

import jax.numpy as np

# ================================== #
#        Forward Flux Sampling       #
# ================================== #

class FFSState(
    namedtuple(
        "FFSState",
        ("Phi_A", "Previous_Window", "Current_Window", "Prob_window"),
    )
):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)

#FFS needs a 1D grid in Collective Variable to define the path
class FFS(GriddedSamplingMethod):
    def __call__(self, snapshot, helpers):
        #Number of parallel replicas per window
        Nmax_replicas= np.asarray(self.kwargs.get('M', 2000))
        N0_steps= np.asarray(self.kwargs.get('M', 2000))
        sampling_time= np.asarray(self.kwargs.get('M', 2000))
        system=0
        run=0
        return _ffs(snapshot, self.cv, self.grid, Nmax_replicas, N0_steps,sampling_time,system,run,helpers)


def _ffs(snapshot, cv, grid, Nmax_replicas, N0_steps,sampling_time,system,run,helpers):
    dt = snapshot.dt
    dims = grid.shape.size
    N_windows=len(grid)
    indices, momenta = helpers.indices, helpers.momenta
#initialize method
    def initialize():
        Phi_A = 0.0
        Previous_Window = 0.0
        Current_Window =0.0
        Prob_window= 0.0
        return FFSState(Phi_A,Previous_Window,Current_Window,Prob_window)
 #Function to determine if the initial snapshot is good (below first window)   
    def good_input(snapshot,cv,grid,rs,ids):
        ξ, Jξ=cv(rs, indices(ids))
        if ξ<grid[0]:
            return True
        else:
            return False
#Function to generate initial configurations from the Basin A
#requires system to integrate, how many snapshots you need, the window-A threshold and sampling time 
#The function returns a list of snapshots
    def Basin_sampling(system,number_basin_conf,grid,sampling_time,rs,ids):
#list for returned snapshots
        basin_snapshots=[]
#Initial snapshot used to restore the system in case of leaving the basin
        snap_restore=system.take_snapshot()
#Here it starts the sampling
        while len(basin_snapshots)<int(number_basin_conf):
            run(sampling_time)
            snap=system.take_snapshot()
            ξ, Jξ=cv(rs, indices(ids))
            if ξ<grid[0]:
                basin_snapshots.append(snap)
            else:
                system.restore_snapshot(snap_restore)
        return basin_snapshots
#Function that detects when it has returned to origin the simulation
    def CrossedtoA(current_window,grid):
        window_A=grid[0]
        if current_window>= window_A:
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
#Function to calculate initial flux from a ensemble of initial configurations in basin A
#and store the initial snapshots for the running windows
    def Initial_flow(cv,Num_window0,dt,grid,initial_snapshots,system,rs,ids):
        success=0
        time_count=0.0
        win_A=grid[0]
        window0_snaps=[]
        for i in range(0,Num_window0):
            randin=random.randint(0,len(initial_snapshots)-1)
            snapshot=initial_snapshots[randin]
            #how pysages takes snapshots from system?
            system.restore_snapshot(snapshot)
            has_reachedA=False
            while not has_reachedA:
                run(1)
                time_count+=timestep
                snap = system.take_snapshot()
                ξ, Jξ=cv(rs, indices(ids))
                Crossed=CrossedtoA(ξ,grid)
                if Crossed:
                    success+=1;
                    has_reachedA=True;
                    if len(window0_snaps)<=Num_window0:
                        window0_snaps.append(snap)
        Phi_a=float(success)/(time_count)
        return Phi_a,window0_snaps
#Function to run each window and calculate transition probabilities
    def running_window(cv,grid,step,old_snapshots,system,rs,ids):
        success=0
        win_A=grid[0]
        win_value=grid[step]
        new_snapshots=[]
        for i in range(0,len(old_snapshots)):
            snapshot=old_snapshots[i]
            #sages snapshots can be used to restore the simulation?
            system.restore_snapshot(snapshot)
            select=0
            while select==0:
                run(1)
                snap=system.take_snapshot()
                ξ, Jξ=cv(rs, indices(ids))
                select=CrossingInterface(ξ,grid,step)
                if select==-1:
                    break
                if select==1:
                    new_snapshots.append(snapshot)
                    success+=1
                    break
        if success==0:
            return print('Error in window'+' '+str(win_value)+'\n')
        if len(new_snapshots)>0:
            prob_local=float(success)/float(len(old_snapshots))
        return prob_local,new_snapshots
    def update(snapshot, cv, grid, Nmax_replicas, N0_steps,sampling_time,system,run):
        Check=good_input(system,grid,cv)
        if not Check:
            print('Bad initial configuration'+'\n')
            exit()

#Initial runs to sample from basin A
        Ini_snapshots=Basin_sampling(system,limit,sampling_time,grid,rs,ids)
#Calculate initial flow
        Phi_a,Ini_0=Initial_flow(limit,dt,grid,Ini_snapshots,system,rs,ids)
        write_to_file(Phi_a)
        write_trajectory(0)
        INI=Ini_0*100
        Hist=np.zeros(len(windows))
        Hist[0]=Phi_a
#Calculate conditional probability for each window
        for k in range(1,len(windows)):
            if k==1:
                old_snaps=INI
            prob, w1_snapshots=running_window(Nw_steps,grid,k,old_snaps,system,rs,ids)
            write_to_file(prob)
            Hist[k]=prob
            old_snaps=increase_snaps(w1_snapshots,INI)
            print('size of snapshots= '+str(len(old_snaps))+'\n')
        K_t=np.prod(Hist)
        write_to_file('#Flux_Constant')
        write_to_file(K_t)
        return FFSState(Phi_A,Previous_Window,Current_Window,Prob_window)
    return snapshot, initialize, generalize(update)







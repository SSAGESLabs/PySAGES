# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Artificial neural network based collective variables
"""

from jax import numpy as np, vmap
from pysages.colvars.core import CollectiveVariable
from pysages.colvars import coordinates
from pysages.colvars.utils import rational_switching_function
import numpy as onp
        
class PIV(CollectiveVariable):
    """
    Permutation Invariant Vector (PIV) of a given system of points in space.

    PIV collective variable is generated using the user-defined set of points in space.
    Reference to the points is provided as a list of lists containing indices of 
    unique atom types. PIV is constructed by first computing the unique pairwise distances 
    between the indices across all the combinations of lists except those specifically
    excluded by user. The generated pairwise distances in each cross-pair 
    list are referred as a block and are wrapped by a switching function. 
    The values in each block are then sorted. Finally, PIV is 
    returned as a single vector by catenating all blocks.
    
    
    Parameters
    ----------
    indices: ``list`` of ``lists``
            List of lists containing indices of unique atom types.
    exclude_indices: ``list``
            List of indices pairs referring index of lists to exclude 
            in PIV calculation.
    switching_params: ``list``
            List of switching function parameters. It contains R_0, m and n values.
            
    Returns
    -------
    piv: JaxArray
        Permutation Invariant Vector (PIV)
    """
    
    def __init__(self, indices, position_pairs, solute_list, solvent_oxygen_list,
                 hydrogen_dict, switching_params,  neighbor_list):
        super().__init__(indices, group_length=None)
        self.position_pairs = position_pairs
        self.solute_list = solute_list
        self.solvent_oxygen_list = solvent_oxygen_list
        self.hydrogen_dict = hydrogen_dict
        self.switching_params = switching_params
        self.neighbor_list = neighbor_list['neighbor_list']
        
        print("sol list")
        print(solute_list)
        print(solvent_oxygen_list)
        print("end")
        
    @property
    def function(self):
        """
        Function generator

        Returns
        -------
        Function that generates PIV from a simulation snapshot.
        Look at `pysages.colvars.ann.piv` for details.
        """
        return lambda *positions, neighbor_list=self.neighbor_list, params=self: piv(positions, neighbor_list, params)


def piv(positions, neighbor_list, params):
    """
    Implementation of permutation invariant vector as described in
    [Section 4, Handb. Mater. Model. 597-619 (2020)](https://doi.org/10.1007/978-3-319-44677-6_51).

    Parameters
    ----------
    positions: A ``list`` of ``lists`` containing positions of all atoms and of each atomtype.
    
    params: A ``object`` containing all the helper parameters. This includes
        indices combination to exclude from PIV calculation, switching function
        parameters.

    Returns
    -------
    piv : DeviceArray
        Permutation Invariant Vector (PIV).
    """
    
    all_atom_positions = np.array(positions)
    
    neighbor_list = neighbor_list.update(all_atom_positions)
    position_pairs = onp.array(params.position_pairs)
    solute_list = params.solute_list
    solvent_oxygen_list = params.solvent_oxygen_list
    
    print("solvent_oxygen_list")
    print(solvent_oxygen_list)
    print("end solvent_oxygen_list")
    
    hydrogen_dict = params.hydrogen_dict
    
    i_pos = all_atom_positions[position_pairs[:,1]]
    j_pos = all_atom_positions[position_pairs[:,3]]
    
    piv_solute_blocks = vmap(get_piv_block, in_axes=(0, 0, None))(i_pos, j_pos, params.switching_params)
    
    # sort based ij combination
    
    
    print(piv_solute_blocks)
    
    if solvent_oxygen_list:
                
        nsolute_types = len(solute_list)
        # for each solute keep oxygen atoms that are their neighbors
        solvent_i_j = []
        
        solute_list_indices = list(np.arange(nsolute_types))
        
        
        for i in range(nsolute_types):
            _solvent_i_j = []
            
            for i_p in solute_list[i]:
            
                for j in neighbor_list.idx[i_p]:
                
                    # check if j is the id of oxygen atom
                    if j in solvent_oxygen_list:
                        _solvent_i_j.append([i, i_p, j])
                        _solvent_i_j.append([i, i_p, hydrogen_dict[int(j)][0]])
                        _solvent_i_j.append([i, i_p, hydrogen_dict[int(j)][1]])
                            
            solvent_i_j.append(_solvent_i_j)
            
        solvent_i_j = onp.array(solvent_i_j)
        i_pos = all_atom_positions[solvent_i_j[:,1]]
        j_pos = all_atom_positions[solvent_i_j[:,2]]
                
        piv_solute_solvent_blocks = vmap(get_piv_block, in_axes=(0, 0, None))(i_pos, j_pos, params.switching_params)
        
        print("sol")
        print(piv_solute_solvent_blocks)
        print("end")
        
    print("\n")
    
    
def get_solute_atoms(solute_list, solute_list_index):
    
    return solute_list[solute_list_index]

def get_neighbors_solute_atom(neigbor_list_solute, solute_atom_id):
    
    return neigbor_list_solute[solute_atom_id]
    

def get_piv_block(i_pos, j_pos, switching_params):
    
    r_0 = switching_params['r_0']
    d_0 = switching_params['d_0']
    n = switching_params['n']
    m = switching_params['m']
    
    r = coordinates.distance(i_pos, j_pos)
    s_r = rational_switching_function(r, r_0, d_0, n, m)
    
    return s_r
    
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Implementation of Permutation Invariant Vector (PIV) as described in 
[Handb. Mater. Model.: Theor. and Model., 597-619 (2020)]
(https://doi.org/10.1007/978-3-319-44677-6_51) by Fabio Petrucci
"""

from jax import numpy as np, vmap

from pysages.colvars.core import CollectiveVariable
from pysages.colvars import coordinates
from pysages.colvars.utils import rational_switching_function

import numpy as onp
        
class PIV(CollectiveVariable):
    """
    Permutation Invariant Vector (PIV) of a given system of points 
    in space as described in Section 4 of 
    [Handb. Mater. Model.: Theor. and Model., 597-619 (2020)]
    (https://doi.org/10.1007/978-3-319-44677-6_51).

    PIV collective variable is generated using the user-defined 
    set of points in space. These points are typically the coordinates 
    of solute and solvent. Single or multiple solutes in a given 
    solvent can be defined. For determining solvent atoms in a solvation 
    shell around solute, 
    [JAX MD](https://jax-md.readthedocs.io/en/main/jax_md.partition.html) 
    neighborlist library is utilized. This requires the user
    to define the indices of all the atoms in the system and a JAX MD 
    neighbor list function
    (see alanine dipeptide example in examples/openmm/metad/pivcv).
    
    solute-solute blocks of the PIV are determined by using the indices 
    of the solute pairs defined by the user. To sort solute-solute blocks 
    and solute-solvent blocks, user need to provide the indices of all 
    solute atoms and oxygen atoms in two separate lists. The indices 
    of the hydrogen atoms bonded to the oxygen atoms are determined 
    by using the oxygen-hydrogen dictionary provided by the user.
    
    The switching function parameters parameters for each block 
    should be provided as a list of dictionaries.
    
    Example definition:
        cvs = [PIV( all_atoms, position_pairs, solute_list, oxygen_list,
        hydrogen_dict, [{'r_0': 0.4, 'd_0': 2.3, 'n': 3, 'm': 6}, ...],
        {'neighbor_list': gen_neighbor_list()})]
    
    Parameters
    ----------
    allatoms: list
            List of indices of all atoms required for updating neighbor list.
    position_pairs: Array
            Array containing indices of solute-solute pairs for the solute-solute
            block of PIV.
    solute_list: Array
            Indices of all solute atoms
    oxygen_list: Array
            Indices of all oxygen atoms
    hydrogen_dict: dict
            Dictionary mapping each oxygen in water with their hydrogen atoms.
    switching_params: dict
            Dictionary containing switching function parameters.
    neighbor_list: Object
            JAX MD neighbor list function to update the neighbor list.
            
    Returns
    -------
    piv: JaxArray
        Permutation Invariant Vector (PIV)
    """
    
    def __init__(self, indices, position_pairs, solute_list, solvent_oxygen_list,
                 hydrogen_dict, switching_params, neighbor_list):
        super().__init__(indices, group_length=None)
        self.position_pairs = position_pairs
        self.solute_list = solute_list
        self.solvent_oxygen_list = solvent_oxygen_list
        self.hydrogen_dict = hydrogen_dict
        self.switching_params = switching_params
        self.neighbor_list = neighbor_list['neighbor_list']
        
    @property
    def function(self):
        """
        Function generator

        Returns
        -------
        Function that generates PIV from a simulation snapshot.
        Look at `pysages.colvars.ann.piv` for details.
        """
        #return lambda *positions, neighbor_list=self.neighbor_list, params=self: piv(positions, neighbor_list, params)
        return lambda *positions: piv(positions, self.neighbor_list, self)
        

def piv(positions, neighbor_list, params):
    """
    Implementation of permutation invariant vector as described in
    [Section 4, Handb. Mater. Model. 597-619 (2020)]
    (https://doi.org/10.1007/978-3-319-44677-6_51) by Fabio Petrucci.

    Parameters
    ----------
    positions: Array
            Contains positions of all atoms in the system.
    neighbor_list: Object
            Points to function to update neighbor list.
    params: Object
        Links to all the helper parameters. This includes
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
    piv_solute_block_index = vmap(cantor_pair, in_axes=(0,0))(position_pairs[:,0], position_pairs[:,2])
    
    idx_solute_sort = np.argsort(piv_solute_block_index)
    piv_solute_blocks = piv_solute_blocks[idx_solute_sort]

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
        piv_solute_solvent_block_index = solvent_i_j[:,0]
        
        idx_solvent_sort = np.argsort(piv_solute_block_index)
        piv_solute_solvent_blocks = piv_solute_solvent_blocks[idx_solvent_sort]

        piv_blocks = np.concatenate( (piv_solute_blocks, piv_solute_solvent_blocks), axis=0)

    else:
        
        piv_blocks = piv_solute_blocks
        
    return piv_blocks
    
    
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
    
    
def cantor_pair(index1, index2):
    """
    Generates an uniuqe integer using two integers via Cantor pair function.
    This unique integer can be mapped back to the two integers, if needed.
    """
    
    pi = index1 + index2
    pi = pi * (pi + 1)
    pi *= 0.5
    pi += index2
    
    return np.int32(pi)
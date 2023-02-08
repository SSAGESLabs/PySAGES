# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

import time

from jax import lax
from jax import numpy as np
from jax import random, vmap
from jax.numpy import linalg
from jax_md.partition import space
from jaxopt import GradientDescent as minimize

from pysages.colvars.core import CollectiveVariable
from pysages.utils import gaussian, quaternion_from_euler, quaternion_matrix


def rotate_pattern_with_quaternions(rot_q, pattern):
    return np.transpose(np.dot(quaternion_matrix(rot_q)[:3, :3], np.transpose(pattern)))


def func_to_optimise(Q, modified_pattern, local_pattern):
    return np.linalg.norm(rotate_pattern_with_quaternions(Q, modified_pattern) - local_pattern)


# Main class implementing the GeM CV
class Pattern:
    """
    For determining nearest neighbors,
    [JAX MD](https://jax-md.readthedocs.io/en/main/jax_md.partition.html)
    neighborlist library is utilized. This requires the user
    to define the indices of all the atoms in the system and a JAX MD
    neighbor list callable for updating the state.
    """

    def __init__(
        self,
        simulation_box,
        fractional_coords,
        reference,
        neighborlist,
        characteristic_distance,
        centre_j_id,
        standard_deviation,
        mesh_size,
    ):

        self.characteristic_distance = characteristic_distance
        self.reference = reference
        self.neighborlist = neighborlist
        self.simulation_box = simulation_box
        self.centre_j_id = centre_j_id
        # This is added to handle neighborlists with fractional coordinates
        # (needed for NPT simulations)
        if fractional_coords:
            self.positions = self.neighborlist.reference_position * np.diag(self.simulation_box)
        else:
            self.positions = self.neighborlist.reference_position
        self.centre_j_coords = self.positions[self.centre_j_id]
        self.standard_deviation = standard_deviation
        self.mesh_size = mesh_size

    def comp_pair_distance_squared(self, pos1):
        displacement_fn, shift_fn = space.periodic(np.diag(self.simulation_box))
        mic_vector = displacement_fn(self.centre_j_coords, pos1)
        mic_norm = linalg.norm(mic_vector)
        return mic_norm, mic_vector

    def _generate_neighborhood(self):
        self._neighborhood = []

        positions_of_all_nbrs = self.positions[self.neighborlist.idx[self.centre_j_id]]
        distances, mic_vectors = vmap(self.comp_pair_distance_squared)(positions_of_all_nbrs)
        # remove the same atom from the neighborhood
        distances = np.where(distances != 0.0, distances, 1e5)
        # remove the number of atoms from the list of indices
        distances = np.where(
            self.neighborlist.idx[self.centre_j_id] != len(self.neighborlist.idx), distances, 1e5
        )

        ids_of_neighbors = np.argsort(distances)[: len(self.reference)]

        coordinates = mic_vectors[ids_of_neighbors] + self.centre_j_coords
        # Step 1: Translate to origin;
        coordinates = coordinates.at[:].set(coordinates - np.mean(coordinates, axis=0))
        for vec_id, mic_vector in enumerate(mic_vectors[ids_of_neighbors]):
            neighbor = {
                "id": ids_of_neighbors[vec_id],
                "coordinates": coordinates[vec_id],
                "mic_vector": mic_vector,
                "pos_wrt_j": self.centre_j_coords - mic_vector,
                "distance": distances[ids_of_neighbors[vec_id]],
            }
            self._neighborhood.append(neighbor)

        self._neighbor_coords = np.array([n["coordinates"] for n in self._neighborhood])
        self._orig_neighbor_coords = positions_of_all_nbrs[ids_of_neighbors]

    def compute_score(self, optim_reference):
        r = self._neighbor_coords - optim_reference
        return np.prod(gaussian(1, self.standard_deviation, r))

    def rotate_reference(self, random_euler_point):
        # Perform rotation of the reference pattern;
        # Using Euler angles in radians construct a quaternion base;
        # Convert the quaternion to a 3x3 rotation matrix.
        rot_q = quaternion_from_euler(*random_euler_point)
        return rotate_pattern_with_quaternions(rot_q, self.reference)

    def resort(self, pattern_to_resort, key):
        # This subroutine shuffles randomly the input local pattern
        # and resorts the reference indices in order to "minimise"
        # the distance of the corresponding sites

        random_indices = random.permutation(
            key, np.arange(len(self._neighborhood)), axis=0, independent=False
        )
        random_neighbor_coords = self._neighbor_coords[random_indices]

        def find_closest(carry, neighbor_coords):
            ref_positions = carry
            distances = [np.linalg.norm(vector - neighbor_coords) for vector in ref_positions]
            min_index = np.argmin(np.array(distances))
            positions = ref_positions.at[min_index].set(np.array([-1e10, -1e10, -1e10]))
            new_ref_positions = ref_positions[min_index]
            return positions, new_ref_positions

        _, closest_reference = lax.scan(find_closest, pattern_to_resort, random_neighbor_coords)
        # Reorder the reference to match the positions of the neighbors
        reshuffled_reference = np.zeros_like(closest_reference)
        reshuffled_reference = reshuffled_reference.at[random_indices].set(closest_reference)
        return reshuffled_reference, random_indices

    def check_settled_status(self, modified_reference):
        def mark_close_sites(_, reference_pos):
            def return_close(_, n):
                return lax.cond(
                    np.linalg.norm(n - reference_pos) < self.characteristic_distance / 2.0,
                    lambda x: (None, x + 1),
                    lambda x: (None, x),
                    0,
                )

            _, close_sites_per_reference = lax.scan(return_close, None, self._neighbor_coords)
            return None, close_sites_per_reference

        _, close_sites = lax.scan(mark_close_sites, None, modified_reference)
        _, indices = lax.scan(
            lambda _, sites: (
                None,
                lax.cond(np.sum(sites) == 1, lambda s: s, lambda s: np.zeros_like(s), sites),
            ),
            None,
            close_sites,
        )
        # Return the locations of settled nighbours in the neighborhood;
        # Settlled site should have a unique neighbor
        settled_neighbor_indices = np.where(np.sum(indices, axis=0) >= 1, 1, 0)
        return settled_neighbor_indices

    def driver_match(self, number_of_rotations, number_of_opt_steps, num):

        self._generate_neighborhood()

        """Step2: Scale the reference so that the spread matches
        with the current local pattern"""
        local_distance = 0.0
        reference_distance = 0.0
        for n_index, neighbor in enumerate(self._neighborhood):
            local_distance += np.dot(neighbor["coordinates"], neighbor["coordinates"])
            reference_distance += np.dot(self.reference[n_index], self.reference[n_index])

        self.reference *= np.sqrt(local_distance / reference_distance)

        """Step3: mesh-loop -> Define angles in reduced Euler domain,
        and for each rotate, resort and score the pattern

        The implementation below follows the article Martelli et al. 2018


        (a) Randomly with uniform probability pick a point in the Euler domain,
        (b) Rotate the reference
        (c) Resort the local pattern and assign the closest reference sites,
        (d) Perform the optimisation step (conjugate gradient),
        and (e) store the score with (f) the final settled status"""

        def get_all_scores(newkey, euler_point):
            # b. Rotate the reference pattern
            rotated_reference = self.rotate_reference(euler_point)
            # c. Resort; shuffle the local pattern
            # and assign ids to the closest reference sites
            newkey, newsubkey = random.split(random.PRNGKey(newkey))
            reshuffled_reference, random_indices = self.resort(rotated_reference, newsubkey)
            # d. Find the best rotation that aligns the settled sites
            # in both patterns;
            # Here, ‘optimal’ or ‘best’ is in terms of least squares errors
            solver = minimize(fun=func_to_optimise, maxiter=number_of_opt_steps)
            # We are fixing the initial guess for the quaternions;
            # different starting conditions are obtained by working
            # with a rotated reference (this can be changed)
            optim = solver.run(
                init_params=np.array([0.1, 0.0, 0.0, 0.1]),
                modified_pattern=reshuffled_reference,
                local_pattern=self._neighbor_coords,
            )
            optimal_reference = rotate_pattern_with_quaternions(optim.params, reshuffled_reference)
            # e. Compute and store the score
            score = self.compute_score(optimal_reference)
            result = dict(
                score=score,
                rotated_pattern=rotated_reference,
                random_indices=random_indices,
                reshuffled_pattern=reshuffled_reference,
                pattern=optimal_reference,
                quaternions=optim.params,
            )
            return result

        # a. Randomly pick a point in the Euler domain

        key, subkey = random.split(random.PRNGKey(num))
        mesh_size = self.mesh_size
        grid_dimension = np.pi / mesh_size
        euler_angles = np.arange(
            0, 0.125 * np.pi + (mesh_size / 2 + 1) * grid_dimension, grid_dimension
        )
        random_points = random.randint(
            subkey, (number_of_rotations, 3), minval=0.0, maxval=mesh_size
        )
        # Excute find_max_score for each angle
        # and store the result with the highest score

        scoring_results = vmap(get_all_scores)(
            num + np.arange(number_of_rotations), euler_angles[random_points]
        )
        optimal_case = np.argmax(scoring_results["score"])

        # f. Check how many settled sites there are
        settled_neighbor_ids = self.check_settled_status(scoring_results["pattern"][optimal_case])

        # Storing all the data is only for validation and analysis;
        # For FFS, only score will be returned, i.e., optimal_result['score'];
        # This then can be removed
        optimal_result = dict(
            score=scoring_results["score"][optimal_case],
            rotated_pattern=scoring_results["rotated_pattern"][optimal_case],
            random_indices=scoring_results["random_indices"][optimal_case],
            reshuffled_pattern=scoring_results["reshuffled_pattern"][optimal_case],
            pattern=scoring_results["pattern"][optimal_case],
            quaternions=scoring_results["quaternions"][optimal_case],
            settled=settled_neighbor_ids,
            centre_atom=self.centre_j_coords,
            neighborhood=self._neighbor_coords,
            neighborhood_orig=self._orig_neighbor_coords,
        )
        return optimal_result


def calculate_lom(all_positions: np.array, neighborlist, simulation_box, params):

    if params.fractional_coords:
        update_neighborlist = neighborlist.update(np.divide(all_positions, np.diag(simulation_box)))
    else:
        update_neighborlist = neighborlist.update(all_positions)

    """Step1: Move the reference and
    local patterns so that their centers coincide with the origin"""

    reference_positions = params.reference_positions.at[:].set(
        params.reference_positions - np.mean(params.reference_positions, axis=0)
    )

    # Calculate scores
    seed = np.int64(time.process_time() * 1e5)
    optimal_results = vmap(
        lambda i: Pattern(
            params.box,
            params.fractional_coords,
            reference_positions,
            update_neighborlist,
            params.characteristic_distance,
            i,
            params.standard_deviation,
            params.mesh_size,
        ).driver_match(
            params.number_of_rotations,
            params.number_of_opt_it,
            seed + i * params.number_of_rotations,
        )
    )(np.arange(len(all_positions), dtype=np.int64))
    average_score = np.sum(optimal_results["score"]) / len(all_positions)
    return average_score


class GeM(CollectiveVariable):
    """
    This CV implements a Geometry Matching (GeM) Local Order Metric (LOM).
    The algorithm enabling the measurement of order in the neighborhood of
    an atomic or a molecular site is described in
    [Martelli2018](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.064105).

    Given a pattern, the algorithm is returning an average score (from 0 to 1),
    denoting how closely the atomic neighbors resemble the reference.

    For determining nearest neighbors,
    [JAX MD](https://jax-md.readthedocs.io/en/main/jax_md.partition.html)
    neighborlist library is utilized. This requires the user
    to define the indices of all the atoms in the system and a JAX MD
    neighbor list callable for updating the state.

    Matching a neighborhood to the pattern is an optimization process.
    Based on the number of initial rotations of the reference structure
    and opt. iterations, we aim to find a rotation matrix Q
    that minimizes norm(a-Q*b), where a is the neighborhood
    and b denotes the reference. This is defined in `func_to_optimise`.
    Optimization is performed using [JAXopt](https://github.com/google/jaxopt).

    Parts of the code related to JAX compatible 3d transformations
    (e.g., quaternion_matrix) are taken from
    [jax_transformations3d](https://github.com/cpgoodri/jax_transformations3d).

    Parameters
    ----------
    indices: list
            List of indices of all atoms required for updating neighbor list.
    reference_positions: JaxArray
    box: JaxArray
            Definition of the simulation box.
    number_of_rotations: integer
            Number of initial rotated structures for the optimization study.
    number_of_opt_it: iteger
            Number of iterations for gradient descent.
    standard_deviation: float
            Parameter that controls the spread of the Gaussian function.
    mesh_size: integer
            Defines the size of the angular grid from which we draw
            random Euler angles.
    nbrs: callable
            JAX MD neighbor list function to update the neighbor list.
    fractional_coords: Bool
            Set to True if NPT simulation is considered and the box size
            changes; use periodic_general for constructing the neighborlist.
    Returns
    -------
    calculate_lom: float
        Average score defining the degree of overlap
        with the reference structure. It's a measure of the global order.
    """

    def __init__(
        self,
        indices,
        reference_positions,
        box,
        number_of_rotations,
        number_of_opt_it,
        standard_deviation,
        mesh_size,
        nbrs,
        fractional_coords,
    ):
        super().__init__(indices, group_length=None)

        self.reference_positions = np.asarray(reference_positions)
        self.box = np.asarray(box)
        self.number_of_rotations = number_of_rotations
        self.number_of_opt_it = number_of_opt_it
        self.standard_deviation = standard_deviation
        self.characteristic_distance = standard_deviation * 2
        self.mesh_size = mesh_size
        self.nbrs = nbrs
        self.fractional_coords = fractional_coords

    @property
    def function(self):
        return lambda rs: calculate_lom(rs, self.nbrs, self.box, self)

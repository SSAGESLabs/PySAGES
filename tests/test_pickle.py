import importlib
import inspect
import pathlib
import tempfile

import dill as pickle
import numpy as np
import test_simulations.abf as abf_example

import pysages
import pysages.colvars
import pysages.methods

pi = np.pi


def build_neighbor_list(box_size, positions, r_cutoff, capacity_multiplier):
    """Helper function to generate a jax-md neighbor list"""
    jmd = importlib.import_module("jax_md")

    displacement_fn, _ = jmd.space.periodic(box_size)
    neighbor_list_fn = jmd.partition.neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff,
        capacity_multiplier=capacity_multiplier,
        format=jmd.partition.NeighborListFormat.Dense,
    )
    neighbors = neighbor_list_fn.allocate(positions)

    return neighbors


METHODS_ARGS = {
    "HarmonicBias": {"cvs": [pysages.colvars.Component([0], 0)], "kspring": 15.0, "center": 0.7},
    "Unbiased": {"cvs": [pysages.colvars.Component([0], 0)]},
    "SamplingMethod": {
        "cvs": [pysages.colvars.Component([0], 0)],
    },
    "ABF": {
        "cvs": [pysages.colvars.Component([0], 0)],
        "grid": pysages.Grid(lower=(-pi), upper=(pi), shape=(32), periodic=True),
        "restraints": pysages.CVRestraints(lower=(-pi), upper=(pi), kl=1, ku=1),
    },
    "ANN": {
        "cvs": [pysages.colvars.Component([0], 0)],
        "grid": pysages.Grid(lower=(-pi), upper=(pi), shape=(32), periodic=True),
        "topology": (15, 23),
        "kT": 1.0,
    },
    "CFF": {
        "cvs": [pysages.colvars.Component([0], 0)],
        "grid": pysages.Grid(lower=(-pi), upper=(pi), shape=(32), periodic=True),
        "topology": (14,),
        "kT": 1.0,
    },
    "FFS": {
        "cvs": [pysages.colvars.Component([0], 0)],
    },
    "FUNN": {
        "cvs": [pysages.colvars.Component([0], 0)],
        "grid": pysages.Grid(lower=(-pi), upper=(pi), shape=(32), periodic=True),
        "topology": (15, 23),
    },
    "UmbrellaIntegration": {
        "cvs": [pysages.colvars.Component([0], 0)],
        "kspring": 15.0,
        "center": [0.0, 0.7],
        "hist_periods": 10,
    },
    "Metadynamics": {
        "cvs": [pysages.colvars.Component([0], 0)],
        "height": 1.0,
        "sigma": 5.0,
        "stride": 7.0,
        "ngaussians": 5,
        "deltaT": 0.1,
        "grid": pysages.Grid(lower=(-pi), upper=(pi), shape=(32), periodic=True),
        "kB": 614.0,
    },
    "SpectralABF": {
        "cvs": [pysages.colvars.Component([0], 0), pysages.colvars.Component([0], 1)],
        "grid": pysages.Grid(lower=(1, 1), upper=(5, 5), shape=(32, 32)),
    },
    "HistogramLogger": {
        "period": 1,
        "offset": 1,
    },
    "MetaDLogger": {
        "hills_file": "tmp.txt",
        "log_period": 158,
    },
    "ReplicasConfiguration": {},
    "SerialExecutor": {},
    "CVRestraints": {"lower": (-pi, -pi), "upper": (pi, pi), "kl": (0.0, 1.0), "ku": (1.0, 0.0)},
    "Bias": {"cvs": [pysages.colvars.Component([0], 0)], "center": 0.7},
    "SplineString": {
        "cvs": [pysages.colvars.Component([0], 0)],
        "ksprings": 15.0,
        "centers": [0.0, 0.7],
        "hist_periods": 10,
    },
}


def test_pickle_methods():
    # Iterate all methods of pysages
    for key, pyclass in inspect.getmembers(pysages.methods, inspect.isclass):
        pickle.dumps(pyclass)
        try:
            obj = pyclass(**METHODS_ARGS[key])
        # Filter abstract classes
        except TypeError:
            pass
        else:
            try:
                pickle.dumps(obj)
            except Exception as error:
                print(key)
                raise error


COLVAR_ARGS = {
    "Angle": {"indices": [0, 1, 2]},
    "DihedralAngle": {"indices": [0, 1, 2, 3]},
    "RadiusOfGyration": {"indices": [0, 1, 2, 3]},
    "PrincipalMoment": {"indices": [0, 1, 2, 3], "axis": 0},
    "Asphericity": {"indices": [0, 1, 2, 3]},
    "Acylindricity": {"indices": [0, 1, 2, 3]},
    "ShapeAnisotropy": {"indices": [0, 1, 2, 3]},
    "Component": {"indices": [0, 1, 2, 3], "axis": 0},
    "Distance": {"indices": [0, 1]},
    "Displacement": {"indices": [[0], [1]]},
    "GeM": {
        "indices": np.arange(20),
        "reference_positions": np.array(
            [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
        ),
        "box": 2 * np.eye(3),
        "number_of_rotations": 20,
        "number_of_opt_it": 10,
        "standard_deviation": 0.125,
        "mesh_size": 30,
        "nbrs": None,
        # Disable build_neighbor_list until jax_md stabilizes
        # "nbrs": build_neighbor_list(
        #     2.0, positions=np.random.randn(20, 3), r_cutoff=1.5, capacity_multiplier=1.0
        # ),
        "fractional_coords": True,
    },
}


def test_pickle_colvars():
    # Iterate all collective variables of PySAGES
    for key, pyclass in inspect.getmembers(pysages.colvars, inspect.isclass):
        pickle.dumps(pyclass)
        try:
            obj = pyclass(**COLVAR_ARGS[key])
        # Filter abstract classes
        except TypeError:
            pass
        else:
            try:
                pickle.dumps(obj)
            except Exception as error:
                print(key)
                raise error


def test_pickle_results():
    test_result = abf_example.run_simulation(10, write_output=False)

    with tempfile.NamedTemporaryFile() as tmp_pickle:
        pickle.dump(test_result, tmp_pickle)
        tmp_pickle.flush()

        tmp_result = pickle.load(open(tmp_pickle.name, "rb"))

        assert np.all(test_result.states[0].xi == tmp_result.states[0].xi).item()
        assert np.all(test_result.states[0].bias == tmp_result.states[0].bias).item()
        assert np.all(test_result.states[0].hist == tmp_result.states[0].hist).item()
        assert np.all(test_result.states[0].Fsum == tmp_result.states[0].Fsum).item()

    tmp_file = pathlib.Path(".tmp_test_pickle")
    pysages.save(test_result, tmp_file)
    tmp_result = pysages.load(tmp_file.name)

    assert np.all(test_result.states[0].xi == tmp_result.states[0].xi).item()
    assert np.all(test_result.states[0].bias == tmp_result.states[0].bias).item()
    assert np.all(test_result.states[0].hist == tmp_result.states[0].hist).item()
    assert np.all(test_result.states[0].Fsum == tmp_result.states[0].Fsum).item()

    tmp_file.unlink()

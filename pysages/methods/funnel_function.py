# funnel functions
from functools import partial

import jax.numpy as np
from jax import grad, jit

from pysages.colvars.funnels import perp_projection_mobile, projection_mobile


def y_function(x, Z_0, Zcc, R):
    m = (R - Z_0) / Zcc
    return m * x + Z_0


def cone(x, x_p, Zcc, Z_0, R, k):
    F = x_p - y_function(x, Z_0, Zcc, R)
    return np.where(F < 0.0, 0.0, 0.5 * k * F * F)


def cylinder(x, R, k):
    F = x - R
    return np.where(F < 0.0, 0.0, 0.5 * k * F * F)


def borderU(x, k, cv_max):
    B = x - cv_max  # upper limit
    return np.where(B < 0.0, 0.0, 0.5 * k * B * B)


def borderL(x, k, cv_min):
    B = x - cv_min  # lower limit
    return np.where(B > 0.0, 0.0, 0.5 * k * B * B)


def funnel(xi_par, xi_perp, Zcc, Z_0, R, k, k_cv, cv_min, cv_max):
    return np.where(
        xi_par < Zcc,
        cone(xi_par, xi_perp, Zcc, Z_0, R, k)
        + borderL(xi_par, k_cv, cv_min)
        + borderU(xi_par, k_cv, cv_max),
        cylinder(xi_perp, R, k) + borderU(xi_par, k_cv, cv_max) + borderL(xi_par, k_cv, cv_max),
    )


def intermediate_funnel(
    pos,
    ids,
    indexes,
    references,
    weights_ligand,
    weights_protein,
    A,
    B,
    Zcc,
    Z_0,
    R,
    k,
    k_cv,
    cv_min,
    cv_max,
    box,
):
    indices_ligand = np.array(indexes[0])
    indices_protein = np.array(indexes[1])
    indices_anchor = np.array(indexes[2])
    pos_anchor = pos[ids[indices_anchor]]
    pos_protein = pos[ids[indices_protein]]
    pos_ligand = pos[ids[indices_ligand]]
    xi_par = projection_mobile(
        pos_ligand,
        pos_protein,
        pos_anchor,
        np.asarray(references),
        weights_ligand,
        weights_protein,
        np.asarray(A),
        np.asarray(B),
        np.asarray(box),
    )
    xi_perp = perp_projection_mobile(
        pos_ligand,
        pos_protein,
        pos_anchor,
        np.asarray(references),
        weights_ligand,
        weights_protein,
        np.asarray(A),
        np.asarray(B),
        np.asarray(box),
    )
    return funnel(xi_par, xi_perp, Zcc, Z_0, R, k, k_cv, cv_min, cv_max)


def log_funnel(
    pos,
    ids,
    indexes,
    references,
    weights_ligand,
    weights_protein,
    A,
    B,
    Zcc,
    Z_0,
    R,
    k,
    k_cv,
    cv_min,
    cv_max,
    box,
):
    indices_ligand = np.array(indexes[0])
    indices_protein = np.array(indexes[1])
    indices_anchor = np.array(indexes[2])
    pos_anchor = pos[ids[indices_anchor]]
    pos_protein = pos[ids[indices_protein]]
    pos_ligand = pos[ids[indices_ligand]]
    xi_perp = perp_projection_mobile(
        pos_ligand,
        pos_protein,
        pos_anchor,
        np.asarray(references),
        weights_ligand,
        weights_protein,
        np.asarray(A),
        np.asarray(B),
        np.asarray(box),
    )
    return xi_perp


def external_funnel(
    data,
    indexes,
    references,
    weights_ligand,
    weights_protein,
    A,
    B,
    Zcc,
    Z_0,
    R,
    k,
    k_cv,
    cv_min,
    cv_max,
    box,
):
    pos = data.positions[:, :3]
    ids = data.indices
    bias = grad(intermediate_funnel)(
        pos,
        ids,
        indexes,
        references,
        weights_ligand,
        weights_protein,
        A,
        B,
        Zcc,
        Z_0,
        R,
        k,
        k_cv,
        cv_min,
        cv_max,
        box,
    )
    proj = log_funnel(
        pos,
        ids,
        indexes,
        references,
        weights_ligand,
        weights_protein,
        A,
        B,
        Zcc,
        Z_0,
        R,
        k,
        k_cv,
        cv_min,
        cv_max,
        box,
    )
    return bias, proj


def get_funnel_force(
    indices_sys,
    ref_positions,
    A,
    B,
    Zcc,
    Z_0,
    R_cyl,
    k_cone,
    k_cv,
    cv_min,
    cv_max,
    cv_buffer,
    box,
    w_ligand=None,
    w_protein=None,
):
    funnel_force = partial(
        external_funnel,
        indexes=indices_sys,
        references=ref_positions,
        weights_ligand=w_ligand,
        weights_protein=w_protein,
        A=A,
        B=B,
        Zcc=Zcc,
        Z_0=Z_0,
        R=R_cyl,
        k=k_cone,
        k_cv=k_cv,
        cv_min=cv_min - cv_buffer,
        cv_max=cv_max + cv_buffer,
        box=box,
    )
    return jit(funnel_force)

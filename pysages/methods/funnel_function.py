# funnel functions
from functools import partial

import jax.numpy as np
from jax import grad, jit
from jax.numpy import linalg

from pysages.colvars.funnels import center, kabsch, periodic


def y_function(x, Z_0, Zcc, R):
    m = (R - Z_0) / Zcc
    return m * x + Z_0


def cone(x, eje, Zcc, Z_0, R, k):
    x_coord = np.dot(x, eje)
    proj = x_coord * eje
    x_perp = x - proj
    F = linalg.norm(x_perp) - y_function(x_coord, Z_0, Zcc, R)
    return np.where(F < 0.0, 0.0, 0.5 * k * F * F)


def cylinder(x, eje, R, k):
    x_perp = x - np.dot(x, eje) * eje
    F = linalg.norm(x_perp) - R
    return np.where(F < 0.0, 0.0, 0.5 * k * F * F)


def borderU(x, eje, k, cv_max):
    # x = lig_com-A
    proj_restr = np.dot(x, eje)
    B = proj_restr - cv_max  # upper limit
    return np.where(B < 0.0, 0.0, 0.5 * k * B * B)


def borderL(x, eje, k, cv_min):
    # x = lig_com-A
    proj = np.dot(x, eje)
    B = proj - cv_min  # lower limit
    return np.where(B > 0.0, 0.0, 0.5 * k * B * B)


def rotation_lig(pos_lig, pos_ref, references, weights, com_ref):
    com_prot = center(pos_ref, weights)
    lig_rot = np.dot(pos_lig - com_prot, kabsch(pos_ref, references, weights))
    return lig_rot + com_ref


def funnel(x, A, B, Zcc, Z_0, R, k, k_cv, cv_min, cv_max):
    A_r = A
    B_r = B
    norm_eje = linalg.norm(B_r - A_r)
    eje = (B_r - A_r) / norm_eje
    #    Z_pos = Zcc * eje
    x_fit = x - A_r
    proj = np.dot(x_fit, eje)
    return np.where(
        proj < Zcc,
        cone(x_fit, eje, Zcc, Z_0, R, k) + borderL(x_fit, eje, k_cv, cv_min) + borderU(x_fit, eje, k_cv, cv_max),
        cylinder(x_fit, eje, R, k) + borderU(x_fit, eje, k_cv, cv_max) + borderL(x_fit, eje, k_cv, cv_max),
    )


def proj_funnel(x, A, B, Zcc, Z_0, R, k, k_cv, cv_min, cv_max):
    A_r = A
    B_r = B
    norm_eje = linalg.norm(B_r - A_r)
    eje = (B_r - A_r) / norm_eje
    #    Z_pos = Zcc * eje
    x_fit = x - A_r
    proj = np.dot(x_fit, eje)
    perp = x_fit - proj * eje
    return linalg.norm(perp)


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
    ligand_distances = periodic(pos[ids[indices_ligand]] - pos_anchor, np.asarray(box))
    new_lig_pos = pos_anchor + ligand_distances
    pos_ligand = center(new_lig_pos, weights_ligand)
    pos_ref = center(np.asarray(references), weights_protein)
    lig_rot = rotation_lig(
        pos_ligand, pos_protein, np.asarray(references), weights_protein, pos_ref
    )
    return funnel(lig_rot, np.asarray(A), np.asarray(B), Zcc, Z_0, R, k, k_cv, cv_min, cv_max)


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
    ligand_distances = periodic(pos[ids[indices_ligand]] - pos_anchor, np.asarray(box))
    new_lig_pos = pos_anchor + ligand_distances
    pos_ligand = center(new_lig_pos, weights_ligand)
    pos_ref = center(np.asarray(references), weights_protein)
    lig_rot = rotation_lig(
        pos_ligand, pos_protein, np.asarray(references), weights_protein, pos_ref
    )
    return proj_funnel(lig_rot, np.asarray(A), np.asarray(B), Zcc, Z_0, R, k, k_cv, cv_min, cv_max)


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

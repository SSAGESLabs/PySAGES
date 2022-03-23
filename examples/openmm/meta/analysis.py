import numpy as np


def sum_1Dgaussians(x, hills):

    fes = np.zeros(len(x))

    for i in range(len(x)):

        for j in range(len(hills.cv.values)):

            delta_xi = x[i] - hills.cv.values[j]

            fes[i] += hills.height.values[j] * np.exp(
                -(delta_xi * delta_xi) / (2 * hills.sigma.values[j] * hills.sigma.values[j])
            )

    return fes


def sum_gaussians(x, cv_values, sigma, height, ncvs, angle):

    fes = np.zeros(np.shape(x)[0])

    for i in range(np.shape(x)[0]):

        for j in range(np.shape(cv_values)[0]):

            local_height = height[j]

            exp_product = 1
            for k in range(ncvs):

                delta_xi = x[i][k] - cv_values[j][k]

                if angle[k]:

                    if delta_xi > np.pi:
                        delta_xi -= 2.0 * np.pi

                    elif delta_xi < -np.pi:
                        delta_xi += 2.0 * np.pi

                local_sigma = sigma[j][k]

                arg = (delta_xi * delta_xi) / (2 * local_sigma * local_sigma)

                exp_product = exp_product * np.exp(-arg)

            fes[i] += local_height * exp_product

    return fes

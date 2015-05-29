#!/usr/bin/env python


from __future__ import print_function
import random
import math
import sys

import matplotlib.pyplot as plt
import numpy as np


def simulate(pop_size_1, pop_size_2, time_change, mu_rate, n_samples):
    """
    Generates distribution of the number of mutations with the given
    population history and parameters
    """
    samples = []
    num_snd = 0
    for _ in xrange(n_samples):
        snd_time = random.expovariate(float(1) / pop_size_2)
        if snd_time < time_change:
            samples.append(2 * mu_rate * snd_time)
            num_snd += 1
        else:
            fst_time = random.expovariate(float(1) / pop_size_1)
            samples.append(2 * mu_rate * (fst_time + time_change))

    fst_perc = 100 * (len(samples) - num_snd) / len(samples)
    snd_perc = 100 * num_snd / len(samples)
    #print("Fst epoch: {0}%, snd epoch: {1}%".format(fst_perc, snd_perc))
    return samples


def em(sample, shift):
    """
    An implementation of EM, that separates a mixture of two exponentials in form
    pi_1 * lam_1 * exp(-t * lam_1) + pi_2 * lam_2 * exp(-(t + s) * lam_2).
    Shift s is given as a parameter
    """
    N_EXP = 2
    lambd = np.ones(N_EXP)
    density = np.ones(N_EXP) / N_EXP
    shift = np.array([0, shift, float("inf")])
    memb_prob = np.zeros((N_EXP, len(sample)))
    for exp in xrange(N_EXP):
        for n in xrange(len(sample)):
            memb_prob[exp][n] = (1 if shift[exp + 1] >= sample[n] > shift[exp]
                                 else 0)

    for _ in xrange(100):
        #M-step
        for exp in xrange(N_EXP):
            avg = 0
            dens = 0
            for n in xrange(len(sample)):
                avg += memb_prob[exp][n] * (sample[n] - shift[exp])
                dens += memb_prob[exp][n]
            lambd[exp] = dens / avg
            density[exp] = dens / len(sample)

        #E-step
        for n in xrange(len(sample)):
            prob = np.zeros(N_EXP)
            for exp in xrange(N_EXP):
                t = sample[n] - shift[exp]
                prob[exp] = (density[exp] * math.exp(-t * lambd[exp]) * lambd[exp]
                             if t > 0 else 0)
            sum_prob = sum(prob)
            for exp in xrange(N_EXP):
                memb_prob[exp][n] = prob[exp] / sum_prob

    return lambd, density


def get_shift(sample):
    """
    Estimates a shift of the second exponential from zero
    """
    hist = np.histogram(sample, 200, density=True)
    xx, yy = hist[1][:-1], hist[0]

    deriv = [0] * len(xx)
    for i in xrange(1, len(deriv) - 1):
        deriv[i] = yy[i + 1] - yy[i - 1]
    shift = xx[np.argmax(deriv)]
    return shift


def make_sample(pop_mult, time_change):
    """
    Simulates a whole-genome sample
    """
    POP_SIZE = 10000
    MU = 1.1 * 10 ** (-8)
    GENOME_SIZE = 3 * 10 ** 9
    REGION_SIZE = 1 * 10 ** 6
    NUM_REG = GENOME_SIZE / REGION_SIZE
    EFF_MU = MU * REGION_SIZE
    return simulate(POP_SIZE, POP_SIZE * pop_mult,
                    time_change, EFF_MU, NUM_REG)


def make_table():
    """
    Performs the simulation for different values of lambda
    and time, and outputs a table with both visible
    and hidden (inferred) parameters
    """
    print("D1\tD2\tL1\tL2\tN1\tN2\tShift\tTime")
    for mult in np.linspace(2, 12, 20):
        for time in np.linspace(5000, 50000, 20):
            sample = make_sample(mult, time)
            shift = get_shift(sample)
            lambd, density = em(sample, shift)

            print("{0:.2f}\t{1:.2f}\t{2:.5f}\t{3:.5f}\t{4:d}\t{5:d}\t{6:d}\t{7:d}"
                  .format(density[0], density[1],
                          lambd[0], lambd[1], 10000, int(10000 * mult),
                          int(shift), int(time)))

            sys.stdout.flush()


def plot_sample():
    """
    Performs one simulation and makes a fancy plot
    """
    sample = make_sample(4, 30000)

    shift = get_shift(sample)
    lambd, density = em(sample, shift)

    print("Shift: {0}".format(shift))
    print("Estimated lambdas: {0}".format(lambd))
    print("Estimated density: {0}".format(density))

    def fitted_func(t):
        fst = math.exp(-t * lambd[0]) * lambd[0]
        snd = math.exp(-(t - shift) * lambd[1]) * lambd[1] if t > shift else 0
        return fst * density[0] + snd * density[1]

    hist = np.histogram(sample, 200, density=True)
    xx, yy = hist[1][:-1], hist[0]

    fit = list(map(fitted_func, xx))

    plt.plot(xx, yy)
    plt.plot(xx, fit, linewidth=2)
    plt.plot([shift, shift], [0, max(yy)], linewidth=2, color="red")
    plt.show()


def main():
    plot_sample()
    make_table()
    return


if __name__ == "__main__":
    main()

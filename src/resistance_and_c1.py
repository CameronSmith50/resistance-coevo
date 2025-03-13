"""
resistance_and_c1.py
Code to generate a three panel figure which plots measures of host and parasite fitness as a function of resistance.

author: Cameron Smith
"""

#%% Import packages and other functions

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm
import pickle as pkl
from scipy.optimize import fsolve

from modeling import *
from evolutionary_simulation import *

#%% Function which will calculate the optimum alpha_P

def findAlphaP(alphaP, y, c1, ode_parameters):
    '''
    Function which will output the derivative of the fitness gradient to be
    passed to a root finder.
    '''

    # Update ode_parameters
    ode_parameters['alpha_P'] = alphaP[0]
    ode_parameters['resistance'] = y
    ode_parameters['c1'] = c1

    # Steady state
    Ht, Dt, Pt, Bt = run_ode_solver(ode_parameters, make_system(ode_parameters))
    SS = approximate_steady_state(Ht, Dt, Pt, Bt)

    # Calculate the fitness gradient
    fitGrad = build_fitness_gradient(ode_parameters, SS)
    return(fitGrad(alphaP))


#%% Parameter values and storage

if __name__ == '__main__':

    # ODE parameters
    ode_parameters = set_default_ode_parameters()

    # Compute ancestral values
    ancestral_virulence = compute_ancestral_virulence(ode_parameters)
    ancestral_steady_state = compute_ancestral_steady_state(ode_parameters)
    ancestral_population_size = sum(ancestral_steady_state)

    # Resistance
    MIN_RESISTANCE = 0
    MAX_RESISTANCE = 1
    nRes = 201
    resistance_vector = np.linspace(MIN_RESISTANCE, MAX_RESISTANCE, nRes)

    # Virulence
    MIN_VIRULENCE = 0.1
    MAX_VIRULENCE = 1.0
    nVir = 201
    virulence_vector = np.linspace(MIN_VIRULENCE, MAX_VIRULENCE, nVir)
    dVir = virulence_vector[1] - virulence_vector[0]
    initial_virulence_index = round((ancestral_virulence - MIN_VIRULENCE)/dVir)

    # Cost values
    C1_LOW = 0.2
    C1_MODERATE = 0.5
    C1_HIGH = 0.8
    c1_vector = np.array([C1_LOW, C1_MODERATE, C1_HIGH])
    nc1 = len(c1_vector)

    # Storage matrix
    # Will be a nRes x nc1 x 6 matrix. 
    # Each axis 0 corresponds to a resistance value.
    # Axis 1 are cost values
    # Axis 2 is evolved virulence and flag for ES or not, H, D, P and B as functions of resistance and cost
    storageMat = np.zeros((nRes, nc1, 6))

    # Evolutionary timesteps
    NUMBER_OF_TIMESTEPS = 2001
    evolutionary_timesteps = np.arange(NUMBER_OF_TIMESTEPS)

    # Extinction threshold
    extinction_threshold = 1e-5

    # Save directory
    saveDir = './data/paper_data/figure2/'
    tempStr = saveDir.split('/')
    tempStr[1] = 'results'
    tempStr[2] = 'paper_figs'
    saveFigDir = '/'.join(tempStr) + '/'
    if not os.path.isdir(saveDir):
        os.mkdir(saveDir)
        runData = True
    else:
        runInput = input('Data already exists. Should you re-run the code? [y/n]: ')
        runData = runInput == 'y'

    if not os.path.isdir(saveFigDir):
        os.mkdir(saveFigDir)

    if runData:

        # Loop through the resistance and c1 values
        for yInd, y in tqdm(enumerate(resistance_vector), total=len(resistance_vector)):

            # Update the resistance values
            ode_parameters['resistance'] = y

            for c1Ind, c1 in enumerate(c1_vector):

                # Update the ODE parameters
                ode_parameters['c1'] = c1

                optAlphaP = fsolve(lambda alphaP: findAlphaP(alphaP, y, c1, ode_parameters), x0=ancestral_virulence)
                storageMat[yInd, c1Ind, 0] = optAlphaP[0]
                # storageMat[yInd, c1Ind, 1] = optAlphaP[0]

                # Now calculate the ecological dynamics at this value
                ode_parameters['alpha_P'] = storageMat[yInd, c1Ind, 0]
                Ht, Dt, Pt, Bt = run_ode_solver(ode_parameters,
                            make_system(ode_parameters))
                SS = approximate_steady_state(Ht, Dt, Pt, Bt)
                
                # Check if this is evolutionary stable
                secondDer = build_invasion_fitness_2nd_derivative_wrt_mutant(
                    ode_parameters, SS)(optAlphaP[0])
                storageMat[yInd, c1Ind, 1] = int(secondDer < 0)
                
                # Store
                storageMat[yInd, c1Ind, 2:] = [Ht[-1], Dt[-1], Pt[-1], Bt[-1]]

        # Save the data
        pdict = {
            'resistance_vector': resistance_vector,
            'ancestral_virulence': ancestral_virulence,
            'ancestral_steady_state': ancestral_steady_state,
            'c1_vector': c1_vector,
            'storageMat': storageMat
            }
        file = open(saveDir + 'data.pkl', 'wb')
        pkl.dump(pdict, file)
        file.close()

    # If the data wasn't generated, load the data
    if not runData:
        file = open(saveDir + 'data.pkl', 'rb')
        pdict = pkl.load(file)
        file.close()
        
    # Now we set up the plots
    figWidth = 25
    figHeight = 18
    fig = plt.figure(figsize=(figWidth, figHeight))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    # Fontsizes
    fSize = 30
    mpl.rcParams['font.size'] = fSize

    # Line colours
    cols = ['k', 'r', 'b']
    styles = ['--', ':', '-']

    ax1.plot(pdict['resistance_vector'], np.ones(len(pdict['resistance_vector'])), '-.', c='#777777')
    ax2.plot(pdict['resistance_vector'], np.ones(len(pdict['resistance_vector'])), '-.', c='#777777')
    ax3.plot(pdict['resistance_vector'], np.ones(len(pdict['resistance_vector'])), '-.', c='#777777')

    # Create the figures
    for c1Ind, c1 in enumerate(pdict['c1_vector']):

        # Create a mask where the parasite dies out
        mask = np.sum(pdict['storageMat'][:, c1Ind, 4:], axis=1) > extinction_threshold
        finalInd = sum(mask) - 1

        # Figure for relative virulence
        ax1.plot(pdict['resistance_vector'][mask], pdict['storageMat'][mask, c1Ind, 0]/pdict['ancestral_virulence'], lw=4, c=cols[c1Ind], ls=styles[c1Ind], label = rf'$c_1$ = {c1}')
        # ax1.plot(pdict['resistance_vector'][mask], pdict['storageMat'][mask, c1Ind, 1]/pdict['ancestral_virulence'], lw=4, c=cols[c1Ind], ls=styles[c1Ind])
        if finalInd+1 != len(pdict['resistance_vector']):
            ax1.plot(pdict['resistance_vector'][finalInd], pdict['storageMat'][finalInd, c1Ind, 0]/pdict['ancestral_virulence'], f'{cols[c1Ind]}.', markersize=20)
            # ax1.plot(pdict['resistance_vector'][finalInd], pdict['storageMat'][finalInd, c1Ind, 1]/pdict['ancestral_virulence'], f'{cols[c1Ind]}.', markersize=20)

        # Figure for change in host population size
        ax2.plot(pdict['resistance_vector'][mask], np.sum(pdict['storageMat'][mask, c1Ind, 2:], axis=1)/np.sum(pdict['ancestral_steady_state']), lw=4, c=cols[c1Ind], ls=styles[c1Ind])
        if finalInd+1 != len(pdict['resistance_vector']):
            ax2.plot(pdict['resistance_vector'][finalInd], np.sum(pdict['storageMat'][finalInd, c1Ind, 2:])/np.sum(pdict['ancestral_steady_state']), f'{cols[c1Ind]}.', markersize=20)

        # Figure for change in parasitised hosts
        ax3.plot(pdict['resistance_vector'][mask], np.sum(pdict['storageMat'][mask, c1Ind, 4:], axis=1)/pdict['ancestral_steady_state'][1], lw=4, c=cols[c1Ind], ls=styles[c1Ind])
        if finalInd+1 != len(pdict['resistance_vector']):
            ax3.plot(pdict['resistance_vector'][finalInd], np.sum(pdict['storageMat'][finalInd, c1Ind, 4:])/pdict['ancestral_steady_state'][1], f'{cols[c1Ind]}.', markersize=20)

    # Options
    ax1.set_xlim([pdict['resistance_vector'][0], pdict['resistance_vector'][-1]])
    ax1.set_xlabel('')
    ax1.set_ylabel('Relative\nevolved virulence', fontsize = fSize)
    ax1.set_xticks([pdict['resistance_vector'][0], 0.5*np.sum(pdict['resistance_vector'][[0,-1]]), pdict['resistance_vector'][-1]])
    ax1.set_xticklabels(['', '', ''])
    ax1.set_yticks([1, 1.5])
    ax1.set_yticklabels(['1', '1.5'], fontsize = fSize)
    ax1.legend(fontsize = fSize)

    ax2.set_xlim([pdict['resistance_vector'][0], pdict['resistance_vector'][-1]])
    ax2.set_xlabel('')
    ax2.set_ylabel('Relative host\npopulation size', fontsize = fSize)
    ax2.set_xticks([pdict['resistance_vector'][0], 0.5*np.sum(pdict['resistance_vector'][[0,-1]]), pdict['resistance_vector'][-1]])
    ax2.set_xticklabels(['', '', ''])
    ax2.set_yticks([1, 1.2])
    ax2.set_yticklabels(['1', '1.2'], fontsize = fSize)

    ax3.set_xlim([pdict['resistance_vector'][0], pdict['resistance_vector'][-1]])
    ax3.set_xlabel(r'Resistance ($y$)', fontsize = fSize)
    ax3.set_ylabel('Relative\ndisease prevalence', fontsize = fSize)
    ax3.set_xticks([pdict['resistance_vector'][0], 0.5*np.sum(pdict['resistance_vector'][[0,-1]]), pdict['resistance_vector'][-1]])
    ax3.set_xticklabels([int(pdict['resistance_vector'][0]), 0.5*np.sum(pdict['resistance_vector'][[0,-1]]), int(pdict['resistance_vector'][-1])], fontsize = fSize)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['0', '1'], fontsize = fSize)

    # Add the label text
    ax1.text(-0.1, 1.58, '(A)')
    ax2.text(-0.1, 1.25, '(B)')
    ax3.text(-0.1, 1.10, '(C)')

    # Save the figure
    plt.savefig(saveFigDir + 'Fig2.png', bbox_inches = 'tight')
    plt.savefig(saveFigDir + 'Fig2.pdf', bbox_inches = 'tight')

    # plt.show
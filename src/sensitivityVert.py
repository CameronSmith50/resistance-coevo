# A script to contain everything required to answer if the increase in host
# outcomes is caused by the mechanism, strength of protection or the density of
# primed individuals. The ultimate goal is a series of scatter plots of both
# priming rate and protection level against either of the two host outcome
# measures.

#%%

from modeling import *
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from resistance_and_c1 import findAlphaP
import sys
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import linregress
from initialProt import initProt

#%% Function to generate the appropriate data
def generateData(n = 25, save = False):
    '''
    Code to generate data for randomly parameterised data.
    '''

    # Generate the default parmeter set
    pars = set_default_ode_parameters()

    # Set the boundaries for various parameters by setting them to be
    # (100-perc)% and (100+perc)% of the default parameter
    perc = 50
    bRange = [pars['b']*(1-perc/100), pars['b']*(1+perc/100)]
    alphaDRange = [pars['alpha_D']*(1-perc/100), pars['alpha_D']*(1+perc/100)]
    betaHatDRange = [pars['beta_hat_D']*(1-perc/100), 
                     pars['beta_hat_D']*(1+perc/100)]
    betaHatPRange = [pars['beta_hat_P']*(1-perc/100), 
                     pars['beta_hat_P']*(1+perc/100)]
    gammaDRange = [pars['gamma_D']*(1-perc/100), pars['gamma_D']*(1+perc/100)]
    gammaPRange = [pars['gamma_P']*(1-perc/100), pars['gamma_P']*(1+perc/100)]
    yRange = [0, 1]
    c1Range = [0, 1]
    c2Range = [-5, 5]
    vertRange = [0, 1]

    # Specify the columns of the dataframe and create it
    dfcols = ['b', 'alphaD', 'betaD', 'betaP', 'gammaD', 'gammaP', 'y', 'vert',
              'c1', 'c2', 'H', 'D', 'P', 'B', 'Hdot', 'Pdot', 'alphaP',
              'alphaPdot']
    df = pd.DataFrame(columns=dfcols)
    
    # Loop through the number of repeats
    for ii in tqdm(range(n), ncols=40, total=n):
    # for ii in range(n):

        # Create a new set of default parameters
        ode_pars = set_default_ode_parameters()

        # Choose each of the parameters
        ode_pars['b'] = (bRange[0] + np.diff(bRange)*np.random.rand())[0]
        ode_pars['alpha_D'] = (alphaDRange[0] + \
                                np.diff(alphaDRange)*np.random.rand())[0]
        ode_pars['beta_hat_D'] = (betaHatDRange[0] + \
                                np.diff(betaHatDRange)*np.random.rand())[0]
        ode_pars['beta_hat_P'] = (betaHatPRange[0] + \
                                np.diff(betaHatPRange)*np.random.rand())[0]
        ode_pars['gamma_D'] = (gammaDRange[0] + \
                                np.diff(gammaDRange)*np.random.rand())[0]
        ode_pars['gamma_P'] = (gammaPRange[0] + \
                                np.diff(gammaPRange)*np.random.rand())[0]
        ode_pars['resistance'] = (yRange[0] + \
                                  np.diff(yRange)*np.random.rand())[0]
        ode_pars['c1'] = (c1Range[0] + np.diff(c1Range)*np.random.rand())[0]
        ode_pars['c2'] = (c2Range[0] + np.diff(c2Range)*np.random.rand())[0]
        ode_pars['rho'] = (vertRange[0]+ np.diff(vertRange)*np.random.rand())[0]

        # Calculate the ancestral virulence
        alphaPdot = compute_ancestral_virulence(ode_pars)
        # print(f'{ii}: {ode_pars["beta_hat_P"]:.3f}, {alphaPdot:.3f}')

        # Calculate the ancestral population size
        Hdot, Pdot = compute_ancestral_steady_state(ode_pars)

        # Calculate the evolved virulence
        alphaP = fsolve(lambda alphaP: findAlphaP(alphaP, 
                                                ode_pars['resistance'], 
                                                ode_pars['c1'],
                                                ode_pars), x0=alphaPdot)
        # print(f'{ii}: {ode_pars["beta_hat_P"]:.3f}, {alphaP[0]:.3f}')
        
        # Calculate the steady state at this CSS
        ode_pars['alpha_P'] = alphaP[0]
        system = make_system(ode_pars)
        Ht, Dt, Pt, Bt = run_ode_solver(ode_pars, system)
        SS = approximate_steady_state(Ht, Dt, Pt, Bt)
        H = SS['H']
        D = SS['D']
        P = SS['P']
        B = SS['B']

        # Create a dictionary to contain the dataframe entry
        dfDict = {'b': [ode_pars['b']],
                  'alphaD': [ode_pars['alpha_D']],
                  'betaD': [ode_pars['beta_hat_D']],
                  'betaP': [ode_pars['beta_hat_P']],
                  'gammaD': [ode_pars['gamma_D']],
                  'gammaP': [ode_pars['gamma_P']],
                  'y': [ode_pars['resistance']],
                  'c1': [ode_pars['c1']],
                  'c2': [ode_pars['c2']],
                  'vert': [ode_pars['rho']],
                  'H': [H],
                  'D': [D],
                  'P': [P],
                  'B': [B],
                  'Hdot': [Hdot],
                  'Pdot': [Pdot],
                  'alphaP': [alphaP[0]],
                  'alphaPdot': [alphaPdot]}
        dfTemp = pd.DataFrame.from_dict(dfDict)
        df = pd.concat([df, dfTemp], ignore_index=True)

    if save == True:
        
        # Generate a directory in the data directory
        date = datetime.today().strftime('%Y-%m-%d')
        savedir = f'sensitivityVert_{date}'
        os.mkdir(f'./data/{savedir}')

        # Save the dataframe to the directory
        df.to_csv(f'./data/{savedir}/dataset.csv')
        
    else:
        return(df['H'])
    
def plotData(savedir, save=False):
    '''
    Plotting code for the data contained in savedir of the form YYYY-MM-DD
    '''

    # Load the data
    if os.path.isdir(f'./data/{savedir}'):
        df = pd.read_csv(f'./data/{savedir}/dataset.csv')
    else:
        print('No such directory')
        return
    
    # Create a figures directory
    if not os.path.isdir(f'./results/{savedir}'):
        os.mkdir(f'./results/{savedir}')
    else:
        print('Directory already exists, overwriting any figures')

    # Generate three figures, which will mirror the outputs of Fig. 2 from the
    # maintext 
    figWidth = 30
    figHeight = 40
    fsize = 32
    fig1, axs1 = plt.subplots(nrows=4, ncols=3, figsize=(figWidth, figHeight),
                                sharey=True)
    fig2, axs2 = plt.subplots(nrows=4, ncols=3, figsize=(figWidth, figHeight),
                                sharey=True)
    fig3, axs3 = plt.subplots(nrows=4, ncols=3, figsize=(figWidth, figHeight),
                                sharey=True)
    
    # Create a flag for extinction of D, extinction of P and coexistance
    df['coExist'] = np.logical_and(df['D'] + df['B'] > 1e-4,
                                   df['P'] + df['B'] > 1e-4)
    df['defExt'] = df['D'] + df['B'] < 1e-4
    df['parExt'] = df['P'] + df['B'] < 1e-4

    # Generate auxiliary rows
    N = np.array(df['H']) + np.array(df['D']) + np.array(df['P']) + \
        np.array(df['B'])
    Ndot = np.array(df['Hdot']) + np.array(df['Pdot'])
    if 'Q1' in df.columns():
        df['Q1'] = np.array(df['alphaP'])/np.array(df['alphaPdot'])
        df['Q2'] = N/Ndot    
        df['Q3'] = (np.array(df['P']) + np.array(df['B']))/np.array(df['Pdot'])

        df.to_csv(f'./data/{savedir}/dataset.csv')

    # Find the maximum and minimum Q1 values and round them to the nearest 0.1
    Q1min = np.floor(np.min(df['Q1'])*10)/10
    Q1max = np.ceil(np.max(df['Q1'])*10)/10
    Q2min = max(1, np.floor(np.min(df['Q2'])*10)/10)
    Q2max = np.ceil(np.max(df['Q2'])*10)/10
    Q3min = max(1, np.floor(np.min(df['Q3'])*10)/10)
    Q3max = np.ceil(np.max(df['Q3'])*10)/10
    
    # Define the order for plotting the parameters
    pars = ['b', 'alphaD', 'betaD', 'betaP', 
                'gammaD', 'gammaP', 'y', 'c1', 'c2', 'vert']

    # Loop through figures and parameters and plot
    for ii, axs in enumerate([axs1, axs2, axs3]):
        RHSVal = df['Q1']*(ii==0) + df['Q2']*(ii==1) + df['Q3']*(ii==2)
        if ii == 0:
            ymin = 1
        elif ii == 2:
            ymin = 0
        else:
            ymin = np.floor(np.min(RHSVal)*10)/10
        if ii == 2:
            ymax = 1
        else:
            ymax = np.ceil(np.max(RHSVal)*10)/10
        for jj, par in enumerate(pars):
            if jj < 9:
                ax = axs[int(np.floor(jj/3)), jj%3]
            else:
                ax = axs[3,1]
            ax.plot(df[df['coExist']][par], RHSVal[df['coExist']], 'r.',
                       ms=5, alpha=0.25)
            # ax.plot(df[df['defExt']][par], RHSVal[df['defExt']], 'k.',
            #         ms=5, alpha=0.25)
            # ax.plot(df[df['parExt']][par], RHSVal[df['parExt']], 'b.',
            #         ms=5, alpha=0.25)
            ax.set_xlabel(par, fontsize=fsize)
            if jj in [0, 4, 5]:
                xmin = np.floor(np.min(df[par])*100)/100
                xmax = np.ceil(np.max(df[par])*100)/100
            else:
                xmin = np.floor(np.min(df[par])*10)/10
                xmax = np.ceil(np.max(df[par])*10)/10
            ax.set_xlim([xmin, xmax])
            ax.set_xticks([xmin, xmax])
            if jj in [0, 4, 5]:
                ax.set_xticklabels([f'{xmin:.2f}', f'{xmax:.2f}'],
                        fontsize=fsize)
            else:
                ax.set_xticklabels([f'{xmin:.1f}', f'{xmax:.1f}'],
                        fontsize=fsize)
            ax.plot([xmin, xmax], [1,1], 'k', lw=5)
            axs[3,0].axis('off')
            axs[3,2].axis('off')

            # Find the line of best fit and the correlation matrix, and plot
            slopec, iceptc, rc, _, _ = linregress(df[df['coExist']][par],
                                                RHSVal[df['coExist']])
            ax.plot([xmin, xmax], [iceptc+slopec*xmin, iceptc+slopec*xmax], 
                        'k', lw=5)
            # sloped, iceptd, rd, _, _ = linregress(df[df['defExt']][par],
            #                                     RHSVal[df['defExt']])
            # ax.plot([xmin, xmax], [iceptd+sloped*xmin, iceptd+sloped*xmax], 
            #             'k', lw=5)
            # slopep, iceptp, rp, _, _ = linregress(df[df['parExt']][par],
            #                                     RHSVal[df['parExt']])
            # ax.plot([xmin, xmax], [iceptp+slopep*xmin, iceptp+slopep*xmax], 
            #             'b', lw=5)

            # Add the correlation coefficient to each plot
            ax.text((xmax+xmin)*0.5, ymax - (ymax-ymin)*0.1, 
                        f'correlation: {rc:.2f}',
                        ha ='center', fontsize=fsize)

    
        # Update the y axes
        axs[0,0].set_ylim([ymin, ymax])
        axs[1,0].set_ylim([ymin, ymax])
        axs[2,0].set_ylim([ymin, ymax])
        axs[0,0].set_yticks([ymin, ymax])
        axs[1,0].set_yticks([ymin, ymax])
        axs[2,0].set_yticks([ymin, ymax])
        axs[0,0].set_yticklabels([f'{ymin:.1f}', f'{ymax:.1f}'],
                    fontsize=fsize)
        axs[1,0].set_yticklabels([f'{ymin:.1f}', f'{ymax:.1f}'],
                    fontsize=fsize)
        axs[2,0].set_yticklabels([f'{ymin:.1f}', f'{ymax:.1f}'],
                    fontsize=fsize)

    axs1[1,0].set_ylabel('Relative evolved virulence', fontsize=fsize)
    axs2[1,0].set_ylabel('Relative host population size', fontsize=fsize)
    axs3[1,0].set_ylabel('Relative disease prevalence', fontsize=fsize)

    # Now add a common axis

    # # Axis labels
    # # Axis 1
    # ax1.set_xlim([0, 1])
    # ax1.set_xticks([0, 1])
    # ax1.set_xticklabels([0, 1], fontsize = fsize)
    # ax1.set_ylim([Q2min, Q2max])
    # ax1.set_yticks([Q2min, Q2max])
    # ax1.set_yticklabels([Q2min, Q2max], fontsize = fsize)
    # ax1.set_ylabel('Relative\nevolved virulence', fontsize=fsize)
    # ax1.text(0, Q2min + (Q2max-Q2min)*1.025, '(A)', fontsize=fsize)

    # # Axis 2
    # ax2.set_xlim([0, 1])
    # ax2.set_xticks([0, 0.5, 1])
    # ax2.set_xticklabels([0, 0.5, 1], fontsize = fsize)
    # ax2.set_ylim([Q1min, Q1max])
    # ax2.set_yticks([Q1min, 1, np.floor(Q1max)])
    # ax2.set_yticklabels([Q1min, 1, f'{int(np.floor(Q1max))}'], fontsize = fsize)
    # ax2.set_ylabel('Relative host\npopulation size', fontsize=fsize)
    # ax2.text(0.0, Q1min + (Q1max-Q1min)*1.025, '(B)', fontsize=fsize)

    # # Axis 3
    # ax3.set_xlim([0, 1])
    # ax3.set_xticks([0, 0.5, 1])
    # ax3.set_xticklabels([0, 0.5, 1], fontsize = fsize)
    # ax3.set_xlabel(r'Resistance, $y$', fontsize=fsize)
    # ax3.set_ylim([Q3min, Q3max])
    # ax3.set_yticks([Q3min, Q3max])
    # ax3.set_yticklabels([Q3min, Q3max], fontsize = fsize)
    # ax3.set_ylabel('Relative\ndisease prevalence', fontsize=fsize)
    # ax3.text(0.0, Q3min + (Q3max-Q3min)*1.025, '(C)', fontsize=fsize)

    if save:
        fig1.savefig(f'./results/{savedir}/sensitivity_figure_1.png',
                bbox_inches='tight')
        fig2.savefig(f'./results/{savedir}/sensitivity_figure_2.png',
                bbox_inches='tight')
        fig3.savefig(f'./results/{savedir}/sensitivity_figure_3.png',
            bbox_inches='tight')

    plt.show()

def compareInitial(saveDir, save = False):
    '''
    Plotting code to compare the outcomes between initial and evolved outcomes
    '''

    # Load the data
    if os.path.isdir(f'./data/{saveDir}'):
        df = pd.read_csv(f'./data/{saveDir}/dataset.csv')
    else:
        print('No such directory')
        return
    
    # Create a figures directory
    if not os.path.isdir(f'./results/{saveDir}'):
        os.mkdir(f'./results/{saveDir}')
    else:
        print('Directory already exists, overwriting any figures')

    # Generate new data
    initProt(saveDir)

    # Generate six figures similarly to the outputs from plotData
    figWidth = 30
    figHeight = 40
    fsize = 32
    fig1, axs1 = plt.subplots(nrows=4, ncols=3, figsize=(figWidth, figHeight),
                                sharey=True)
    fig2, axs2 = plt.subplots(nrows=4, ncols=3, figsize=(figWidth, figHeight),
                                sharey=True)
    fig3, axs3 = plt.subplots(nrows=4, ncols=3, figsize=(figWidth, figHeight),
                                sharey=True)
    fig4, axs4 = plt.subplots(nrows=4, ncols=3, figsize=(figWidth, figHeight),  
                                sharey=True)
    
    # Create a flag for extinction of D, extinction of P and coexistance
    df['coExist'] = np.logical_and(df['D'] + df['B'] > 1e-4,
                                   df['P'] + df['B'] > 1e-4)
    df['defExt'] = df['D'] + df['B'] < 1e-4
    df['parExt'] = df['P'] + df['B'] < 1e-4

    # Set the limits for the axes
    Q2min = np.floor(np.min(df[df['coExist']]['Q2Init'])*10)/10
    Q2max = np.ceil(np.max(df[df['coExist']]['Q2Init'])*10)/10
    Q3min = np.floor(np.min(df[df['coExist']]['Q3Init'])*10)/10
    Q3max = np.ceil(np.max(df[df['coExist']]['Q3Init'])*10)/10
    Q2mn = np.floor(np.min(df[df['coExist']]['Q2'])*10)/10
    Q2mx = np.ceil(np.max(df[df['coExist']]['Q2'])*10)/10
    Q3mn = np.floor(np.min(df[df['coExist']]['Q3'])*10)/10
    Q3mx = np.ceil(np.max(df[df['coExist']]['Q3'])*10)/10

    # Define the order for plotting the parameters
    pars = ['b', 'alphaD', 'betaD', 'betaP', 
                'gammaD', 'gammaP', 'y', 'c1', 'c2', 'vert']

    # Loop through figures and parameters and plot
    for ii, axs in enumerate([axs1, axs2, axs3, axs4]):

        # Find the y axis variables
        yVal = df[df['coExist']]['Q2Init']*(ii==0) + \
            df[df['coExist']]['Q3Init']*(ii==1) + \
            df[df['coExist']]['Q2']*(ii==2) + \
            df[df['coExist']]['Q3']*(ii==3)
        
        # Loop through the parameters
        for jj, par in enumerate(pars):

            # Find the x value
            xVal = df[df['coExist']][par]*(ii in [0,1]) + \
                df[df['coExist']]['Q2Init']*(ii==2) + \
                df[df['coExist']]['Q3Init']*(ii==3)

            # Plot the data
            if jj < 9:
                ax = axs[int(np.floor(jj/3)), jj%3]
            else:
                ax = axs[3,1]
            ax.plot(xVal, yVal, 'r.', ms=5, alpha=0.25)
            ax.set_xlabel(par, fontsize=fsize)

            # Set the limits
            # x limits
            if jj in [0, 4, 5]:
                xmin = np.floor(np.min(df[df['coExist']][par])*100)/100
                xmax = np.ceil(np.max(df[df['coExist']][par])*100)/100
            else:
                xmin = np.floor(np.min(df[df['coExist']][par])*10)/10
                xmax = np.ceil(np.max(df[df['coExist']][par])*10)/10
            ax.set_xlim([xmin, xmax])
            ax.set_xticks([xmin, xmax])
            if jj in [0, 4, 5]:
                ax.set_xticklabels([f'{xmin:.2f}', f'{xmax:.2f}'],
                        fontsize=fsize)
            else:
                ax.set_xticklabels([f'{xmin:.1f}', f'{xmax:.1f}'],
                        fontsize=fsize)
            axs[3,0].axis('off')
            axs[3,2].axis('off')
            
            # y limits
            if ii == 0:
                ymin = Q2min
                ymax = Q2max
            elif ii == 1:
                ymin = Q3min
                ymax = Q3max
            elif ii == 2:
                xmin = Q2min
                xmax = Q2max
                ymin = Q2mn
                ymax = Q2mx
            else:
                xmin = Q3min
                xmax = Q3max
                ymin = Q3mn
                ymax = Q3mx
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])

            if ii in [0,1]:
                ax.plot([xmin, xmax], [1,1], 'k', lw=5)

                # Find the line of best fit and the correlation matrix, and plot
                slopec, iceptc, rc, _, _ = linregress(xVal, yVal)
                ax.plot([xmin, xmax], [iceptc+slopec*xmin, iceptc+slopec*xmax], 
                        'r', lw=5)
                
                # Add the correlation coefficient to each plot
                ax.text((xmax+xmin)*0.5, ymax - (ymax-ymin)*0.1, 
                            f'correlation: {rc:.2f}',
                            ha ='center', fontsize=fsize)

            else:
                ax.plot([xmin, xmax], [xmin, xmax], 'k', lw=5)
            
            # Update the y axes
            axs[0,0].set_ylim([ymin, ymax])
            axs[1,0].set_ylim([ymin, ymax])
            axs[2,0].set_ylim([ymin, ymax])
            axs[0,0].set_yticks([ymin, ymax])
            axs[1,0].set_yticks([ymin, ymax])
            axs[2,0].set_yticks([ymin, ymax])
            axs[0,0].set_yticklabels([f'{ymin:.1f}', f'{ymax:.1f}'],
                        fontsize=fsize)
            axs[1,0].set_yticklabels([f'{ymin:.1f}', f'{ymax:.1f}'],
                        fontsize=fsize)
            axs[2,0].set_yticklabels([f'{ymin:.1f}', f'{ymax:.1f}'],
                    fontsize=fsize)
            
            if ii == 2:
                for ax in list(axs.flatten()):
                    ax.set_xlabel('')
                    ax.set_xticks([xmin, xmax])
                axs[3,1].set_xlabel('Initial relative population size',
                                fontsize=fsize)

            if ii == 3:
                for ax in list(axs.flatten()):
                    ax.set_xlabel('')
                    ax.set_xticks([xmin, xmax])
                axs[3,1].set_xlabel('Initial relative disease prevalence',
                                fontsize=fsize)
            
        axs1[1,0].set_ylabel('Relative host population size', fontsize=fsize)
        axs2[1,0].set_ylabel('Relative disease prevalence', fontsize=fsize)
        axs3[1,0].set_ylabel('Relative host population size', fontsize=fsize)
        axs4[1,0].set_ylabel('Relative disease prevalence', fontsize=fsize)

    if save:
        fig1.savefig(f'./results/{saveDir}/sensitivityInit_figure_2.png',
                bbox_inches='tight')
        fig2.savefig(f'./results/{saveDir}/sensitivityInit_figure_3.png',
                bbox_inches='tight')
        fig3.savefig(f'./results/{saveDir}/sensitivityInit_compare_2.png',
                bbox_inches='tight')
        fig4.savefig(f'./results/{saveDir}/sensitivityInit_compare_3.png',
                bbox_inches='tight')
        

    plt.show()

if __name__ == '__main__':

    np.random.seed(1)

    # Testing
    if len(sys.argv) < 2:
        # print(generateData(n=25))
        # generateData(n = 5000, save = True)
        # plotData('sensitivity_2024-08-02')
        compareInitial('sensitivityVert_2024-10-21')
    
    # Saving data
    elif sys.argv[1] == 'save':
        generateData(n = 10000, save = True)

    # Plotting in window
    elif sys.argv[1] == 'plot':
        plotData(sys.argv[2])

    # Save plotting with no cost
    elif sys.argv[1] == 'saveplot':
        plotData(sys.argv[2], save=True)

    # elif sys.argv[1] == 'saveplotcost':
    #     plotDataCost(sys.argv[2], save=True)

    elif sys.argv[1] == 'saveplotinit':
        compareInitial(sys.argv[2], save=True)
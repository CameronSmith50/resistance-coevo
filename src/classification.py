"""
classification.py
Functions required to classify 

author: Cameron Smith
based from: https://github.com/CameronSmith50/Defensive-Symbiosis/blob/main/Virulence_Coevo.py
"""

#%%

from coevolutionary_simulation import *
import pickle as pkl
import pandas as pd
import os
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from ast import literal_eval
from copy import deepcopy

#%%

def classifyc1c2(c1, c2, parameters, level=0):

    # Create a dictionary for the dataframe
    # Create the dataframe columns
    dfCols = ['level', 'c1', 'c2', 'y0', 'y1', 'alpha_P0', 'alpha_P1', 'classify']
    df = pd.DataFrame(columns=dfCols)

    df1Dict = {
        'level': [level],
        'c1': [c1],
        'c2': [c2]
    }

    for y0 in [0,1]:

        # Initialise a simulation
        parameters['c1'] = c1
        parameters['c2'] = c2
        parameters['alpha_P'] = compute_ancestral_virulence(parameters)
        parameters['resistance'] = y0
        discretized_system = make_discretized_system_coevolution(parameters)

        MIN_VIRULENCE = 0.0
        MAX_VIRULENCE = 1.0
        NUMBER_OF_VIRULENCE_VALUES = 21
        virulence_vector = np.linspace(MIN_VIRULENCE, MAX_VIRULENCE, NUMBER_OF_VIRULENCE_VALUES)

        MIN_RESISTANCE = 0.0
        MAX_RESISTANCE = 1.0
        NUMBER_OF_RESISTANCE_VALUES = 21
        resistance_vector = np.linspace(MIN_RESISTANCE, MAX_RESISTANCE, NUMBER_OF_RESISTANCE_VALUES)

        distance_between_virulence_values = virulence_vector[1] - virulence_vector[0]
        initial_virulence_index = round( (parameters['alpha_P'] - MIN_VIRULENCE)/distance_between_virulence_values )

        distance_between_resistance_values = resistance_vector[1] - resistance_vector[0]
        initial_resistance_index = round( (parameters['resistance'] - MIN_RESISTANCE)/distance_between_resistance_values )

        initial_state = build_initial_state_coevolution(initial_resistance_index, initial_virulence_index, resistance_vector, virulence_vector)

        NUMBER_OF_TIMESTEPS = 1000
        evolutionary_timesteps = np.arange(NUMBER_OF_TIMESTEPS)
        
        extinction_threshold = 1e-5

        # Simulate
        defensive_symbiont_frequencies, parasite_frequencies = coevolutionary_simulation(
            parameters,
            discretized_system,
            initial_state,
            initial_resistance_index,
            initial_virulence_index,
            resistance_vector,
            virulence_vector,
            evolutionary_timesteps,
            extinction_threshold
        )

        # Look at the final time and write the values of y and alpha_P to the dataframe.
        DD = defensive_symbiont_frequencies[-1,:]
        PP = parasite_frequencies[-1,:]
        # DD = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # PP = np.array([0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3])

        # Check for a branch in defensive symbiont
        # Which of each type exist?
        nRes = np.arange(NUMBER_OF_RESISTANCE_VALUES) + 1
        nVir = np.arange(NUMBER_OF_VIRULENCE_VALUES) + 1
        Dpop = DD > 1e-8
        Ppop = PP > 1e-8
        DRemain = (Dpop*nRes)[Dpop*nRes > 0] - 1
        PRemain = (Ppop*nVir)[Ppop*nVir > 0] - 1
        dBrInd = np.diff(DRemain)
        dBrInd = DRemain[:-1][dBrInd > 4]
        pBrInd = np.diff(PRemain)
        pBrInd = PRemain[:-1][pBrInd > 4]

        # How many values do we have
        y_vals = []
        alpha_P_vals = []

        # Save to the _vals vectors
        if len(DRemain) == 0:
            y_vals.append(np.nan)
        elif len(dBrInd) == 0:
            y_vals.append(np.sum(DD*resistance_vector))
        else:
            y_vals.append(np.sum(DD[:dBrInd[0]]*resistance_vector[:dBrInd[0]])/np.sum(DD[:dBrInd[0]]))
            for dInd, dbr in enumerate(dBrInd[0:-1]):
                y_vals.append(np.sum(DD[dBrInd[dInd]:dBrInd[dInd+1]]*resistance_vector[dBrInd[dInd]:dBrInd[dInd+1]])/np.sum(DD[dBrInd[dInd]:dBrInd[dInd+1]]))
            y_vals.append(np.sum(DD[dBrInd[-1]:]*resistance_vector[dBrInd[-1]:])/np.sum(DD[dBrInd[-1]:]))

        if len(PRemain) == 0:
            alpha_P_vals.append(np.nan)
        elif len(pBrInd) == 0:
            alpha_P_vals.append(np.sum(PP*virulence_vector))
        else:
            alpha_P_vals.append(np.sum(PP[:pBrInd[0]]*virulence_vector[:pBrInd[0]])/np.sum(PP[:pBrInd[0]]))
            for pInd, pbr in enumerate(pBrInd[0:-1]):
                alpha_P_vals.append(np.sum(PP[pBrInd[pInd]:pBrInd[pInd+1]]*virulence_vector[pBrInd[pInd]:pBrInd[pInd+1]])/np.sum(PP[pBrInd[pInd]:pBrInd[pInd+1]]))
            alpha_P_vals.append(np.sum(PP[pBrInd[-1]:]*virulence_vector[pBrInd[-1]:])/np.sum(PP[pBrInd[-1]:]))

        if y0 == 0:
            df1Dict['y0'] = [y_vals]
            df1Dict['alpha_P0'] = [alpha_P_vals]
        else:
            df1Dict['y1'] = [y_vals]
            df1Dict['alpha_P1'] = [alpha_P_vals]

    # Classification
    # We now classify this simulation. We will classif the def
    # sybiont and parasite separately according to the following
    # values:
    # > 1: Maximiser
    # > 2: Minimiser
    # > 3: CSS
    # > 4: Repeller
    # > 5: Branching
    # > 6: Extinction
    # > 7: Potential repeller - only occurs if extinction
    classList = [[],[]]

    # Support Booleans
    extinct_y0 = np.nan in df1Dict['y0'][0]
    extinct_y1 = np.nan in df1Dict['y1'][0]
    extinct_alpha_P0 = np.nan in df1Dict['alpha_P0'][0]
    extinct_alpha_P1 = np.nan in df1Dict['alpha_P1'][0]
    branch_y0 = len(df1Dict['y0'][0]) > 1
    branch_y1 = len(df1Dict['y1'][0]) > 1
    branch_alpha_P0 = len(df1Dict['alpha_P0'][0]) > 1
    branch_alpha_P1 = len(df1Dict['alpha_P1'][0]) > 1

    # Defensive symbiont
    # We will start with repellers.
    # This first group has no extinction or branching
    if not (extinct_y0 and extinct_y1) and not (branch_y0 or branch_y1) and np.abs(df1Dict['y0'][0][0] - df1Dict['y1'][0][0]) >= 5e-2:
        classList[0].append(4)
        if df1Dict['y0'][0][0] <= (MIN_RESISTANCE + 5e-2) or df1Dict['y1'][0][0] <= (MIN_RESISTANCE + 5e-2):
            classList[0].append(2)
        if df1Dict['y0'][0][0] >= (MAX_RESISTANCE - 5e-2) or df1Dict['y1'][0][0] >= (MAX_RESISTANCE - 5e-2):
            classList[0].append(1)
        if (MIN_RESISTANCE + 5e-2) <= df1Dict['y0'][0][0] <= (MAX_RESISTANCE - 5e-2) or (MIN_RESISTANCE + 5e-2) <= df1Dict['y1'][0][0] <= (MAX_RESISTANCE - 5e-2):
            classList[0].append(3)

    # Now we include branching
    elif not (extinct_y0 and extinct_y1) and branch_y0 and not branch_y1:
        classList[0].append(4)
        classList[0].append(5)
        if df1Dict['y1'][0][0] <= (MIN_RESISTANCE + 5e-2):
            classList[0].append(2)
        elif df1Dict['y1'][0][0] >= (MAX_RESISTANCE - 5e-2):
            classList[0].append(1)
        else:
            classList[0].append(3)
    elif not (extinct_y0 and extinct_y1) and not branch_y0 and branch_y1:
        classList[0].append(4)
        classList[0].append(5)
        if df1Dict['y0'][0][0] <= (MIN_RESISTANCE + 5e-2):
            classList[0].append(2)
        elif df1Dict['y0'][0][0] >= (MAX_RESISTANCE - 5e-2):
            classList[0].append(1)
        else:
            classList[0].append(3)

    # Remove the repeller
    elif not (extinct_y0 and extinct_y1) and not (branch_y0 or branch_y1):
        if df1Dict['y0'][0][0] <= (MIN_RESISTANCE + 5e-2):
            classList[0].append(2)
        elif df1Dict['y0'][0][0] >= (MAX_RESISTANCE - 5e-2):
            classList[0].append(1)
        else:
            classList[0].append(3)

    # Now check if both have branched. Here we assume that if they
    # do both branch then it's the same branching point
    if branch_y0 and branch_y1:
        classList[0].append(5)

    # Now we deal with extinction
    # Both extinct
    if extinct_y0 and extinct_y1 and not (branch_y0 or branch_y1):
        classList[0].append(6)
    elif extinct_y0 and not extinct_y1 and not branch_y1:
        classList[0].append(6)
        classList[0].append(7)
        if df1Dict['y1'][0][0] <= (MIN_RESISTANCE + 5e-2):
            classList[0].append(2)
        elif df1Dict['y1'][0][0] >= (MAX_RESISTANCE - 5e-2):
            classList[0].append(1)
        elif (MIN_RESISTANCE + 5e-2) <= df1Dict['y1'][0][0] <= (MAX_RESISTANCE - 5e-2):
            classList[0].append(3)
    elif extinct_y0 and not extinct_y1 and branch_y1:
        classList[0].append(6)
        classList[0].append(7)
        classList[0].append(5)
    elif not extinct_y0 and extinct_y1 and not branch_y0:
        classList[0].append(6)
        classList[0].append(7)
        if df1Dict['y0'][0][0] <= (MIN_RESISTANCE + 5e-2):
            classList[0].append(2)
        elif df1Dict['y0'][0][0] >= (MAX_RESISTANCE - 5e-2):
            classList[0].append(1)
        elif (MIN_RESISTANCE + 5e-2) <= df1Dict['y0'][0][0] <= (MAX_RESISTANCE - 5e-2):
            classList[0].append(3)
    elif not extinct_y0 and extinct_y1 and branch_y0:
        classList[0].append(6)
        classList[0].append(7)
        classList[0].append(5)
        
    # Parasite
    # We will start with repellers.
    # This first group has no extinction or branching
    if not (extinct_alpha_P0 and extinct_alpha_P1) and not (branch_alpha_P0 or branch_alpha_P1) and np.abs(df1Dict['alpha_P0'][0][0] - df1Dict['alpha_P1'][0][0]) >= 5e-2:
        classList[1].append(4)
        if df1Dict['alpha_P0'][0][0] <= (MIN_VIRULENCE + 5e-2) or df1Dict['alpha_P1'][0][0] <= (MIN_VIRULENCE + 5e-2):
            classList[1].append(2)
        if df1Dict['alpha_P0'][0][0] >= (MAX_VIRULENCE - 5e-2) or df1Dict['alpha_P1'][0][0] >= (MAX_VIRULENCE - 5e-2):
            classList[1].append(1)
        if (MIN_VIRULENCE + 5e-2) <= df1Dict['alpha_P0'][0][0] <= (MAX_VIRULENCE - 5e-2) or (MIN_VIRULENCE + 5e-2) <= df1Dict['alpha_P1'][0][0] <= (MAX_VIRULENCE - 5e-2):
            classList[1].append(3)

    # Now we include branching
    elif not (extinct_alpha_P0 and extinct_alpha_P1) and branch_alpha_P0 and not branch_alpha_P1:
        classList[1].append(4)
        classList[1].append(5)
        if df1Dict['alpha_P1'][0][0] <= (MIN_VIRULENCE + 5e-2):
            classList[1].append(2)
        elif df1Dict['alpha_P1'][0][0] >= (MAX_VIRULENCE - 5e-2):
            classList[1].append(1)
        else:
            classList[1].append(3)
    elif not (extinct_alpha_P0 and extinct_alpha_P1) and not branch_alpha_P0 and branch_alpha_P1:
        classList[1].append(4)
        classList[1].append(5)
        if df1Dict['alpha_P0'][0][0] <= (MIN_VIRULENCE + 5e-2):
            classList[1].append(2)
        elif df1Dict['alpha_P0'][0][0] >= (MAX_VIRULENCE - 5e-2):
            classList[1].append(1)
        else:
            classList[1].append(3)

    # Remove the repeller
    elif not (extinct_alpha_P0 and extinct_alpha_P1) and not (branch_alpha_P0 or branch_alpha_P1):
        if df1Dict['alpha_P0'][0][0] <= (MIN_VIRULENCE + 5e-2):
            classList[1].append(2)
        elif df1Dict['alpha_P0'][0][0] >= (MAX_VIRULENCE - 5e-2):
            classList[1].append(1)
        else:
            classList[1].append(3)

    # Now check if both have branched. Here we assume that if they
    # do both branch then it's the same branching point
    if branch_alpha_P0 and branch_alpha_P1:
        classList[1].append(5)

    # Now we deal with extinction
    # Both extinct
    if extinct_alpha_P0 and extinct_alpha_P1 and not (branch_alpha_P0 or branch_alpha_P1):
        classList[1].append(6)
    elif extinct_alpha_P0 and not extinct_alpha_P1 and not branch_alpha_P1:
        classList[1].append(6)
        classList[1].append(7)
        if df1Dict['alpha_P1'][0][0] <= (MIN_VIRULENCE + 5e-2):
            classList[1].append(2)
        elif df1Dict['alpha_P1'][0][0] >= (MAX_VIRULENCE - 5e-2):
            classList[1].append(1)
        elif (MIN_VIRULENCE + 5e-2) <= df1Dict['alpha_P1'][0][0] <= (MAX_VIRULENCE - 5e-2):
            classList[1].append(3)
    elif extinct_alpha_P0 and not extinct_alpha_P1 and branch_alpha_P1:
        classList[1].append(6)
        classList[1].append(7)
        classList[1].append(5)
    elif not extinct_alpha_P0 and extinct_alpha_P1 and not branch_alpha_P0:
        classList[1].append(6)
        classList[1].append(7)
        if df1Dict['alpha_P0'][0][0] <= (MIN_VIRULENCE + 5e-2):
            classList[1].append(2)
        elif df1Dict['alpha_P0'][0][0] >= (MAX_VIRULENCE - 5e-2):
            classList[1].append(1)
        elif (MIN_VIRULENCE + 5e-2) <= df1Dict['alpha_P0'][0][0] <= (MAX_VIRULENCE - 5e-2):
            classList[1].append(3)
    elif not extinct_alpha_P0 and extinct_alpha_P1 and branch_alpha_P0:
        classList[1].append(6)
        classList[1].append(7)
        classList[1].append(5)

    # Add to the dictionary
    classList[0] = list(sorted(np.unique(classList[0])))
    classList[1] = list(sorted(np.unique(classList[1])))
    df1Dict['classify'] = [classList]

    # Add to the dataframe
    df1 = pd.DataFrame(df1Dict)
    df = pd.concat([df, df1], ignore_index=True)

    return(df)

def initialClassify(saveDataDir, c1min=0.0, c1max=1.0, c2min=-5.0, c2max=5.0, n=21):
    """
    A function to conduct an initial classification. Will create 
    the save directory specified and then add the base parameters.
    Will also output a dataframe which will later be used for plotting.
    """

    # Create the directory. If it already exists, warn the user and exit
    if os.path.isdir(saveDataDir):
        print('Data directory already exists. Please specify a new one.')
        return
    else:
        os.mkdir(saveDataDir)

        # ODE parameters
        ode_parameters = set_default_ode_parameters()

        # Firsty we need to create and save a parameters list
        with open(saveDataDir + 'Parameters.txt', 'w') as f:
            f.write('Parameters\n----------\n')
            for key, val in ode_parameters.items():
                f.write(f'{key}={val}\n')
            f.write(f'c1min={c1min}\n')
            f.write(f'c1max={c1max}\n')
            f.write(f'c2min={c2min}\n')
            f.write(f'c2max={c2max}\n')
            f.write(f'nLevel1={n}')
        
        # Generate the c1 and c2 vectors
        c1vec = np.linspace(c1min, c1max, n)
        c2vec = np.linspace(c2min, c2max, n)

        # Create the dataframe columns
        dfCols = ['level', 'c1', 'c2', 'y0', 'y1', 'alpha_P0', 'alpha_P1', 'classify']
        df = pd.DataFrame(columns=dfCols)

        # Loop through the c1 and c2 vectors
        for c1Ind, c1 in tqdm(enumerate(c1vec), total=len(c1vec), leave=False, ncols=50, desc='c1'):
            for c2Ind, c2 in tqdm(enumerate(c2vec), total=len(c2vec), leave=False, ncols=50, desc='c2'):
                
                # Run the classification for this c1, c2 pair
                df1 = classifyc1c2(c1, c2, ode_parameters)
                
                # Add to the dataframe
                df = pd.concat([df, df1], ignore_index=True)

        df.to_csv(saveDataDir + 'classifyDF.csv', index=False)

def refinementStep(dataDir):
    """
    Code to refine the classification
    """

    # Check if the data directory exists
    if not os.path.isdir(dataDir):
        print('No such directory. Please check you have written your input correcyly, or ran initialClassify to get some data.')
    else:

        # Extract the dataframe and the parameters
        df = pd.read_csv(dataDir + 'classifyDF.csv')
        parameters = {}
        with open(dataDir + 'parameters.txt', 'r') as f:
            lines = [kk for kk in f][2:]
            for ll in lines[:-1]:
                key, val = ll.split('=')
                parameters[key] = float(val[:-1])
            key, val = lines[-1].split('=')
            parameters[key] = float(val)

        # Find the number of levels in the dataframe
        levels = sorted(np.unique(list(df['level'])))
        nLevel = len(levels)

        # Find the next level
        level = levels[-1] + 1

        # Find the value of n
        n = int(np.round((parameters[f'nLevel{nLevel}'] - 1)*2 + 1))

        # Write this to the parameters file
        with open(dataDir + 'parameters.txt', 'a') as f:
            f.write(f'\nnLevel{level+1}={n}')

        # Find the c1 and c2 vectors
        c1Vec = np.linspace(parameters['c1min'], parameters['c1max'], n)
        c2Vec = np.linspace(parameters['c2min'], parameters['c2max'], n)
        dc1 = c1Vec[1] - c1Vec[0]
        dc2 = c2Vec[1] - c2Vec[0]

        # Create a matrix which will store the classifications on each side
        noneArray = [None]*n
        Dmat = [noneArray[:] for ii in range(n)]
        Pmat = [noneArray[:] for ii in range(n)]

        # Loop through the dataframe and store the relevant classification in the matrix
        for index, row in df.iterrows():

            # Find the index to store to
            c1Ind = int(np.round((row['c1'] - c1Vec[0])/dc1))
            c2Ind = int(np.round((row['c2'] - c2Vec[0])/dc2))

            # Store
            classify = literal_eval(row['classify'])
            Dmat[c2Ind][c1Ind] = classify[0]
            Pmat[c2Ind][c1Ind] = classify[1]

        # Loop through the unknown columns on the known rows. Compare either side. If
        # the same then imput and add this to dataframe. Otherwise, run a simulation
        # and add this way.
        for c1I, c1 in tqdm(enumerate(c1Vec[::2]), total=len(c1Vec[::2]), leave=False, ncols=50, desc='rUcK, c1'):
            for c2I, c2 in tqdm(enumerate(c2Vec[1::2]), total=len(c2Vec[1::2]), leave=False, ncols=50, desc='rUcK, c2'):
                
                # Alter c1Ind and c2Ind
                c1Ind = c1I*2
                c2Ind = (c2I + 1)*2 - 1

                # Check either side in both matrices
                if Dmat[c2Ind-1][c1Ind] == Dmat[c2Ind+1][c1Ind] and Pmat[c2Ind-1][c1Ind] == Pmat[c2Ind+1][c1Ind]:

                    # We impute the classification and don't need to run the simulation
                    Dmat[c2Ind][c1Ind] = deepcopy(Dmat[c2Ind-1][c1Ind])
                    Pmat[c2Ind][c1Ind] = deepcopy(Pmat[c2Ind-1][c1Ind])

                    # Save to the dataframe
                    df1Dict = {
                        'level': [level],
                        'c1': [c1],
                        'c2': [c2],
                        'y0': ['Classification Imputed'],
                        'y1': ['Classification Imputed'],
                        'alpha_P0': ['Classification Imputed'],
                        'alpha_P1': ['Classification Imputed'],
                        'classify': [[Dmat[c2Ind][c1Ind], Pmat[c2Ind][c1Ind]]]
                    }

                    # Add to the dataframe
                    df1 = pd.DataFrame(df1Dict)
                    df = pd.concat([df, df1], ignore_index=True)

                else:

                    # Run the dataframe
                    df1 = classifyc1c2(c1, c2, parameters, level=level)

                    # Add to the matrix
                    Dmat[c2Ind][c1Ind] = list(df1['classify'])[0][0]
                    Pmat[c2Ind][c1Ind] = list(df1['classify'])[0][1]

                    # Combine
                    df = pd.concat([df, df1])

        # Now known columns and unknown rows
        for c1I, c1 in tqdm(enumerate(c1Vec[1::2]), total=len(c1Vec[1::2]), leave=False, ncols=50, desc='rKcU, c1'):
            for c2I, c2 in tqdm(enumerate(c2Vec[::2]), total=len(c2Vec[::2]), leave=False, ncols=50, desc='rKcU, c2'):
                
                # Alter c1Ind and c2Ind
                c1Ind = (c1I + 1)*2 - 1
                c2Ind = c2I*2

                # Check either side in both matrices
                if Dmat[c2Ind][c1Ind-1] == Dmat[c2Ind][c1Ind+1] and Pmat[c2Ind][c1Ind-1] == Pmat[c2Ind][c1Ind+1]:

                    # We impute the classification and don't need to run the simulation
                    Dmat[c2Ind][c1Ind] = deepcopy(Dmat[c2Ind][c1Ind-1])
                    Pmat[c2Ind][c1Ind] = deepcopy(Pmat[c2Ind][c1Ind-1])

                    # Save to the dataframe
                    df1Dict = {
                        'level': [level],
                        'c1': [c1],
                        'c2': [c2],
                        'y0': ['Classification Imputed'],
                        'y1': ['Classification Imputed'],
                        'alpha_P0': ['Classification Imputed'],
                        'alpha_P1': ['Classification Imputed'],
                        'classify': [[Dmat[c2Ind][c1Ind], Pmat[c2Ind][c1Ind]]]
                    }

                    # Add to the dataframe
                    df1 = pd.DataFrame(df1Dict)
                    df = pd.concat([df, df1], ignore_index=True)

                else:

                    # Run the dataframe
                    df1 = classifyc1c2(c1, c2, parameters, level=level)

                    # Add to the matrix
                    Dmat[c2Ind][c1Ind] = list(df1['classify'])[0][0]
                    Pmat[c2Ind][c1Ind] = list(df1['classify'])[0][1]

                    # Combine
                    df = pd.concat([df, df1])

        # Finally, row and column unknown
        for c1I, c1 in tqdm(enumerate(c1Vec[1::2]), total=len(c1Vec[1::2]), leave=False, ncols=50, desc='rUcU, c1:'):
            for c2I, c2 in tqdm(enumerate(c2Vec[1::2]), total=len(c2Vec[1::2]), leave=False, ncols=50, desc='rUcU, c2:'):
                
                # Alter c1Ind and c2Ind
                c1Ind = (c1I + 1)*2 - 1
                c2Ind = (c2I + 1)*2 - 1

                # Check either side in both matrices
                if Dmat[c2Ind][c1Ind-1] == Dmat[c2Ind][c1Ind+1] and Pmat[c2Ind][c1Ind-1] == Pmat[c2Ind][c1Ind+1]:

                    # We impute the classification and don't need to run the simulation
                    Dmat[c2Ind][c1Ind] = deepcopy(Dmat[c2Ind][c1Ind-1])
                    Pmat[c2Ind][c1Ind] = deepcopy(Pmat[c2Ind][c1Ind-1])

                    # Save to the dataframe
                    df1Dict = {
                        'level': [level],
                        'c1': [c1],
                        'c2': [c2],
                        'y0': ['Classification Imputed'],
                        'y1': ['Classification Imputed'],
                        'alpha_P0': ['Classification Imputed'],
                        'alpha_P1': ['Classification Imputed'],
                        'classify': [[Dmat[c2Ind][c1Ind], Pmat[c2Ind][c1Ind]]]
                    }

                    # Add to the dataframe
                    df1 = pd.DataFrame(df1Dict)
                    df = pd.concat([df, df1], ignore_index=True)

                else:

                    # Run the dataframe
                    df1 = classifyc1c2(c1, c2, parameters, level=level)

                    # Add to the matrix
                    Dmat[c2Ind][c1Ind] = list(df1['classify'])[0][0]
                    Pmat[c2Ind][c1Ind] = list(df1['classify'])[0][1]

                    # Combine
                    df = pd.concat([df, df1])
                    
        df.to_csv(dataDir + 'classifyDF.csv', index=False)

def plottingClassify(dataDir):
    """
    Code which will plot any classification data in dataDir
    """

    # Check if the data directory exists
    if not os.path.isdir(dataDir):
        print(f'No such directory at {dataDir}. Please check you have written your input correctly, or ran initialClassify to get some data.')
    else:

        # Extract the dataframe and the parameters
        df = pd.read_csv(dataDir + 'classifyDF.csv')
        parameters = {}
        with open(dataDir + 'Parameters.txt', 'r') as f:
            lines = [kk for kk in f][2:]
            for ll in lines[:-1]:
                key, val = ll.split('=')
                parameters[key] = float(val[:-1])
            key, val = lines[-1].split('=')
            parameters[key] = float(val)

        # Find the number of levels in the dataframe
        levels = sorted(np.unique(list(df['level'])))
        nLevel = len(levels)

        # Create several lists for the different levels
        nVec = [int(parameters[f'nLevel{kk+1}']) for kk in range(nLevel)]
        c1Vecs = [np.linspace(parameters['c1min'], parameters['c1max'], nVec[kk]) for kk in range(nLevel)]
        c2Vecs = [np.linspace(parameters['c2min'], parameters['c2max'], nVec[kk]) for kk in range(nLevel)]
        dfs = [df[df['level']<=levels[kk]] for kk in levels]
        matDrep = [np.zeros((int(parameters[f'nLevel{kk+1}']), nVec[kk])) for kk in range(nLevel)]
        matDmin = [np.zeros((int(parameters[f'nLevel{kk+1}']), nVec[kk])) for kk in range(nLevel)]
        matDmax = [np.zeros((int(parameters[f'nLevel{kk+1}']), nVec[kk])) for kk in range(nLevel)]
        matDCSS = [np.zeros((int(parameters[f'nLevel{kk+1}']), nVec[kk])) for kk in range(nLevel)]
        matDbra = [np.zeros((int(parameters[f'nLevel{kk+1}']), nVec[kk])) for kk in range(nLevel)]
        matDext = [np.zeros((int(parameters[f'nLevel{kk+1}']), nVec[kk])) for kk in range(nLevel)]
        matDpos = [np.zeros((int(parameters[f'nLevel{kk+1}']), nVec[kk])) for kk in range(nLevel)]
        matPext = [np.zeros((int(parameters[f'nLevel{kk+1}']), nVec[kk])) for kk in range(nLevel)]
        DExtBoundc1 = [ [] for _ in range(nLevel) ]
        DExtBoundc2 = [ [] for _ in range(nLevel) ]
        PExtBoundc1 = [ [] for _ in range(nLevel) ]
        PExtBoundc2 = [ [] for _ in range(nLevel) ]
        
        # Create a colourmap for each index
        cmapMin = ListedColormap(['#ffffff00', '#1f78b4'])
        cmapMax = ListedColormap(['#ffffff00', '#a6cee3'])
        cmapCSS = ListedColormap(['#ffffff00', '#33a02c'])
        cmapRep = ListedColormap(['#ffffff00', '#b2df8a'])
        cmapBra = ListedColormap(['#ffffff00', '#ff000060'])
        cmapDExt = ListedColormap(['#ffffff00', '#00000040'])
        cmapPExt = ListedColormap(['#ffffff00', '#00000080'])

        # Loop through the levels and populate each of the matrices
        for level in levels:

            # Find the size in c1 and c2 vectors at this level
            dc1 = c1Vecs[level][1] - c1Vecs[level][0]
            dc2 = c2Vecs[level][1] - c2Vecs[level][0]

            # Loop through the rows of the dataframe
            for ii in range(nVec[level]**2):

                # Extract the values of c1, c2 and the classification for each row in turn
                dfEntry = dfs[level].iloc[ii]

                # Calculate the index in c1 and c2
                c1Ind = int(np.round((dfEntry['c1'] - c1Vecs[level][0])/dc1))
                c2Ind = int(np.round((dfEntry['c2'] - c2Vecs[level][0])/dc2))

                # Extract the classification vectors
                Dclass = literal_eval(dfEntry['classify'])[0]
                Pclass = literal_eval(dfEntry['classify'])[1]

                # Now add to the relevant matrix
                if 1 in Dclass:
                    matDmax[level][c2Ind, c1Ind] = 1
                if 2 in Dclass:
                    matDmin[level][c2Ind, c1Ind] = 1
                if 3 in Dclass:
                    matDCSS[level][c2Ind, c1Ind] = 1
                if 4 in Dclass:
                    matDrep[level][c2Ind, c1Ind] = 1
                if 5 in Dclass:
                    matDbra[level][c2Ind, c1Ind] = 1
                if 6 in Dclass:
                    matDext[level][c2Ind, c1Ind] = 1
                if 7 in Dclass:
                    matDpos[level][c2Ind, c1Ind] = 1
                if 6 in Pclass:
                    matPext[level][c2Ind, c1Ind] = 1

            # Calculate the extinction boundaries
            HDDiff = np.c_[matDext[level][:,1:], matDext[level][:,-1]] - matDext[level]
            VDDiff = np.r_[matDext[level][1:,:], np.array([list(matDext[level][-1,:])])] - matDext[level]
            HPDiff = np.c_[matPext[level][:,1:], matPext[level][:,-1]] - matPext[level]
            VPDiff = np.r_[matPext[level][1:,:], np.array([list(matPext[level][-1,:])])] - matPext[level]
            for row in range(nVec[level]):
                for col in range(nVec[level]):
                    if HDDiff[row, col] != 0:
                        DExtBoundc1[level].append([c1Vecs[level][col] + dc1/2, c1Vecs[level][col] + dc1/2])
                        DExtBoundc2[level].append([c2Vecs[level][row] - dc2/2, c2Vecs[level][row] + dc2/2])
                    if VDDiff[row, col] != 0:
                        DExtBoundc1[level].append([c1Vecs[level][col] - dc1/2, c1Vecs[level][col] + dc1/2])
                        DExtBoundc2[level].append([c2Vecs[level][row] + dc2/2, c2Vecs[level][row] + dc2/2])
                    if HPDiff[row, col] != 0:
                        PExtBoundc1[level].append([c1Vecs[level][col] + dc1/2, c1Vecs[level][col] + dc1/2])
                        PExtBoundc2[level].append([c2Vecs[level][row] - dc2/2, c2Vecs[level][row] + dc2/2])
                    if VPDiff[row, col] != 0:
                        PExtBoundc1[level].append([c1Vecs[level][col] - dc1/2, c1Vecs[level][col] + dc1/2])
                        PExtBoundc2[level].append([c2Vecs[level][row] + dc2/2, c2Vecs[level][row] + dc2/2])

        # Plotting
        # We create 1 plot for each level
        figWidth = 40
        figHeight = 20
        gs = mpl.gridspec.GridSpec(nrows=3, ncols=5)
        figs = [plt.figure(figsize=(figWidth, figHeight)) for kk in range(len(levels))]
        axC = [fig.add_subplot(gs[:, 0:3]) for fig in figs]
        axD = [fig.add_subplot(gs[:, 3]) for fig in figs]
        axP = [fig.add_subplot(gs[:, 4]) for fig in figs]
        fsize = 30
        mpl.rcParams['font.size'] = fsize

        # Run an evolutionary simulation for a branching example
        parameters = set_default_ode_parameters()
        parameters['c1'] = 0.05
        parameters['c2'] = -2.75
        parameters['alpha_P'] = compute_ancestral_virulence(parameters)
        parameters['resistance'] = 0.5
        discretized_system = make_discretized_system_coevolution(parameters)

        MIN_VIRULENCE = 0.0
        MAX_VIRULENCE = 1.0
        NUMBER_OF_VIRULENCE_VALUES = 21
        virulence_vector = np.linspace(MIN_VIRULENCE, MAX_VIRULENCE, NUMBER_OF_VIRULENCE_VALUES)

        MIN_RESISTANCE = 0.0
        MAX_RESISTANCE = 1.0
        NUMBER_OF_RESISTANCE_VALUES = 21
        resistance_vector = np.linspace(MIN_RESISTANCE, MAX_RESISTANCE, NUMBER_OF_RESISTANCE_VALUES)

        distance_between_virulence_values = virulence_vector[1] - virulence_vector[0]
        initial_virulence_index = round( (parameters['alpha_P'] - MIN_VIRULENCE)/distance_between_virulence_values )

        distance_between_resistance_values = resistance_vector[1] - resistance_vector[0]
        initial_resistance_index = round( (parameters['resistance'] - MIN_RESISTANCE)/distance_between_resistance_values )

        initial_state = build_initial_state_coevolution(initial_resistance_index, initial_virulence_index, resistance_vector, virulence_vector)
        time_vec = np.arange(2001)
        D, P = coevolutionary_simulation(
            parameters,
            discretized_system,
            initial_state,
            initial_resistance_index,
            initial_virulence_index,
            resistance_vector,
            virulence_vector,
            evolutionary_timesteps=time_vec,
            extinction_threshold=1e-5
        )

        # Now plot each matrix
        for level in levels:
            axC[level].pcolormesh(c1Vecs[level], c2Vecs[level],
                matDext[level], cmap=cmapDExt)
            axC[level].pcolormesh(c1Vecs[level], c2Vecs[level],
                matPext[level], cmap=cmapPExt)
            axC[level].pcolormesh(c1Vecs[level], c2Vecs[level], matDbra[level], cmap=cmapBra)
            for ii in range(len(DExtBoundc1[level])):
                axC[level].plot(DExtBoundc1[level][ii], DExtBoundc2[level][ii], 'k', lw=4)
            for ii in range(len(PExtBoundc1[level])):
                axC[level].plot(PExtBoundc1[level][ii], PExtBoundc2[level][ii], '#444444', lw=4)

        # Add axis labels etc. to each classification figure
        for axind, ax in enumerate(axC):
            ax.set_xlabel(r'Strength of the cost of protection, $c_1$', fontsize=fsize)
            ax.set_ylabel(r'Shape of the trade-off, $c_2$', fontsize=fsize)
            ax.set_xticks([0.0, 1.0])
            ax.set_xticklabels(['0', '1'], fontsize=fsize)
            ax.set_yticks([-5, 0, 5])
            ax.set_yticklabels(['-5', '0', '5'], fontsize=fsize)

            # Add on text for the accelerating and decelerating regions
            ax.text(-0.12, -2.5, 'Decelerating', ha='center', va='center', rotation='vertical', size=fsize)
            ax.text(-0.12, 2.5, 'Accelerating', ha='center', va='center', rotation='vertical', size=fsize)
            ax.text(-0.12, 0, '-----', ha='center', va='center')

        axC[-1].text(0.06, -1.4, 'Symbiont\nbranching', ha='center', va='center', rotation=80, size=48)
        axC[-1].text(0.3, 0.0, 'Possible parasite\nextinction', ha='center', va='center', size=48)
        axC[-1].text(0.92, -0.2, 'Possible\nsymbiont\nextinction', ha='center', va='center', size=48)
        axC[-1].text(0.68, 0.0, 'Guaranteed\ncoexistence', ha='center', va='center', rotation=270, size=48)

        c1Vec = [0.15, 0.75, 0.3, 0.75]
        c2Vec = [4, 4, -4, -4]
        for ii in range(len(c1Vec)):
            axC[-1].plot(c1Vec[ii], c2Vec[ii], 'kx', mew=5, ms=20)

        # Next we plot the evolutionary simulations
        axD[-1].pcolormesh(np.log(resistance_vector+1), time_vec, D, cmap='Greys')
        axP[-1].pcolormesh(np.log(virulence_vector+1), time_vec, P, cmap='Greys')

        # Add an arrow to the right side of the right plot
        axP[-1].arrow(x=1.05, y=0.0, dx=0.0, dy=time_vec[-1]*0.95, width=0.01,
                        color='k', head_width=0.05, 
                        head_length=0.05*time_vec[-1])
        axP[-1].text(1.10, time_vec[-1]/2, 'Evolutionary time', rotation=270,
                        ha = 'center', va = 'center')

        # Add the axes to these plots
        axD[-1].set_xlabel(r'Resistance strength, $y$', fontsize=fsize)
        axD[-1].set_ylabel('')
        axP[-1].set_xlabel(r'Parasite virulene, $\alpha_P$', fontsize=fsize)
        
        axD[-1].set_xticks([MIN_RESISTANCE, MAX_RESISTANCE])
        axD[-1].set_xticklabels([f'{MIN_RESISTANCE:.0f}', f'{MAX_RESISTANCE:.0f}'], fontsize=fsize)
        axP[-1].set_xticks([MIN_VIRULENCE, MAX_VIRULENCE])
        axP[-1].set_xticklabels([f'{MIN_VIRULENCE:.0f}', f'{MAX_VIRULENCE:.0f}'], fontsize=fsize)
        axD[-1].set_yticks([0, max(time_vec)])
        axD[-1].set_yticklabels([])
        axP[-1].set_yticks([0, max(time_vec)])
        axP[-1].set_yticklabels([])
        
        # Remove the surrounding box from evo plots
        axD[-1].spines['top'].set_visible(False)
        axD[-1].spines['right'].set_visible(False)
        axD[-1].spines['left'].set_visible(False)
        axP[-1].spines['top'].set_visible(False)
        axP[-1].spines['right'].set_visible(False)
        axP[-1].spines['left'].set_visible(False)

        # Finally add labels
        axC[-1].text(0, 5.25,'A.', fontsize=fsize)
        axD[-1].text(0, 1.025*max(time_vec), 'B.', fontsize=fsize)
        axP[-1].text(0, 1.025*max(time_vec), 'C.', fontsize=fsize)

        print('D. Blue:  Minimiser')
        print('L. Blue:  Maximiser')
        print('D. Green: CSS')
        print('L. Green: Repeller')
        print('Orange:   Branching')

        # Saving the figure
        saveDir = dataDir.split('/')
        saveDir[1] = 'results'
        saveDir = '/'.join(saveDir)
        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)
        figs[-1].savefig(saveDir + 'classification.png', bbox_inches = 'tight')

    # plt.show()

if __name__ == '__main__':

    # If no task, send a message to the console
    if len(sys.argv) == 1:
        print('Please specify one of the following functions as a second argument: initial, refine or plot')

    # If initialising, give the directory name and number of initial points in
    # each direction (default 21)
    elif sys.argv[1] == 'initial':
        initialClassify(f'data/{sys.argv[2]}', n=int(sys.argv[3]))

    # If refining, give the directory to be refined
    elif sys.argv[1] == 'refine':
        refinementStep(f'data/{sys.argv[2]}')

    # If plotting, give the directory to plot
    elif sys.argv[1] == 'plot':
        plottingClassify(f'data/{sys.argv[2]}')

    # Othereise, throw an error
    else:
        print('Please specify one of the following functions as a second argument: initial, refine or plot')

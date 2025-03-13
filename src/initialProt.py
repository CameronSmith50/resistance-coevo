# Code which will take in a dataset already saved and will conduct an analysis
# on it to determine whether the two host outcomes increase on immediate
# introduction of any protection

#%%

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from modeling import *

#%% Function

def initProt(dataDir):
    
    # Check that the data directory exists
    if not os.path.isdir(f'./data/{dataDir}'):
        print('Data directory does not exist, please try again')
        return

    # Load the dataset
    df = pd.read_csv(f'./data/{dataDir}/dataset.csv')

    # Only run this code if the Q2 and Q3 columns exist
    if 'Q2Init' in df.columns and 'Q3Init' in df.columns:
        pass
    else:

        # Create a vector for the Q1 and Q2 values
        Q2vec = []
        Q3vec = []

        # Initialise the parameters
        pars = set_default_ode_parameters()

        # Loop through the datset and set parameters to the pars dictionary
        for ii, row in tqdm(enumerate(df.iterrows()), leave=False, ncols=50,
                                desc='Ver', total = len(df)):

            pars['b'] = row[1]['b']
            pars['alpha_D'] = row[1]['alphaD']
            pars['beta_hat_D'] = row[1]['betaD']
            pars['beta_hat_P'] = row[1]['betaP']
            pars['gamma_D'] = row[1]['gammaD']
            pars['gamma_P'] = row[1]['gammaP']
            pars['resistance'] = row[1]['y']
            pars['c1'] = row[1]['c1']
            pars['c2'] = row[1]['c2']    
            pars['alpha_P'] = row[1]['alphaPdot']  
            try:
                pars['rho']= row[1]['vert']
            except:
                pars['rho'] = 0.0 
            
            # Run the ODE system and calculate host outcomes
            RHS = make_system(pars)
            H, D, P, B = run_ode_solver(pars, RHS)
            N = H[-1] + D[-1] + P[-1] + B[-1]
            Nancestral = row[1]['Hdot'] + row[1]['Pdot']
            Q2 = N/Nancestral
            Q3num = P[-1] + B[-1]
            Q3den = row[1]['Pdot']
            Q3 = Q3num/Q3den

            # Store to vectors
            Q2vec.append(Q2)
            Q3vec.append(Q3)

        # Store the Q1 and Q2 to new columns
        df['Q2Init'] = Q2vec
        df['Q3Init'] = Q3vec

        # Save the dataframe
        df.to_csv(f'./Data/{dataDir}/dataset.csv', index=False)

#%% Code execution

if __name__ == '__main__':

    # Run the code using the terminal:
    # python initialProt.py PREF-word1-word2-word3
    initProt(sys.argv[1])
    # initProt('v2-SCAT-point-seem-cost')
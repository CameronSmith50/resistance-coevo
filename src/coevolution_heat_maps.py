"""
coevolutionary_heat_maps.py
Algorithm for simulating trait coevolution, and subsequently plotting on host-outcome
measures.

author: Cameron Smith
based from: https://github.com/CameronSmith50/Defensive-Symbiosis/blob/main/Virulence_Coevo.py
"""

from coevolutionary_simulation import *
import pickle as pkl

# Create a function which will input c1 and c2 values, and output files into a folder
def trajectoryHeatMap(
        c1Vals,
        c2Vals
):
    """
    Code which will generate multiple trajectories whcih will be transposed onto host-
    specific heatmaps
    """

    # Create two lists from the c1 and c2 inputs
    c1Vec = np.repeat(c1Vals, 3)
    c2Vec = np.repeat(c2Vals, 3)

    # Vectors for initial protection levels
    yInit = [0.1, 0.5, 0.9]*len(c1Vals)
    # yInit = [0.5]*len(c1Vals)

    # Create the appendage to the save directory
    currDate = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    dataDirAdd = f'traj_{currDate}/'
    dataDir = os.path.abspath(os.path.join(os.path.join(os.path.join(__file__, os.path.pardir, os.path.pardir), 'data/'), dataDirAdd))
    check_for_directory(dataDir)

    # Loop through the pairs of c1 and c2
    for c1, c2, y0 in tqdm(zip(c1Vec, c2Vec, yInit), leave=False, total=len(c1Vec)):

        # Set up a simulation
        ode_parameters = set_default_ode_parameters()
        ode_parameters['c1'] = c1
        ode_parameters['c2'] = c2
        ode_parameters['resistance'] = y0
        discretized_system = make_discretized_system_coevolution(ode_parameters)

        MIN_VIRULENCE = 0.0
        MAX_VIRULENCE = 1.0
        NUMBER_OF_VIRULENCE_VALUES = 51
        virulence_vector = np.linspace(MIN_VIRULENCE, MAX_VIRULENCE, NUMBER_OF_VIRULENCE_VALUES)

        MIN_RESISTANCE = 0.0
        MAX_RESISTANCE = 1.0
        NUMBER_OF_RESISTANCE_VALUES = 51
        resistance_vector = np.linspace(MIN_RESISTANCE, MAX_RESISTANCE, NUMBER_OF_RESISTANCE_VALUES)
        
        ode_parameters['virMin'] = MIN_VIRULENCE
        ode_parameters['virMax'] = MAX_VIRULENCE
        ode_parameters['nVir'] = NUMBER_OF_VIRULENCE_VALUES
        ode_parameters['resMin'] = MIN_RESISTANCE
        ode_parameters['resMax'] = MAX_RESISTANCE
        ode_parameters['nRes'] = NUMBER_OF_RESISTANCE_VALUES

        ancestral_virulence = compute_ancestral_virulence(ode_parameters)
        distance_between_virulence_values = virulence_vector[1] - virulence_vector[0]
        initial_virulence_index = round( (ancestral_virulence - MIN_VIRULENCE)/distance_between_virulence_values )

        distance_between_resistance_values = resistance_vector[1] - resistance_vector[0]
        initial_resistance_index = round( (y0 - MIN_RESISTANCE)/distance_between_resistance_values )

        initial_state = build_initial_state_coevolution(initial_resistance_index, initial_virulence_index, resistance_vector, virulence_vector)

        NUMBER_OF_TIMESTEPS = 1501
        evolutionary_timesteps = np.arange(NUMBER_OF_TIMESTEPS)
        
        extinction_threshold = 1e-5

        dataset1_filename = 'def_symbiont_freqs_from_coevo_sim.csv'
        dataset2_filename = 'parasite_freqs_from_coevo_sim.csv'

        current_date = datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")
        data_directory = os.path.abspath(os.path.join(os.path.join(os.path.join(__file__, os.path.pardir, os.path.pardir), 'data/'), dataDirAdd))
        data_saving_directory = os.path.join(data_directory, current_date)
        check_for_directory(data_saving_directory)

        defensive_symbiont_frequencies, parasite_frequencies = coevolutionary_simulation(
            ode_parameters,
            discretized_system,
            initial_state,
            initial_resistance_index,
            initial_virulence_index,
            resistance_vector,
            virulence_vector,
            evolutionary_timesteps,
            extinction_threshold
        )

        np.savetxt(os.path.join(data_saving_directory, dataset1_filename), defensive_symbiont_frequencies, delimiter=',')
        np.savetxt(os.path.join(data_saving_directory, dataset2_filename), parasite_frequencies, delimiter=',')
        update_lookup_table(dataset1_filename, current_date)
        update_lookup_table(dataset2_filename, current_date)
        savetxt_parameters(ode_parameters, data_saving_directory)

        # Calculate the averages, ensuring branches are calculated properly
        resLower = [y0]
        resUpper = [y0]
        virLower = [ancestral_virulence]
        virUpper = [ancestral_virulence]

        # Loop through time and populate the vectors
        for tInd, t in enumerate(evolutionary_timesteps):
            DD = defensive_symbiont_frequencies[tInd,:]
            PP = parasite_frequencies[tInd,:]

            # Which trait values exist?
            nRes = np.arange(NUMBER_OF_RESISTANCE_VALUES) + 1
            nVir = np.arange(NUMBER_OF_VIRULENCE_VALUES) + 1
            Dpop = DD > 1e-8
            Ppop = PP > 1e-8
            DRemain = (Dpop*nRes)[Dpop*nRes > 0] - 1
            PRemain = (Ppop*nVir)[Ppop*nVir > 0] - 1

            # Check for branching
            if len(DRemain) == 1:
            
                # No branching
                resLower.append(np.sum(resistance_vector*DD))
                resUpper.append(np.sum(resistance_vector*DD))

            elif len(DRemain) == 0:

                # Extinction
                resLower.append(np.nan)
                resUpper.append(np.nan)

            elif np.max(np.diff(DRemain)) <= 2:
                
                # No branching
                resLower.append(np.sum(resistance_vector*DD))
                resUpper.append(np.sum(resistance_vector*DD))

            else:

                # Branching
                brInd = np.diff(DRemain) > 2
                brInd = DRemain[:-1][brInd][0]
                resLower.append(np.sum(resistance_vector[:(brInd+1)]*DD[:(brInd+1)])/np.sum(DD[:(brInd+1)]))
                resUpper.append(np.sum(resistance_vector[(brInd+1):]*DD[(brInd+1):])/np.sum(DD[(brInd+1):]))

            # Check for branching
            if len(PRemain) == 1:
            
                # No branching
                virUpper.append(np.sum(virulence_vector*PP))
                virLower.append(np.sum(virulence_vector*PP))

            elif len(PRemain) == 0:

                # Extinction
                virLower.append(np.nan)
                virUpper.append(np.nan)
                
            elif np.max(np.diff(PRemain)) <= 2:
                
                # No branching
                virLower.append(np.sum(virulence_vector*PP))
                virUpper.append(np.sum(virulence_vector*PP))

            else:

                # Branching
                brInd = np.diff(PRemain) > 2
                brInd = PRemain[:-1][brInd][0]
                virLower.append(np.sum(virulence_vector[:(brInd+1)]*PP[:(brInd+1)])/np.sum(PP[:(brInd+1)]))
                virUpper.append(np.sum(virulence_vector[(brInd+1):]*PP[(brInd+1):])/np.sum(PP[(brInd+1):]))

            # Pickle the vectors
            pklDict = {
                'resLower': resLower,
                'resUpper': resUpper,
                'virLower': virLower,
                'virUpper': virUpper
            }
            file = open(data_saving_directory + '/branches.pkl', 'wb')
            pkl.dump(pklDict, file)
            file.close()

if __name__ == '__main__':

    # Choose your vectors for c1 and c2
    c1Vec = [0.625, 0.45, 0.1, 0.65]
    c2Vec = [-4.5, -2.0, 3.5, 3.0]

    # Run the data
    trajectoryHeatMap(c1Vec, c2Vec)

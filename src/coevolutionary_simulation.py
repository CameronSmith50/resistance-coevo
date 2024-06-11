"""
coevolutionary_simulation.py
Algorithm for simulating trait coevolution.

author: Scott Renegado
based from: https://github.com/CameronSmith50/Defensive-Symbiosis/blob/main/Virulence_Coevo.py
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm

from modeling import set_default_ode_parameters
from modeling import compute_ancestral_virulence
from modeling import make_discretized_system_coevolution
from modeling import run_ode_solver_on_discretized_system_coevolution
from modeling import build_initial_state_coevolution
from lookup_table import check_for_directory
from lookup_table import get_date_from_lookup_table
from lookup_table import update_lookup_table
from lookup_table import savetxt_parameters


def microbe_mutation(state, present_trait_values_bitmask, present_trait_value_indices, number_of_virulence_values, number_of_resistance_values, microbe_densities, microbe=''):
    """
    Mutation routine for either the parasite or defensive symbiont.
    Return the mutated state and updated bitmask.
    """
    if microbe.upper() != 'PARASITE' and microbe.upper() != 'DEFENSIVE SYMBIONT':
        print("Need to specify microbe")
        return state, present_trait_values_bitmask
    last_trait_index = number_of_virulence_values - 1 if microbe.upper() == 'PARASITE' else number_of_resistance_values - 1
    index_shift = 1 + number_of_resistance_values if microbe.upper() == 'PARASITE' else 1

    microbe_densities_sum = np.sum(microbe_densities)
    microbe_densities_sum_times_random_number = microbe_densities_sum * np.random.rand()
    microbe_densities_cumulative_sum = microbe_densities[0]
    microbe_mutator = 0
    while microbe_densities_cumulative_sum < microbe_densities_sum_times_random_number:
        microbe_mutator += 1
        microbe_densities_cumulative_sum += microbe_densities[microbe_mutator]

    if present_trait_value_indices[microbe_mutator] == 0:
        microbe_mutant = 1
    elif present_trait_value_indices[microbe_mutator] == last_trait_index:
        microbe_mutant = last_trait_index - 1
    else:
        microbe_mutant = round(present_trait_value_indices[microbe_mutator] + np.sign(np.random.rand() - 0.5))

    MUTATION = 1e-3
    state[index_shift + microbe_mutant] += MUTATION
    present_trait_values_bitmask[microbe_mutant] = 1
    
    return state, present_trait_values_bitmask


def build_harbouring_both_microbes(
    harbouring_both_present_microbes,
    present_resistance_value_indices,
    present_virulence_value_indices,
    number_of_resistance_values,
    number_of_virulence_values,
    number_of_present_resistance_values,
    number_of_present_virulence_values
):
    """
    Build and return the 'full-sized' harbouring_both_microbes vector from the harbouring_both_present_microbes matrix.
    """
    harbouring_both_microbes = np.zeros((number_of_resistance_values,number_of_virulence_values))

    for row in range(number_of_present_resistance_values):
        for column in range(number_of_present_virulence_values):
            harbouring_both_microbes[present_resistance_value_indices[row], present_virulence_value_indices[column]] = harbouring_both_present_microbes[row, column]

    harbouring_both_microbes = np.reshape(harbouring_both_microbes, (number_of_resistance_values*number_of_virulence_values,))
    return harbouring_both_microbes


def coevolutionary_simulation(
    ode_parameters,
    system,
    initial_state,
    initial_resistance_index,
    initial_virulence_index,
    resistance_vector,
    virulence_vector,
    evolutionary_timesteps,
    extinction_threshold
):
    """
    Perform a numerical simulation where both parasite virulence and symbiont-conferred resistance are evolving.
    Return the defensive symbiont frequencies and parasite frequencies.
    """
    ode_parameters_copy = dict(ode_parameters); 

    number_of_virulence_values = len(virulence_vector)
    virulence_value_shifted_indices = np.arange(number_of_virulence_values) + 1
    present_virulence_values_bitmask = np.zeros(number_of_virulence_values)
    present_virulence_values_bitmask[initial_virulence_index] = 1

    number_of_resistance_values = len(resistance_vector)
    resistance_value_shifted_indices = np.arange(number_of_resistance_values) + 1
    present_resistance_values_bitmask = np.zeros(number_of_resistance_values)
    present_resistance_values_bitmask[initial_resistance_index] = 1

    state = np.array(initial_state)

    number_of_timesteps = len(evolutionary_timesteps)
    parasite_frequencies = np.zeros((number_of_timesteps, number_of_virulence_values))
    defensive_symbiont_frequencies = np.zeros((number_of_timesteps, number_of_resistance_values))

    rows = 0
    columns = 1

    dStop = []
    pStop = []

    for timestep in tqdm(evolutionary_timesteps, leave=False, ncols=50):

        present_virulence_value_shifted_indices = present_virulence_values_bitmask*virulence_value_shifted_indices
        present_virulence_value_indices = np.array(present_virulence_value_shifted_indices[present_virulence_value_shifted_indices > 0], dtype=int) - 1
        present_virulence_values = virulence_vector[present_virulence_value_indices]
        number_of_present_virulence_values = len(present_virulence_value_indices)

        present_resistance_value_shifted_indices = present_resistance_values_bitmask*resistance_value_shifted_indices
        present_resistance_value_indices = np.array(present_resistance_value_shifted_indices[present_resistance_value_shifted_indices > 0], dtype=int) - 1
        present_resistance_values = resistance_vector[present_resistance_value_indices]
        number_of_present_resistance_values = len(present_resistance_value_indices)

        # print(f"Timestep = {timestep}")
        # print(f"Present parasite strains: {present_virulence_values}")
        # print(f"Present defensive symbiont strains: {present_resistance_values}")

        uninfected = state[0]
        harbouring_present_defensive_symbiont = state[1 + present_resistance_value_indices]
        infected_with_present_parasite_strain = state[1 + number_of_resistance_values + present_virulence_value_indices]
        harbouring_both_microbes = state[1 + number_of_resistance_values + number_of_virulence_values:]
        harbouring_both_microbes = np.reshape(harbouring_both_microbes, (number_of_resistance_values,number_of_virulence_values))
        harbouring_both_present_microbes = harbouring_both_microbes[present_resistance_value_indices][:,present_virulence_value_indices]
        harbouring_both_present_microbes = np.reshape(harbouring_both_present_microbes, (number_of_present_resistance_values*number_of_present_virulence_values,))

        initial_state_filtered = np.hstack((uninfected, harbouring_present_defensive_symbiont, infected_with_present_parasite_strain, harbouring_both_present_microbes))

        # Run population dynamics
        solution_time_series = run_ode_solver_on_discretized_system_coevolution(ode_parameters_copy, system, initial_state_filtered, present_virulence_values, present_resistance_values)
        steady_state_approximation = solution_time_series[:, -1]

        uninfected = steady_state_approximation[0]
        harbouring_present_defensive_symbiont = steady_state_approximation[1 : 1 + number_of_present_resistance_values]
        infected_with_present_parasite_strain = steady_state_approximation[1 + number_of_present_resistance_values : 1 + number_of_present_resistance_values + number_of_present_virulence_values]
        harbouring_both_present_microbes = steady_state_approximation[1 + number_of_present_resistance_values + number_of_present_virulence_values:]
        harbouring_both_present_microbes = np.reshape(harbouring_both_present_microbes, (number_of_present_resistance_values,number_of_present_virulence_values))

        # Enact any extinction events
        parasite_densities = infected_with_present_parasite_strain + np.sum(harbouring_both_present_microbes, rows)
        present_virulence_value_renumbered_indices = np.arange(number_of_present_virulence_values) + 1
        extinct_parasite_shifted_indices = (parasite_densities < extinction_threshold)*present_virulence_value_renumbered_indices
        extinct_parasite_indices = extinct_parasite_shifted_indices[extinct_parasite_shifted_indices > 0] - 1

        defensive_symbiont_densities = harbouring_present_defensive_symbiont + np.sum(harbouring_both_present_microbes, columns)
        present_resistance_value_renumbered_indices = np.arange(number_of_present_resistance_values) + 1
        extinct_defensive_symbiont_shifted_indices = (defensive_symbiont_densities < extinction_threshold)*present_resistance_value_renumbered_indices
        extinct_defensive_symbiont_indices = extinct_defensive_symbiont_shifted_indices[extinct_defensive_symbiont_shifted_indices > 0] - 1

        infected_with_present_parasite_strain = np.delete(infected_with_present_parasite_strain, extinct_parasite_indices)
        harbouring_both_present_microbes = np.delete(harbouring_both_present_microbes, extinct_parasite_indices, columns)
        present_virulence_values_bitmask[present_virulence_value_indices[extinct_parasite_indices]] = 0
        parasite_densities = np.delete(parasite_densities, extinct_parasite_indices)
        present_virulence_value_indices = np.delete(present_virulence_value_indices, extinct_parasite_indices)
        number_of_present_virulence_values = len(present_virulence_value_indices)

        harbouring_present_defensive_symbiont = np.delete(harbouring_present_defensive_symbiont, extinct_defensive_symbiont_indices)
        harbouring_both_present_microbes = np.delete(harbouring_both_present_microbes, extinct_defensive_symbiont_indices, rows)
        present_resistance_values_bitmask[present_resistance_value_indices[extinct_defensive_symbiont_indices]] = 0
        defensive_symbiont_densities = np.delete(defensive_symbiont_densities, extinct_defensive_symbiont_indices)
        present_resistance_value_indices = np.delete(present_resistance_value_indices, extinct_defensive_symbiont_indices)
        number_of_present_resistance_values = len(present_resistance_value_indices)

        parasite_driven_extinct = present_virulence_value_indices.size == 0
        defensive_symbiont_driven_extinct = present_resistance_value_indices.size == 0
        if defensive_symbiont_driven_extinct and parasite_driven_extinct: 
            print("Both microbes driven extinct")
            return defensive_symbiont_frequencies, parasite_frequencies

        # Update storage state array and record frequencies
        state[0] = uninfected
        state[1 + present_resistance_value_indices] = harbouring_present_defensive_symbiont
        state[1 + number_of_resistance_values + present_virulence_value_indices] = infected_with_present_parasite_strain
        state[1 + number_of_virulence_values + number_of_resistance_values:] = build_harbouring_both_microbes(
            harbouring_both_present_microbes,
            present_resistance_value_indices,
            present_virulence_value_indices,
            number_of_resistance_values,
            number_of_virulence_values,
            number_of_present_resistance_values,
            number_of_present_virulence_values
        )

        parasite_densities_sum = np.sum(parasite_densities)
        parasite_frequencies[timestep, present_virulence_value_indices] = parasite_densities/parasite_densities_sum

        defensive_symbiont_densities_sum = np.sum(defensive_symbiont_densities)
        defensive_symbiont_frequencies[timestep, present_resistance_value_indices] = defensive_symbiont_densities/defensive_symbiont_densities_sum

        # Mutate the system
        if np.random.rand() < 1/2 and not defensive_symbiont_driven_extinct:
            state, present_resistance_values_bitmask = microbe_mutation(state, present_resistance_values_bitmask, present_resistance_value_indices, number_of_virulence_values, number_of_resistance_values, defensive_symbiont_densities, 'defensive symbiont')
        elif not parasite_driven_extinct:
            state, present_virulence_values_bitmask = microbe_mutation(state, present_virulence_values_bitmask, present_virulence_value_indices, number_of_virulence_values, number_of_resistance_values, parasite_densities, 'parasite')

        # # We want to stop the simulation if neither the parasite nor symbiont have change too much for nStop timesteps
        # nStop = 200
        # if timestep > nStop:
        #     dStop.pop(0)
        #     pStop.pop(0)
        #     dStop.append(0.5*np.sum(np.abs(defensive_symbiont_frequencies[timestep-1,:]-defensive_symbiont_frequencies[timestep,:])))
        #     pStop.append(0.5*np.sum(np.abs(parasite_frequencies[timestep-1,:]-parasite_frequencies[timestep,:])))
        #     stopScore = np.sum(np.array(dStop+pStop))/nStop

        #     # If this stopScore is sufficiently small
        #     if stopScore < 0.013:
        #         for tt in range(timestep, len(evolutionary_timesteps)):
        #             parasite_frequencies[tt, present_virulence_value_indices] = parasite_densities/parasite_densities_sum
        #             defensive_symbiont_frequencies[tt, present_resistance_value_indices] = defensive_symbiont_densities/defensive_symbiont_densities_sum
        #         break

        # else:
        #     dStop.append(0.5*np.sum(np.abs(defensive_symbiont_frequencies[timestep-1,:]-defensive_symbiont_frequencies[timestep,:])))
        #     pStop.append(0.5*np.sum(np.abs(parasite_frequencies[timestep-1,:]-parasite_frequencies[timestep,:])))
        #     stopScore = np.inf

    return defensive_symbiont_frequencies, parasite_frequencies


def main(save_option='', dataDirAdd=''):
    ode_parameters = set_default_ode_parameters()
    ode_parameters['c1'] = 0.3
    ode_parameters['c2'] = 3.5
    initial_resistance_value = 0.8
    ode_parameters['resistance'] = initial_resistance_value 
    discretized_system = make_discretized_system_coevolution(ode_parameters)

    MIN_VIRULENCE = 0.0
    MAX_VIRULENCE = 1.0
    NUMBER_OF_VIRULENCE_VALUES = 51
    virulence_vector = np.linspace(MIN_VIRULENCE, MAX_VIRULENCE, NUMBER_OF_VIRULENCE_VALUES)

    MIN_RESISTANCE = 0.0
    MAX_RESISTANCE = 1.0
    NUMBER_OF_RESISTANCE_VALUES = 51
    resistance_vector = np.linspace(MIN_RESISTANCE, MAX_RESISTANCE, NUMBER_OF_RESISTANCE_VALUES)
    
    ancestral_virulence = compute_ancestral_virulence(ode_parameters)
    distance_between_virulence_values = virulence_vector[1] - virulence_vector[0]
    initial_virulence_index = round( (ancestral_virulence - MIN_VIRULENCE)/distance_between_virulence_values )

    distance_between_resistance_values = resistance_vector[1] - resistance_vector[0]
    initial_resistance_index = round( (initial_resistance_value - MIN_RESISTANCE)/distance_between_resistance_values )

    initial_state = build_initial_state_coevolution(initial_resistance_index, initial_virulence_index, resistance_vector, virulence_vector)

    NUMBER_OF_TIMESTEPS = 1501
    evolutionary_timesteps = np.arange(NUMBER_OF_TIMESTEPS)
    
    extinction_threshold = 1e-5

    dataset1_filename = 'def_symbiont_freqs_from_coevo_sim.csv'
    dataset2_filename = 'parasite_freqs_from_coevo_sim.csv'
    current_date = datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")
    data_directory = os.path.abspath(os.path.join(os.path.join(os.path.join(__file__, os.path.pardir, os.path.pardir), 'data/'), dataDirAdd))

    if save_option.upper() == 'SAVE':
        data_saving_directory = os.path.join(data_directory, current_date)
        check_for_directory(data_saving_directory)

        start_seconds = time.time()
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
        end_seconds = time.time()
        print("Simulation time taken (min): ", (end_seconds - start_seconds)/60)

        np.savetxt(os.path.join(data_saving_directory, dataset1_filename), defensive_symbiont_frequencies, delimiter=',')
        np.savetxt(os.path.join(data_saving_directory, dataset2_filename), parasite_frequencies, delimiter=',')
        update_lookup_table(dataset1_filename, current_date)
        update_lookup_table(dataset2_filename, current_date)
        savetxt_parameters(ode_parameters, data_saving_directory)

    dataset1_date = get_date_from_lookup_table(dataset1_filename)
    dataset2_date = get_date_from_lookup_table(dataset2_filename)
    if dataset1_date is None:
        return print('Dataset is missing from lookup table')
    if dataset2_date is None:
        return print('Dataset is missing from lookup table')
    
    data_loading_directory1 = os.path.join(data_directory, dataset1_date)
    data_loading_directory2 = os.path.join(data_directory, dataset2_date)
    if check_for_directory(data_loading_directory1, make_directory=False) is False:
        return print('Data loading path is either incorrect or missing from filesystem')
    if check_for_directory(data_loading_directory2, make_directory=False) is False:
        return print('Data loading path is either incorrect or missing from filesystem')

    defensive_symbiont_frequencies = np.genfromtxt(os.path.join(data_loading_directory1, dataset1_filename), delimiter=',')
    parasite_frequencies = np.genfromtxt(os.path.join(data_loading_directory1, dataset2_filename), delimiter=',')

    # print("Summing each row of the parasite_frequencies and defensive_symbiont_frequencies matrix...")
    # for timestep in evolutionary_timesteps:
    #     print(f"Timestep = {timestep}: (D) {np.sum(defensive_symbiont_frequencies[timestep])}; (P) {np.sum(parasite_frequencies[timestep])}")

    results_directory = os.path.abspath(os.path.join(os.path.join(__file__, os.path.pardir, os.path.pardir), 'results/'))
    results_saving_directory = os.path.join(results_directory, current_date)
    check_for_directory(results_saving_directory)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey='row')
    ax1.pcolormesh(resistance_vector, evolutionary_timesteps, defensive_symbiont_frequencies, cmap='Greens')
    ax2.pcolormesh(virulence_vector, evolutionary_timesteps, parasite_frequencies, cmap='Reds')
    ax1.set(xlabel=r'resistance $y$')
    ax2.set(xlabel=r'parasite virulence $\alpha_P$')
    ax1.set(ylabel='evolutionary time')
    ax1.set_ylim([0, NUMBER_OF_TIMESTEPS])
    fig.tight_layout()
    fig.savefig(results_saving_directory + '/coevolution_simulation.png')
    plt.show()

    with open(os.path.join(results_saving_directory, 'datasets_used.csv'), 'w') as datasets_used_file:
        datasets_used_file.write(f'{dataset1_filename},{dataset1_date}\n')
        datasets_used_file.write(f'{dataset2_filename},{dataset2_date}\n')

if __name__ == '__main__':
    # save_option_given = len(sys.argv) == 2
    # main(sys.argv[1]) if save_option_given else main()
    main(save_option='save')


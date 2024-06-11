"""
evolutionary_simulation_v2.py
Algorithm for simulating single trait evolution.

author: Scott Renegado
based from: https://github.com/CameronSmith50/Defensive-Symbiosis/blob/main/Virulence_Coevo.py
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from datetime import datetime

from modeling import set_default_ode_parameters
from modeling import compute_ancestral_virulence
from modeling import make_discretized_system
from modeling import run_ode_solver_on_discretized_system
from lookup_table import check_for_directory
from lookup_table import get_date_from_lookup_table
from lookup_table import update_lookup_table
from lookup_table import savetxt_parameters


def evolutionary_simulation_v2(ode_parameters, system, initial_state, initial_virulence_index, virulence_vector, evolutionary_timesteps, extinction_threshold):
    """
    Perform a numerical simulation where parasite virulence is evolving.
    Return the parasite frequencies.
    """
    ode_parameters_copy = dict(ode_parameters); 

    number_of_virulence_values = len(virulence_vector)
    virulence_value_shifted_indices = np.arange(number_of_virulence_values) + 1
    present_virulence_values_bitmask = np.zeros(number_of_virulence_values)
    present_virulence_values_bitmask[initial_virulence_index] = 1

    state = np.array(initial_state)

    number_of_timesteps = len(evolutionary_timesteps)
    parasite_frequencies = np.zeros((number_of_timesteps, number_of_virulence_values))

    pStop = []

    for timestep in evolutionary_timesteps:

        present_virulence_value_shifted_indices = present_virulence_values_bitmask*virulence_value_shifted_indices
        present_virulence_value_indices = np.array(present_virulence_value_shifted_indices[present_virulence_value_shifted_indices > 0], dtype=int) - 1
        present_virulence_values = virulence_vector[present_virulence_value_indices]
        number_of_present_virulence_values = len(present_virulence_values)

        # print(f"Timestep = {timestep}. Present parasite strains: {present_virulence_values}")
        
        uninfected = state[0]
        harbouring_defensive_symbiont = state[1]
        infected_with_present_parasite_strain = state[2 + present_virulence_value_indices]
        harbouring_both_microbes = state[2 + number_of_virulence_values + present_virulence_value_indices]
        initial_state_filtered = np.hstack((uninfected, harbouring_defensive_symbiont, infected_with_present_parasite_strain, harbouring_both_microbes))

        # Run population dynamics
        solution_time_series = run_ode_solver_on_discretized_system(ode_parameters_copy, system, initial_state_filtered, present_virulence_values)
        steady_state_approximation = solution_time_series[:, -1]

        uninfected = steady_state_approximation[0]
        harbouring_defensive_symbiont = steady_state_approximation[1]
        infected_with_present_parasite_strain = steady_state_approximation[2:number_of_present_virulence_values + 2]
        harbouring_both_microbes = steady_state_approximation[number_of_present_virulence_values + 2:]

        # Enact any extinction events
        parasite_densities = infected_with_present_parasite_strain + harbouring_both_microbes
        present_virulence_value_renumbered_indices = np.arange(number_of_present_virulence_values) + 1
        extinct_parasite_shifted_indices = (parasite_densities < extinction_threshold)*present_virulence_value_renumbered_indices
        extinct_parasite_indices = extinct_parasite_shifted_indices[extinct_parasite_shifted_indices > 0] - 1

        infected_with_present_parasite_strain = np.delete(infected_with_present_parasite_strain, extinct_parasite_indices)
        harbouring_both_microbes = np.delete(harbouring_both_microbes, extinct_parasite_indices)
        present_virulence_values_bitmask[present_virulence_value_indices[extinct_parasite_indices]] = 0
        parasite_densities = np.delete(parasite_densities, extinct_parasite_indices)
        present_virulence_value_indices = np.delete(present_virulence_value_indices, extinct_parasite_indices)

        parasite_driven_extinct = present_virulence_value_indices.size == 0
        if parasite_driven_extinct: 
            # print("Parasite driven extinct")
            return parasite_frequencies

        # Update storage state array and record frequencies
        state[0] = uninfected
        state[1] = harbouring_defensive_symbiont
        state[2 + present_virulence_value_indices] = infected_with_present_parasite_strain
        state[2 + number_of_virulence_values + present_virulence_value_indices] = harbouring_both_microbes

        parasite_densities_sum = np.sum(parasite_densities)
        parasite_frequencies[timestep, present_virulence_value_indices] = parasite_densities/parasite_densities_sum

        # Mutate the system
        parasite_densities_sum_times_random_number = parasite_densities_sum * np.random.rand()
        parasite_densities_cumulative_sum = parasite_densities[0]
        parasite_mutator = 0
        while parasite_densities_cumulative_sum < parasite_densities_sum_times_random_number:
            parasite_mutator += 1
            parasite_densities_cumulative_sum += parasite_densities[parasite_mutator]

        if present_virulence_value_indices[parasite_mutator] == 0:
            parasite_mutant = 1
        elif present_virulence_value_indices[parasite_mutator] == number_of_virulence_values - 1:
            parasite_mutant = number_of_virulence_values - 2
        else:
            parasite_mutant = round(present_virulence_value_indices[parasite_mutator] + np.sign(np.random.rand() - 0.5))

        MUTATION = 1e-3
        state[2 + parasite_mutant] += MUTATION
        present_virulence_values_bitmask[parasite_mutant] = 1

        # We want to stop the simulation if neither the parasite nor symbiont have change too much for nStop timesteps
        nStop = 50
        if timestep > nStop:
            pStop.pop(0)
            pStop.append(0.5*np.sum(np.abs(parasite_frequencies[timestep-1,:]-parasite_frequencies[timestep,:])))
            stopScore = np.sum(np.array(pStop))/nStop

            # If this stopScore is sufficiently small
            if stopScore < 0.015:
                for tt in range(timestep, len(evolutionary_timesteps)):
                    parasite_frequencies[tt, present_virulence_value_indices] = parasite_densities/parasite_densities_sum
                break

        else:
            pStop.append(0.5*np.sum(np.abs(parasite_frequencies[timestep-1,:]-parasite_frequencies[timestep,:])))
            stopScore = np.inf

    return parasite_frequencies


def main(save_option=''):
    ode_parameters = set_default_ode_parameters()
    ode_parameters['c1'] = 0.2
    ode_parameters['resistance'] = 0.71
    discretized_system = make_discretized_system(ode_parameters)

    MIN_VIRULENCE = 0.1
    MAX_VIRULENCE = 1.0
    NUMBER_OF_VIRULENCE_VALUES = 50
    virulence_vector = np.linspace(MIN_VIRULENCE, MAX_VIRULENCE, NUMBER_OF_VIRULENCE_VALUES)
    
    ancestral_virulence = compute_ancestral_virulence(ode_parameters)
    distance_between_virulence_values = virulence_vector[1] - virulence_vector[0]
    initial_virulence_index = round( (ancestral_virulence - MIN_VIRULENCE)/distance_between_virulence_values )

    H = np.array([8])
    D = np.array([1])
    P = np.zeros(NUMBER_OF_VIRULENCE_VALUES); P[initial_virulence_index] = 1
    B = np.zeros(NUMBER_OF_VIRULENCE_VALUES)
    initial_state = np.hstack((H, D, P, B))

    NUMBER_OF_TIMESTEPS = 1000
    evolutionary_timesteps = np.arange(NUMBER_OF_TIMESTEPS)
    
    extinction_threshold = 1e-5

    dataset_filename = 'parasite_frequencies_from_single_evolution_simulation.csv'
    current_date = datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")
    data_directory = os.path.abspath(os.path.join(os.path.join(__file__, os.path.pardir, os.path.pardir), 'data/'))

    if save_option.upper() == 'SAVE':
        data_saving_directory = os.path.join(data_directory, current_date)
        check_for_directory(data_saving_directory)

        start_seconds = time.time()
        parasite_frequencies = evolutionary_simulation_v2(
            ode_parameters,
            discretized_system,
            initial_state,
            initial_virulence_index,
            virulence_vector,
            evolutionary_timesteps,
            extinction_threshold
        )
        end_seconds = time.time()
        print("Simulation time taken (min): ", (end_seconds - start_seconds)/60)

        np.savetxt(data_saving_directory + '/' + dataset_filename, parasite_frequencies, delimiter=',')
        update_lookup_table(dataset_filename, current_date)
        savetxt_parameters(ode_parameters, data_saving_directory)

    dataset_date = get_date_from_lookup_table(dataset_filename)
    if dataset_date is None:
        return print('Dataset missing')
    
    data_loading_directory = os.path.join(data_directory, dataset_date)
    if check_for_directory(data_loading_directory, make_directory=False) is False:
        return print('Data loading path is incorrect or missing from filesystem')
    
    parasite_frequencies = np.genfromtxt(data_loading_directory + '/' + dataset_filename, delimiter=',')

    # print("Summing each row of the parasite_frequencies matrix...")
    # for timestep in evolutionary_timesteps:
    #     print(f"Timestep = {timestep}: {np.sum(parasite_frequencies[timestep])}")

    results_directory = os.path.abspath(os.path.join(os.path.join(__file__, os.path.pardir, os.path.pardir), 'results/'))
    results_saving_directory = os.path.join(results_directory, current_date)
    check_for_directory(results_saving_directory)

    FIGURE_WIDTH_INCHES = 3.5
    plt.figure().set_figwidth(FIGURE_WIDTH_INCHES)
    plt.pcolormesh(virulence_vector, evolutionary_timesteps, parasite_frequencies, cmap='Greys')
    plt.xlabel(r'parasite virulence $\alpha_P$')
    plt.ylabel('evolutionary time')
    plt.ylim([0, NUMBER_OF_TIMESTEPS])
    plt.tight_layout()
    plt.savefig(results_saving_directory + '/evolution_simulation.png')
    
    with open(os.path.join(results_saving_directory, 'datasets_used.csv'), 'w') as datasets_used_file:
        datasets_used_file.write(f'{dataset_filename},{dataset_date}\n')

if __name__ == '__main__':
    save_option_given = len(sys.argv) == 2
    run_main = main(sys.argv[1]) if save_option_given else main()
    run_main
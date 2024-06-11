"""
lookup_table.py
Helper functions for manipulating the lookup table or filesystem.

author: Scott Renegado
"""

import pandas as pd
import os


def get_date_from_lookup_table(dataset):
    """
    Get date string (the convention here is either 'YYYY-MM-DD' or 'YYYY-MM-DD HHhmmmsss') 
    for given dataset in lookup table.
    If dataset does not exist, return None.
    """
    data_directory = os.path.abspath(os.path.join(os.path.join(__file__, os.path.pardir, os.path.pardir), 'data/'))
    lookup_table = pd.read_csv(data_directory + '/lookup_table.csv')

    filter = lookup_table['dataset'] == dataset
    lookup_table_filtered = lookup_table[filter]

    if lookup_table_filtered.empty:
        return
    
    date = lookup_table_filtered['date'].item()
    return date


def update_lookup_table(dataset, date):
    """
    Add a new row to the lookup table.
    If a row with the dataset exists, then update the associated date entry instead.
    """
    data_directory = os.path.abspath(os.path.join(os.path.join(__file__, os.path.pardir, os.path.pardir), 'data/'))
    lookup_table = pd.read_csv(data_directory + '/lookup_table.csv')

    filter = lookup_table['dataset'] == dataset
    lookup_table_filtered = lookup_table[filter]

    if not lookup_table_filtered.empty:
        lookup_table.loc[filter, 'date'] = date
        lookup_table.to_csv(data_directory + '/lookup_table.csv', index=False)
        return
    
    new_row_data = {'dataset': [dataset], 'date': date}
    new_row = pd.DataFrame(data=new_row_data)
    lookup_table = pd.concat([lookup_table, new_row], ignore_index=True)
    lookup_table.to_csv(data_directory + '/lookup_table.csv', index=False)
    return


def check_for_directory(directory, make_directory=True):
    """
    Check if directory is present and return Boolean value.
    If make_directory is True and directory is not present, create it.
    """
    directory_is_present = os.path.isdir(directory)
    if make_directory and not directory_is_present:
        os.mkdir(directory)
        return False
    elif not directory_is_present:
        return False
    return True


def savetxt_parameters(parameters, data_saving_directory):
    """Save the given parameters as a text file"""
    with open(os.path.join(data_saving_directory, 'parameters.txt'), 'w') as parameter_file:
        parameter_file.write('Parameter list\n')
        for parameter, parameter_value in parameters.items():
            parameter_file.write(f'{parameter}={parameter_value}\n')
    

def test():
    print("TEST: update_lookup_table")
    update_lookup_table('test.csv', '2023-XX-XX ')
    update_lookup_table('test.csv', '2023-ZZ-ZZ')
    

if __name__ == '__main__':
    test()

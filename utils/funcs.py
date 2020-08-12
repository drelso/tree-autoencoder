###
#
# Miscellaneous utility functions
#
###

import os
import torch

def dir_validation(dir_path):
    '''
    Model directory housekeeping: make sure
    directory exists, if not create it and make
    sure the path to the directory ends with '/'
    to allow correct path concatenation

    Requirements
    ------------
    import os

    Parameters
    ----------
    dir_path : str
        directory path to validate or correct
    
    Returns
    -------
    str
        validated directory path
    '''
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    if not dir_path.endswith('/'): dir_path += '/'
    return dir_path


def print_parameters(parameters):
    '''
    Pretty print all model parameters

    Parameters
    ----------
    parameters : {str : X }
        parameter dictionary, where the keys are
        the parameter names with their corresponding
        values
    '''
    
    # PRINT PARAMETERS
    print('\n=================== MODEL PARAMETERS: =================== \n')
    for name, value in parameters.items():
        # num_tabs = int((32 - len(name))/8) + 1
        # tabs = '\t' * num_tabs
        num_spaces = 30 - len(name)
        spaces = ' ' * num_spaces
        print(f'{name}: {spaces} {value}')
    print('\n=================== / MODEL PARAMETERS: =================== \n')


def memory_stats(device=torch.device('cpu')):
    '''
    Memory usage for a specific device
    (ONLY WRITTEN FOR GPU MEMORY)
    TODO: implement for CPU

    Parameters
    ----------
    device : torch.device, optional
        the torch device to track memory for
        (default: torch.device('cpu'))
    '''
    conversion_rate = 2**30 # CONVERT TO GB
    # print('\n +++++++++++ torch.cuda.memory_stats\n')
    # print(torch.cuda.memory_stats(device=device))
    
    print('\n +++++++++++ torch.cuda.memory_summary\n')
    print(torch.cuda.memory_summary(device=device))
    
    # print('\n +++++++++++ torch.cuda.memory_snapshot\n')
    # print(torch.cuda.memory_snapshot())

    print('\n\n +++++++++++ torch.cuda.memory_allocated\n')
    print((torch.cuda.memory_allocated(device=device)/conversion_rate), 'GB')
    print('\n\n +++++++++++ torch.cuda.max_memory_allocated\n')
    print((torch.cuda.max_memory_allocated(device=device)/conversion_rate), 'GB')
    print('\n\n +++++++++++ torch.cuda.memory_reserved\n')
    print((torch.cuda.memory_reserved(device=device)/conversion_rate), 'GB')
    print('\n\n +++++++++++ torch.cuda.max_memory_reserved\n')
    print((torch.cuda.max_memory_reserved(device=device)/conversion_rate), 'GB')
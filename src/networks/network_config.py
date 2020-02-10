import argparse

# This function is to parse strings from argparse into bool
def str2bool(v):
    '''Convert string to boolean value

    This function is from stackoverflow:
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    Arguments:
        v {str} -- [description]

    Returns:
        bool -- [description]

    Raises:
        argparse -- [description]
    '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


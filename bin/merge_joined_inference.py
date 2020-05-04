import argparse
import pathlib

import tables
import pandas
import numpy


parser = argparse.ArgumentParser()


parser = argparse.ArgumentParser(
    description     = 'Combine merged dataframes into just one dataframe',
    formatter_class = argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--input-files',
    type    = pathlib.Path,
    default = "",
    nargs   = "+",
    help    = 'The inference dataframe files.')

parser.add_argument('--output-file',
    type    = pathlib.Path,
    default = "",
    help    = 'The merged dataframe file')

args = parser.parse_args()

opened_dataframes = [ pandas.read_hdf(f) for f in args.input_files ]

merged_df = pandas.concat(opened_dataframes)

merged_df.to_hdf(str(args.output_file), key="Merged_inference_dataframe")

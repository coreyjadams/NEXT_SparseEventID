import sys, os
import time

from rl_to_larcv import convert_entry_point

from multiprocessing import Process, Queue


# How many files are there to process?  A lot.
n_files = 25019
# n_files = 200

TOP_FOLDER = "/lus/grand/projects/datascience/cadams/datasets/NEXT/officialProduction/"

RL_OUTPUT       = f"{TOP_FOLDER}/Background/NEXT_v1_05_02_NEXUS_v5_07_10_bkg_v9/dhits_filtered/output/"
PMAPS_OUTPUT    = f"{TOP_FOLDER}/Background/NEXT_v1_05_02_NEXUS_v5_07_10_bkg_v9/pmaps/output/"
DDST_OUTPUT     = f"{TOP_FOLDER}/Background/NEXT_v1_05_02_NEXUS_v5_07_10_bkg_v9/ddst/output/"
LARCV_OUTPUT    = f"{TOP_FOLDER}/Background/NEXT_v1_05_02_NEXUS_v5_07_10_bkg_v9/larcv/output/"

RL_FILE_TEMPLATE    = "bkg_NEW_v1.2.0_v9.dhits_{i_file}.filtered_wMC.h5"
PMAPS_FILE_TEMPLATE = "bkg_NEW_v1.2.0_v9.pmaps_{i_file}.h5"
DDST_FILE_TEMPLATE  = "bkg_NEW_v1.2.0_v9.ddst_{i_file}.h5"
LARCV_FILE_TEMPLATE = "bkg_NEW_v1.2.0_v9.larcv_{i_file}.h5"

DB_FILE             = "/home/cadams/Polaris/NEXT_SparseEventID/database/new_sipm.pkl"

max_processes = 128
p_list = []

for i_file in range(n_files):
    # i_file = 1000 + i_file
    input_files = [
        RL_OUTPUT    + RL_FILE_TEMPLATE.format(    i_file = i_file),
        PMAPS_OUTPUT + PMAPS_FILE_TEMPLATE.format( i_file = i_file),
        DDST_OUTPUT  + DDST_FILE_TEMPLATE.format(  i_file = i_file),
    ]
    output_file = LARCV_OUTPUT + LARCV_FILE_TEMPLATE.format( i_file = i_file)
    # print(input_files)
    print(i_file)
    # Wait for an open process:
    while len(p_list) >= max_processes:
        # Close empty processes:
        for p in p_list:
            p.join(timeout=0.01)
            if p.exitcode is not None:
                p_list.remove(p)
        time.sleep(0.5)

    # Construct the process:
    p = Process(target = convert_entry_point, args=(input_files, output_file, DB_FILE))
    p.start()
    p_list.append(p)

for p in p_list:
    p.join()

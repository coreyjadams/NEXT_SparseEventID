
import glob
import os
import pathlib
from random import shuffle


import numpy as np
import pandas as pd
import tables as tb

import larcv

# top_input_path = "/gpfs/alpine/proj-shared/nph133/nextnew/nextnew_Tl208_hdf/"
# output_path = "/gpfs/alpine/proj-shared/nph133/nextnew/larcv_datafiles/"

top_input_path  = pathlib.Path("/Users/corey.adams/data/NEXT/mmkekic_second_production/data/7470_cdst/")
output_path     = pathlib.Path("/Users/corey.adams/data/NEXT/mmkekic_second_production/data/7470_larcv/")


def main():


    next_new_meta = larcv.ImageMeta3D()
    next_new_meta.set_dimension(0, 510, 51, -205)
    next_new_meta.set_dimension(1, 510, 51, -205)
    next_new_meta.set_dimension(2, 540, 108, 0)
    #next_new_meta.set_dimension(0, 2600, 260, -1300)
    #next_new_meta.set_dimension(1, 2600, 260, -1300)
    #next_new_meta.set_dimension(2, 2600, 260, -1300)

    # # This code loops over training set files:
    file_list = list(top_input_path.glob("*.h5"))

    output_path.mkdir(exist_ok=True)


    print('Found %s input data files'%len(file_list))


    for i, f in enumerate(file_list):

        file_name = f.name
        output_file = output_path / file_name.replace(".root.h5", "_larcv.h5")

        _, run, subrun = file_name.replace(".h5","").split("_")


        print(file_name)


        if os.path.exists(output_file):
            continue

        # output_trn = os.path.basename('NextNEW_Tl208_10mm_larcv_noshf_train_200k.h5')
        io_manager = larcv.IOManager(larcv.IOManager.kWRITE)
        io_manager.set_out_file(str(output_file))
        io_manager.initialize()
        # convert train files
        print(f'Converting file {i}: {file_name}')
        convert_files(io_manager, next_new_meta, f, run, subrun)
        io_manager.finalize()

        break
    # output_tst = os.path.basename('NextNEW_Tl208_10mm_larcv_noshf_test_200k.h5')
    # output_tst = output_path + "/" + output_tst
    # io_manager_tst = larcv.IOManager(larcv.IOManager.kWRITE)
    # io_manager_tst.set_out_file(output_tst)
    # io_manager_tst.initialize()
    # # convert test files
    # print('Converting test files')
    # convert_files(io_manager_tst, next_new_meta, test_list)
    # io_manager_tst.finalize()

# @profile
def convert_files( io_manager, next_new_meta, fname, run, subrun):

    evtfile = tb.open_file(str(fname), 'r')
    events = evtfile.root.Run.events.read()

    event_numbers = events['evt_number']

    convert_low_th = False

    # Instead of slicing and dicing later, we read everything into memory up front:
    high_threshold_voxels = evtfile.root.CHITS.highTh.read()
    
    if convert_low_th:
        low_threshold_voxels = evtfile.root.CHITS.lowTh.read()

    n_events = len(event_numbers)

    for ievt,event in enumerate(event_numbers):
        
        if ievt % 10 == 0:
            print(f"Beginning entry {ievt} of {n_events} which is event {event}")


        io_manager.set_id(int(run), int(subrun), event)




        ################################################################################
        # Store the highTh info:
        event_sparse3d = io_manager.get_data("sparse3d", "voxels")
        st = larcv.SparseTensor3D()
        st.meta(next_new_meta)

        voxel_idcs = high_threshold_voxels['event'] == event
        voxels = high_threshold_voxels[voxel_idcs]

        # Find all the NaNs:
        weights = voxels['Ec']
        is_nan_array = np.isnan(weights)

        # Extract the positions:
        position_array = voxels[['X','Y','Z']]


        # get the index array:
        index = [ next_new_meta.position_to_index(p) for p in position_array ]

        _ = [st.emplace(larcv.Voxel(index[i], weights[i]), True) for i in range(len(index)) if not is_nan_array[i]]

        event_sparse3d.set(st)
        ################################################################################


        if convert_low_th:
            ################################################################################
            # Store the lowTh info:
            event_sparse3d = io_manager.get_data("sparse3d", "voxels_low")
            st = larcv.SparseTensor3D()
            st.meta(next_new_meta)

            voxel_idcs = low_threshold_voxels['event'] == event
            voxels = low_threshold_voxels[voxel_idcs]

            # Find all the NaNs:
            weights = voxels['Ec']
            is_nan_array = np.isnan(weights)

            # Extract the positions:
            position_array = voxels[['X','Y','Z']]


            # get the index array:
            index = [ next_new_meta.position_to_index(p) for p in position_array ]

            _ = [st.emplace(larcv.Voxel(index[i], weights[i]), True) for i in range(len(index)) if not is_nan_array[i]]

            event_sparse3d.set(st)
            ################################################################################


        io_manager.save_entry()

        if ievt > 1000:
            break

    evtfile.close()

    return


if __name__ == '__main__':
    main()
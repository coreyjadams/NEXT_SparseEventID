import larcv

import glob
import os
import numpy as np
import pandas as pd
import tables as tb
from random import shuffle

# top_input_path = "/gpfs/alpine/proj-shared/nph133/nextnew/nextnew_Tl208_hdf/"
# output_path = "/gpfs/alpine/proj-shared/nph133/nextnew/larcv_datafiles/"

top_input_path = "/Users/corey.adams/data/NEXT/mmkekic_second_production/cdst/"
output_path = "/Users/corey.adams/data/NEXT/mmkekic_second_production/larcv/"


def main():


    next_new_meta = larcv.ImageMeta3D()
    next_new_meta.set_dimension(0, 510, 51, -205)
    next_new_meta.set_dimension(1, 510, 51, -205)
    next_new_meta.set_dimension(2, 540, 108, 0)
    #next_new_meta.set_dimension(0, 2600, 260, -1300)
    #next_new_meta.set_dimension(1, 2600, 260, -1300)
    #next_new_meta.set_dimension(2, 2600, 260, -1300)

    next_new_meta_mc = larcv.ImageMeta3D()
    next_new_meta_mc.set_dimension(0, 510, 510, -205)
    next_new_meta_mc.set_dimension(1, 510, 510, -205)
    next_new_meta_mc.set_dimension(2, 540, 540, 0)
    # # This code loops over training set files:
    file_list = "/Users/corey.adams/data/NEXT/mmkekic_second_production/all_labels.cvs"

    # read in list of events
    df = pd.read_csv(file_list)


    groups = list(df.groupby('filename'))

    # Prune off the directory:
    # groups = [ os.path.basename(g) for g in groups]


    # split between test and train files
    nfiles = len(groups)
    ntrain = int(nfiles*0.8)
    # train_list = groups[:ntrain]
    # test_list = groups[ntrain:]

    # print('Found %s input training files'%len(train_list))
    # print('Found %s input testing files'%len(test_list))


    for i, f in enumerate(groups):

        file_name = os.path.basename(f[0])
        output_file = output_path + file_name.replace(".root.h5", "_larcv.h5")

        if os.path.exists(output_file):
            continue

        # output_trn = os.path.basename('NextNEW_Tl208_10mm_larcv_noshf_train_200k.h5')
        io_manager = larcv.IOManager(larcv.IOManager.kWRITE)
        io_manager.set_out_file(output_file)
        io_manager.initialize()
        # convert train files
        print(f'Converting file {i}: {file_name}')
        convert_files(io_manager, [next_new_meta, next_new_meta_mc], (i, f))
        io_manager.finalize()

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
def convert_files( io_manager, next_new_meta, file_groups, convert_mc_hits = False ):


    # for fidc, fname in enumerate(file_groups):
    fidc, (fname, fgroup) = file_groups
    # fname = fname.split('/')[-1]

    fname = top_input_path + os.path.basename(fname)

    # try:
    evtfile = tb.open_file(fname, 'r')
    # print(evtfile)
    # except Exception:
    #     continue

    # Instead of slicing and dicing later, we read everything into memory up front:
    high_threshold_voxels = evtfile.root.CHITS.highTh.read()
    low_threshold_voxels = evtfile.root.CHITS.lowTh.read()

    if convert_mc_hits:
        mc_extents = evtfile.root.MC.extents.read()
        mc_hits_voxels = evtfile.root.MC.hits.read()
        
        mc_hit_first_index = 0

    run = 0
    for ievt,event in fgroup.iterrows():
        label = 0
        if event['positron'] == 1.0 and event['E_add'] == 0.0:
            label = 1

        previous_event = event['event']

        io_manager.set_id(run, fidc, int(event['event']))

        ################################################################################
        # Store the particle information:
        larcv_particle = io_manager.get_data("particle", "label")
        particle = larcv.Particle()
        particle.energy_init(0.)
        particle.pdg_code(label)
        larcv_particle.append(particle)
        ################################################################################
        # Store the voxel information:



        ################################################################################
        # Store the highTh info:
        event_sparse3d = io_manager.get_data("sparse3d", "voxels")
        st = larcv.SparseTensor3D()
        st.meta(next_new_meta[0])

        voxel_idcs = high_threshold_voxels['event'] == event['event']
        voxels = high_threshold_voxels[voxel_idcs]

        # Find all the NaNs:
        weights = voxels['Ec']
        is_nan_array = np.isnan(weights)

        # Extract the positions:
        position_array = voxels[['X','Y','Z']]


        # get the index array:
        index = [ next_new_meta[0].position_to_index(p) for p in position_array ]

        _ = [st.emplace(larcv.Voxel(index[i], weights[i]), True) for i in range(len(index)) if not is_nan_array[i]]

        event_sparse3d.set(st)
        ################################################################################



        ################################################################################
        # Store the lowTh info:
        event_sparse3d = io_manager.get_data("sparse3d", "voxels_low")
        st = larcv.SparseTensor3D()
        st.meta(next_new_meta[0])

        voxel_idcs = low_threshold_voxels['event'] == event['event']
        voxels = low_threshold_voxels[voxel_idcs]

        # Find all the NaNs:
        weights = voxels['Ec']
        is_nan_array = np.isnan(weights)

        # Extract the positions:
        position_array = voxels[['X','Y','Z']]


        # get the index array:
        index = [ next_new_meta[0].position_to_index(p) for p in position_array ]

        _ = [st.emplace(larcv.Voxel(index[i], weights[i]), True) for i in range(len(index)) if not is_nan_array[i]]

        event_sparse3d.set(st)
        ################################################################################


        if convert_mc_hits:
            ################################################################################
            # Store the mchit info:
            event_sparse3d = io_manager.get_data("sparse3d", "mchit")
            st = larcv.SparseTensor3D()
            st.meta(next_new_meta[1])


            if event['event'] >= len(mc_extents):
                break

            mc_hit_last_index = mc_extents[event['event']]['last_hit']

            mc_hits = mc_hits_voxels[mc_hit_first_index:mc_hit_last_index]
            mc_positions = mc_hits['hit_position']
            mc_energy    = mc_hits['hit_energy']

            mc_hit_first_index = mc_hit_last_index


            # Find all the NaNs:
            is_nan_array = np.isnan(mc_energy)

            # get the index array:
            index = [ next_new_meta[1].position_to_index(p) for p in mc_positions ]

            _ = [st.emplace(larcv.Voxel(index[i], mc_energy[i]), True) for i in range(len(index)) if not is_nan_array[i]]

            event_sparse3d.set(st)
            ################################################################################


        io_manager.save_entry()

    evtfile.close()

    return


if __name__ == '__main__':
    main()
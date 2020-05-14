import glob
import os
import pathlib
from random import shuffle
import argparse

import numpy as np
import tables as tb

import larcv

# top_input_path = "/gpfs/alpine/proj-shared/nph133/nextnew/nextnew_Tl208_hdf/"
# output_path = "/gpfs/alpine/proj-shared/nph133/nextnew/larcv_datafiles/"

top_input_path  = pathlib.Path("/lus/theta-fs0/projects/datascience/cadams/datasets/NEXT/new_second_simulation/data/7372_cdst/")
output_path     = pathlib.Path("/lus/theta-fs0/projects/datascience/cadams/datasets/NEXT/new_second_simulation/data/7372_larcv/")


def main(args):


    # Create the meta object:
    next_new_meta = larcv.ImageMeta3D()
    next_new_meta.set_dimension(0, 510, 51, -205)
    next_new_meta.set_dimension(1, 510, 51, -205)
    next_new_meta.set_dimension(2, 540, 108, 0)


    # Next, make sure the input file exists:
    args.input_file = pathlib.Path(args.input_file)
    args.output_file = pathlib.Path(args.output_file)

    if not args.input_file.exists():
        raise Exception("Input file doesn't exist!")

    # Make sure the output directory exists:
    args.output_file.parent.mkdir(exist_ok=True)

    # # This code loops over training set files:
    file_list = list(top_input_path.glob("*.h5"))


    # Now, begin conversion:

    file_name = args.input_file.name

    file_name = args.input_file.name

    _, run, subrun = file_name.replace("_v1.2.0_trigger2_bg", "").replace(".h5","").split("_")

    _, run, subrun = file_name.replace(".h5","").split("_")

    # output_trn = os.path.basename('NextNEW_Tl208_10mm_larcv_noshf_train_200k.h5')
    io_manager = larcv.IOManager(larcv.IOManager.kWRITE)
    io_manager.set_out_file(str(args.output_file))
    io_manager.initialize()
    # convert train files
    print(f'Converting file: {file_name} from {args.start_entry} to {args.end_entry}')
    convert_file(io_manager, next_new_meta, args.input_file, run, subrun, args.start_entry, args.end_entry)
    io_manager.finalize()

# @profile

# @profile
def convert_file(io_manager, next_new_meta, fname, run, subrun, start_entry, end_entry):

    evtfile = tb.open_file(str(fname), 'r')


    events = evtfile.root.Run.events.read()

    tracks = evtfile.root.Tracking.Tracks.read()
    summary = evtfile.root.Summary.Events.read()


    event_numbers = events['evt_number']
    event_energy  = summary['evt_energy']


    convert_low_th = True

    # Instead of slicing and dicing later, we read everything into memory up front:
    high_threshold_voxels = evtfile.root.CHITS.highTh.read()

    if convert_low_th:
        low_threshold_voxels = evtfile.root.CHITS.lowTh.read()

    n_events = len(event_numbers)

    # Only loop over the needed entries:
    for ievt in range(start_entry, end_entry):
        if ievt >= len(event_energy): continue

        event  = event_numbers[ievt]
        energy = event_energy[ievt]

        if energy < 1.0 or energy > 2.0:
            continue

        if ievt % 10 == 0:
            print(f"Beginning entry {ievt} of {n_events} which is event {event}")


        io_manager.set_id(int(run), int(subrun), event)



        ################################################################################
        # Store the particle information:
        larcv_particle = io_manager.get_data("particle", "label")
        particle = larcv.Particle()
        particle.energy_init(energy)
        larcv_particle.append(particle)
        ################################################################################

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

        max_index = next_new_meta.total_voxels()
        _ = [st.emplace(larcv.Voxel(index[i], weights[i]), True) for i in range(len(index)) if (not is_nan_array[i] and index[i] < max_index) ]

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


    evtfile.close()

    return


if __name__ == '__main__':


    parser = argparse.ArgumentParser(
        description     = 'Convert NEXT data files into larcv format',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input-file',
        type    = str,
        default = "",
        help    = 'Input file to convert')

    parser.add_argument('--output-file',
        type    = str,
        default = "",
        help    = 'Output name for the larcv file')


    parser.add_argument('--start-entry',
        type    = int,
        default = 0,
        help    = 'Entry to start conversion at')

    parser.add_argument('--end-entry',
        type    = int,
        default = -1,
        help    = 'Entry to end conversion at')

    args = parser.parse_args()
    main(args)

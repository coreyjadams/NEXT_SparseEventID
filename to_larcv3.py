from larcv import larcv

import pandas
import glob
import os
import argparse
import numpy


def main():
    # This code loops over training set files:
    top_level_path="/Users/corey.adams/data/NEXT/recorded_datasets/"
    output_path="/Users/corey.adams/data/NEXT/recorded_datasets_larcv/"
    # convert_training_set('')
    convert_data_set(top_level_path, output_path)

# files = files[0:5]

def get_NEW_meta():

    next_new_meta = larcv.ImageMeta3D()
    # set_dimension(size_t axis, double image_size, size_t number_of_voxels, double origin = 0);

    next_new_meta.set_dimension(0, 450, 45, -225)
    next_new_meta.set_dimension(1, 450, 45, -225)
    next_new_meta.set_dimension(2, 550, int(550/2), 0)

    return next_new_meta

def convert_training_set(top_input_path, output_path, glob_filter="*.h5"):
    files = glob.glob(top_input_path + glob_filter)

    n_files = len(files)



    # # Open the input file:
    training_file_index = int(0.75 * n_files)
    testing_file_index = int(0.825 * n_files)

    file_dictionary = {
        "train" : files[:training_file_index],
        "test"  : files[training_file_index:testing_file_index],
        "val"   : files[testing_file_index:-1]
    }



    for mode in ['train', 'test', 'val']:

        if mode == 'test': continue
        if mode == 'val':  continue

        output = output_path + "next_new_classification_{}.h5".format(mode)
        io_manager = larcv.IOManager(larcv.IOManager.kWRITE)
        io_manager.set_out_file(output)
        io_manager.initialize()


        for f in file_dictionary[mode]:
            convert_file(io_manager, f, is_mc = True)


        io_manager.finalize()


def convert_data_set(top_input_path, output_path, glob_filter="*.h5"):
    files = glob.glob(top_input_path + glob_filter)

    print(files)
    n_files = len(files)

    # Each data file is processed independently
    for f in files:

        output = os.path.basename(f.replace(".h5", "_larcv.h5"))
        output = output_path + "/" + output
        io_manager = larcv.IOManager(larcv.IOManager.kWRITE)
        io_manager.set_out_file(output)
        io_manager.initialize()


        convert_file(io_manager, f, is_mc = False)


        io_manager.finalize()


def convert_file(io_manager, file_name, is_mc=True):
    print("Opening file ", file_name)
    df = pandas.read_hdf(file_name)

    next_new_meta = get_NEW_meta()

    # Can override is_mc if information is not present:
    if 'true_energy' not in df.keys():
        print("Missing MC Truth keys, not trying to parse MC info")
        is_mc = False

    for event in numpy.unique(df.event):
        sub_df = df.query("event == {}".format(event))
        Run=sub_df.Run.iloc[0]
        if is_mc:
            sub_run = sub_df.file_int.iloc[0]
        else:
            sub_run = 0

        io_manager.set_id(int(Run), int(sub_run), int(event))

        ################################################################################
        # Store the particle information:
        if is_mc:
            larcv_particle = larcv.EventParticle.to_particle(io_manager.get_data("particle", "label"))
            particle = larcv.Particle()
            particle.energy_init(sub_df.true_energy.iloc[0])
            particle.pdg_code(int(sub_df.label.iloc[0]))
            larcv_particle.emplace_back(particle)
        ################################################################################


        ################################################################################
        # Store the voxel information:
        event_sparse3d_E = larcv.EventSparseTensor3D.to_sparse_tensor(
            io_manager.get_data("sparse3d", "voxels_E"))
        event_sparse3d_Q = larcv.EventSparseTensor3D.to_sparse_tensor(
            io_manager.get_data("sparse3d", "voxels_Q"))

        st_E = larcv.SparseTensor3D()
        st_Q = larcv.SparseTensor3D()
        st_E.meta(next_new_meta)
        st_Q.meta(next_new_meta)


        position = larcv.VectorOfDouble()
        position.resize(3)
        for index, row in sub_df.iterrows():
            position[0] = row.X
            position[1] = row.Y
            position[2] = row.Z
            index = next_new_meta.position_to_index(position)
            # coords = next_new_meta.position_to_coordinate(position)
            # print("({}, {}, {}) maps to ({}, {}, {}) ==  {}".format(
            #     row.X,
            #     row.Y,
            #     row.Z,
            #     coords[0],
            #     coords[1],
            #     coords[2],
            #     index))
            if index >= next_new_meta.total_voxels():
                print("Skipping voxel at original coordinates ({}, {}, {}) as it is out of bounds".format(
                    row.X, row.Y, row.Z))
                continue
            st_E.emplace(larcv.Voxel(index, row.E))
            st_Q.emplace(larcv.Voxel(index, row.Q))

        event_sparse3d_E.emplace(st_E)
        event_sparse3d_Q.emplace(st_Q)


        ################################################################################

        io_manager.save_entry()


    return


if __name__ == '__main__':
    main()

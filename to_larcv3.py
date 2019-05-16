from larcv import larcv

import pandas
import glob
import argparse
import numpy



files = glob.glob("/Users/corey.adams/data/NEXT/mmkekic_dataset/Tl208_NEW_v1_03_01_nexus_v5_03_04_cut*.h5")

# files = files[0:5]

n_files = len(files)



# # Open the input file:
training_file_index = int(0.75 * n_files)
testing_file_index = int(0.825 * n_files)

file_dictionary = {
    "train" : files[:training_file_index],
    "test"  : files[training_file_index:testing_file_index],
    "val"   : files[testing_file_index:-1]
}






next_new_meta = larcv.ImageMeta3D()
# set_dimension(size_t axis, double image_size, size_t number_of_voxels, double origin = 0);

next_new_meta.set_dimension(0, 450, 45, -225)
next_new_meta.set_dimension(1, 450, 45, -225)
next_new_meta.set_dimension(2, 550, int(550/2), 0)

print(next_new_meta.dump())

for mode in ['train', 'test', 'val']:

    if mode == 'test': continue
    if mode == 'val':  continue

    output = "next_new_classification_{}.h5".format(mode)
    io_manager = larcv.IOManager(larcv.IOManager.kWRITE)
    io_manager.set_out_file(output)
    io_manager.initialize()


    for f in file_dictionary[mode]:
        print("Opening file ", f)
        df = pandas.read_hdf(f)
        for event in numpy.unique(df.event):
            sub_df = df.query("event == {}".format(event))
            Run=sub_df.Run.iloc[0]
            sub_run = sub_df.file_int.iloc[0]

            io_manager.set_id(int(Run), int(sub_run), int(event))

            ################################################################################
            # Store the particle information:
            larcv_particle = larcv.EventParticle.to_particle(io_manager.get_data("particle", "label"))
            particle = larcv.Particle()
            particle.energy_init(sub_df.true_energy.iloc[0])
            particle.pdg_code(int(sub_df.label.iloc[0]))
            larcv_particle.emplace_back(particle)
            ################################################################################


            ################################################################################
            # Store the voxel information:
            event_sparse3d = larcv.EventSparseTensor3D.to_sparse_tensor(io_manager.get_data("sparse3d", "voxels"))

            st = larcv.SparseTensor3D()
            st.meta(next_new_meta)


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
                st.emplace(larcv.Voxel(index, row.E))

            event_sparse3d.emplace(st)


            ################################################################################

            io_manager.save_entry()


    io_manager.finalize()


# input_file = h5py.File(args.input)
# print(input_file)


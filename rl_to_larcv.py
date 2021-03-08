import argparse
import pathlib

import tables
import larcv
import numpy

pdg_lookup = {
    b'Pb208'           : 30000000,
    b'Pb208[2614.522]' : 30000000,
    b'Pb208[3197.711]' : 30000000,
    b'Pb208[3475.078]' : 30000000,
    b'Pb208[3708.451]' : 30000000,
    b'Pb208[3919.966]' : 30000000,
    b'Pb208[3961.162]' : 30000000,
    b'Pb208[4125.347]' : 30000000,
    b'Pb208[4180.414]' : 30000000,
    b'Pb208[4296.560]' : 30000000,
    b'anti_nu_e'       : -12,
    b'e+'              : -11,
    b'e-'              : 11,
    b'gamma'           : 22,
    b'Tl208'           : 20000000,

}


def main():

    parser = argparse.ArgumentParser(
        description     = 'Convert NEXT data files into larcv format',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input-file",
        type=pathlib.Path,
        nargs="+",
        required = True,
        help="input file[s] to convert"
        )

    parser.add_argument("-o", "--output-location",
        type=pathlib.Path,
        required=True,
        help="Destination for converted files.  Name of file is preserved with _larcv inserted.")

    args = parser.parse_args()

    print(args)
    for input_file in args.input_file:
        convert_file(input_file, args.output_location)

# files = files[0:5]


def get_NEW_meta():

    next_new_meta = larcv.ImageMeta3D()
    # set_dimension(size_t axis, double image_size, size_t number_of_voxels, double origin = 0);

    next_new_meta.set_dimension(0, 450, 45, -225)
    next_new_meta.set_dimension(1, 450, 45, -225)
    next_new_meta.set_dimension(2, 550, int(550/2), 0)

    return next_new_meta

def get_NEW_LR_meta():

    next_new_meta = larcv.ImageMeta3D()
    # set_dimension(size_t axis, double image_size, size_t number_of_voxels, double origin = 0);

    next_new_meta.set_dimension(0, 450, 450, -225)
    next_new_meta.set_dimension(1, 450, 450, -225)
    next_new_meta.set_dimension(2, 549, 305, 0)

    return next_new_meta

def basic_event_pass(summary):


    # 1 track:
    mask = summary['ntrks'] == 1


    # Z Min:
    mask = numpy.logical_and(mask, summary['z_min'] > 20.0) # Z = 0 + 2cm

    # Z Max:
    mask = numpy.logical_and(mask, summary['z_max'] < 510.0) # Z = 55 - 2 cm

    # R Max:
    mask = numpy.logical_and(mask, summary['r_max'] < 180.0) # R = 180 CM from ander

    return mask

def store_lr_hits(io_manager, this_lr_hits):
    event_sparse3d = io_manager.get_data("sparse3d", "lr_hits")

    meta = get_NEW_LR_meta()

    st = larcv.SparseTensor3D()
    st.meta(meta)


    unique = numpy.unique(this_lr_hits['Z'])

    for row in this_lr_hits:
        index = meta.position_to_index([row['X'], row['Y'], row['Z']])

        if index >= meta.total_voxels():
            print("Skipping voxel at original coordinates ({}, {}, {}) as it is out of bounds".format(
                row['X'], row['Y'], row['Z']))
            continue
        st.emplace(larcv.Voxel(index, row['E']), False)

    event_sparse3d.set(st)

def store_mc_info(io_manager, this_hits, this_particles):
    event_cluster3d = io_manager.get_data("cluster3d", "mc_hits")

    cluster_indexes = numpy.unique(this_hits['particle_indx'])

    meta = get_NEW_LR_meta()

    sc = larcv.SparseCluster3D()
    sc.meta(meta)
    # sc.resize(len(cluster_indexes))

    cluster_lookup = {}
    for i, c in enumerate(cluster_indexes):
        cluster_lookup[c] = i

    vs = [ larcv.VoxelSet() for i in cluster_indexes]

    # Add all the hits to the right cluster:
    for hit in this_hits:
        # Get the index from the meta
        index = meta.position_to_index(hit['hit_position'])
        # Create a voxel on the fly with the energy
        vs[cluster_lookup[hit['particle_indx']]].add(larcv.Voxel(index, hit['hit_energy']))

    # Add the voxel sets into the cluster set
    for i, v in enumerate(vs):
        v.id(i)  # Set id
        sc.insert(v)

    # Store the mc_hits as a cluster 3D
    event_cluster3d.set(sc)

    particle_set = io_manager.get_data("particle", "all_particles")

    positron = False

    # Now, store the particles:
    for i, particle in enumerate(this_particles):

        if particle['particle_name'] == b'e+' and particle['initial_volume'] == b'ACTIVE': positron = True

        if b'Pb208' in particle['particle_name']:
            pdg_code = 30000000
        else:
            pdg_code = pdg_lookup[particle['particle_name']]

        p = larcv.Particle()
        p.id(i) # id
        p.track_id(particle['particle_indx'])
        p.nu_current_type(particle['primary']) # Storing primary info in nu_current_type
        p.pdg_code(pdg_code)
        p.parent_track_id(particle['mother_indx'])
        p.position(*particle['initial_vertex'])
        p.end_position(*particle['final_vertex'])
        p.creation_process(particle['creator_proc'])
        p.energy_init(particle['kin_energy'])
        p.momentum(*particle['momentum'])

        particle_set.append(p)      

    return positron



def energy_corrected(energy, z_min, z_max):
    Z_corr_factor = 2.76e-4 

    return energy/(1. - Z_corr_factor*(z_max-z_min))

def convert_file(input_file, output_directory):
    
    # First, open and validate the input file:
    evtfile = tables.open_file(str(input_file), 'r')


    # No DECO table?  No good, just skip this file
    if not hasattr(evtfile.root, "DECO"): 
        evtfile.close()
        return


    # Read if we have MC or not:
    if hasattr(evtfile.root, "MC"):
        is_mc = True
    else:
        is_mc = False


    # Format output name:
    output_name = input_file.name.replace(".h5", "_larcv.h5")
    output      = output_directory /  pathlib.Path(output_name)
    
    # Create an output larcv file:
    io_manager = larcv.IOManager(larcv.IOManager.kWRITE)
    io_manager.set_out_file(str(output))
    io_manager.initialize()

    # Now, ready to go.  Read in a couple tables:

    # - Summary table.  gives event number, ntrks, min and max of all coords.
    #  - Use this to reject multi track events and events near the walls.
    #  - use this to get the event number.
    # - Run has just run info.
    #  - read this and use it to get the run number.
    # - DECO contains the deconvolved hits.  They are stored by event number, contain x/y/z/E
    #  - read this and get the hits from each event.
    # - (ONLY MC): MC contains mc truth information.
    #  - read this for whole-event labels, but also gather out 

    if is_mc:
        mc_extents   = evtfile.root.MC.extents.read()
        mc_hits      = evtfile.root.MC.hits.read()
        mc_particles = evtfile.root.MC.particles.read()


    events = evtfile.root.Run.events.read()
    run = evtfile.root.Run.runInfo.read()
    summary = evtfile.root.SUMMARY.Events.read()
    # event no is events[i_evt][0]
    # run no is run[i_evt][0]
    # We'll set all subrun info to 0, it doesn't matter.

    this_run = run[0][0]


    event_numbers = events['evt_number']
    # event_energy  = summary['evt_energy']

    lr_hits = evtfile.root.DECO.Events.read()
    
    next_new_meta = get_NEW_meta()

    mask = basic_event_pass(summary)

    passed_events = summary['event'][mask]


    # print(numpy.unique(lr_hits['event'], return_counts=True))

    for i_evt, event_no in enumerate(event_numbers):

        # Did this event pass the basic event cuts?
        if event_no not in passed_events: continue

        io_manager.set_id(this_run, 0, event_no)

        # Slice off this summary object:
        this_summary = summary[summary['event'] == event_no]

        # Parse out the deconv hits:
        this_lr_hits = lr_hits[lr_hits['event'] == event_no]
        store_lr_hits(io_manager, this_lr_hits)

        # We store the measured energy, correct, in 'energy_deposit'
        # We store the mc energy, if we have it, in 'energy_init'
        particle = larcv.Particle()

        if is_mc:
            # Store the mc infomation.  Extract this events hits, particles, etc.

            # Slice this extents:
            mc_mask = mc_extents['evt_number'] == event_no
            this_index = numpy.argwhere(mc_mask)[0][0]

            this_mc_extents = mc_extents[this_index]
            particle_stop = int(this_mc_extents['last_particle'] + 1) # Particle index is not inclusive in the last index, add one
            hit_stop      = int(this_mc_extents['last_hit'] + 1)      # Particle index is not inclusive in the last index, add one

            if this_index != 0:
                previous_mc_extents = mc_extents[this_index - 1]
                particle_start = int(previous_mc_extents['last_particle'] + 1)
                hit_start      = int(previous_mc_extents['last_hit'] + 1)
            else:
                particle_start = 0
                hit_start      = 0


            this_particles = mc_particles[particle_start:particle_stop]
            this_hits      = mc_hits[hit_start:hit_stop]

            positron = store_mc_info(io_manager, this_hits, this_particles)

            # First, we figure out the extents for this event.


            if positron:
                particle.pdg_code(0)
            else:
                particle.pdg_code(1)

            # Calculate the true energy of the event:
            true_e = numpy.sum(this_hits['hit_energy'])
            particle.energy_init(true_e)

        # Calculate the reconstructed energy of the event:
        energy = numpy.sum(this_lr_hits['E'])
        energy = energy_corrected(energy, this_summary['z_min'][0], this_summary['z_max'][0])
        particle.energy_deposit(energy)
        # Store the whole measured energy of the event
        event_part   = io_manager.get_data("particle", "event")
        event_part.append(particle)

    # for event in numpy.unique(df.event):
    #     sub_df = df.query("event == {}".format(event))
    #     Run=sub_df.Run.iloc[0]
    #     if is_mc:
    #         sub_run = sub_df.file_int.iloc[0]
    #     else:
    #         sub_run = 0

    #     io_manager.set_id(int(Run), int(sub_run), int(event))

    #     ################################################################################
    #     # Store the particle information:
    #     if is_mc:
    #         larcv_particle = larcv.EventParticle.to_particle(io_manager.get_data("particle", "label"))
    #         particle = larcv.Particle()
    #         particle.energy_init(sub_df.true_energy.iloc[0])
    #         particle.pdg_code(int(sub_df.label.iloc[0]))
    #         larcv_particle.emplace_back(particle)
    #     ################################################################################


      

        io_manager.save_entry()
        # if i_evt > 50:
        #     break

    # Close Larcv:
    io_manager.finalize()

    # Close tables:
    evtfile.close()

    return


if __name__ == '__main__':
    main()
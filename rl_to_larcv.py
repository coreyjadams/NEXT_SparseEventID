import argparse
import pathlib

import tables
import larcv
import numpy
import pandas
# import anytree

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

def has_table(table, table_name):

    try:
        table.get_node(table_name)
        return True
    except tables.exceptions.NoSuchNodeError:
        return False

def read_mandatory_tables(input_files):

    # List of tables that must be found:
    mandatory_tables = [
        "/DECO/Events/",
        "/Summary/Events/",
        "/Run/events/",
        "/Run/runInfo/",
        "/PMAPS/S1/",
        "/PMAPS/S1Pmt/",
        "/PMAPS/S2/",
        "/PMAPS/S2Pmt/",
        "/PMAPS/S2Si/",
    ]

    optional_mc_tables = [
        "/MC/extents/",
        "/MC/hits/",
        "/MC/particles/",
    ]

    image_tables = {}
    mc_tables    = {}

    for _f in input_files:
        open_file = tables.open_file(str(_f), 'r')
        # print(open_file)
        # Look for the mandatory tables in this file:
        m_found_tables = {}
        for table_name in mandatory_tables:
            # print(f"looking for {table_name}")
            if has_table(open_file, table_name):
                m_found_tables[table_name] = open_file.get_node(table_name).read()

                # print(f"Found {table_name}")
            else:
                # print(f"Didn't find {table_name}")
                pass
            # print(m_found_tables.keys())
            # mandatory_tables.remove(table_name)
        # image_tables[table_name] = this_table.read()

        # Copy the found tables into the right spot:
        image_tables.update(m_found_tables)

        # remove everything that's been found:
        for key in m_found_tables.keys():
            if key in mandatory_tables: mandatory_tables.remove(key)


        # Look for the optional MC tables:
        o_found_tables = {}
        for table_name in optional_mc_tables:
            # print(f"looking for {table_name}")
            if has_table(open_file, table_name):
                o_found_tables[table_name] = open_file.get_node(table_name).read()
                # print(f"Found {table_name}")
            else:
                # print(f"Didn't find {table_name}")
                pass
            # print(o_found_tables.keys())

        mc_tables.update(o_found_tables)
        for key in o_found_tables.keys():
            if key in optional_mc_tables: optional_mc_tables.remove(key)


        # Close the file:
        open_file.close()

    if len(mandatory_tables) != 0:
        raise Exception(f"Could not find mandatory tables {mandatory_tables}")

    if len(optional_mc_tables) != 0:
        print("Not all mc tables found, skipping MC")
        mc_tables = None


    return image_tables, mc_tables

def convert_entry_point(input_files, output_file, db_location):

    image_tables, mc_tables = read_mandatory_tables(input_files)

    sipm_db = pandas.read_pickle(db_location)
    db_lookup = {
        "x_lookup" : numpy.asarray(sipm_db['X']),
        "y_lookup" : numpy.asarray(sipm_db['Y']),
        "active"   : numpy.asarray(sipm_db['Active']),
    }


    convert_to_larcv(image_tables, mc_tables, output_file, db_lookup)


def main():

    parser = argparse.ArgumentParser(
        description     = 'Convert NEXT data files into larcv format',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input",
        type=pathlib.Path,
        required=True,
        nargs="+",
        help="Input files.  Will search through files until required tables are found.")

    parser.add_argument("-o", "--output",
        type=pathlib.Path,
        required=True,
        help="Name of output file.")

    parser.add_argument('-db', "--sipm-db-file",
        type=pathlib.Path,
        required=True,
        help="Location of the sipm db file for this input, if pmaps is given.")

    args = parser.parse_args()

    convert_entry_point(args.input, args.output, args.sipm_db_file)


# files = files[0:5]


def get_NEW_meta():

    next_new_meta = larcv.ImageMeta3D()
    # set_dimension(size_t axis, double image_size, size_t number_of_voxels, double origin = 0);

    next_new_meta.set_dimension(0, 480, 48, -240)
    next_new_meta.set_dimension(1, 480, 48, -240)
    next_new_meta.set_dimension(2, 550, int(550/2), 0)
    # print(next_new_meta)
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
    mask = summary['evt_ntrks'] == 1


    # Z Min:
    mask = numpy.logical_and(mask, summary['evt_z_min'] > 20.0) # Z = 0 + 2cm

    # Z Max:
    mask = numpy.logical_and(mask, summary['evt_z_max'] < 510.0) # Z = 55 - 2 cm

    # R Max:
    mask = numpy.logical_and(mask, summary['evt_r_max'] < 180.0) # R = 180 CM from ander

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

    # How to find the vertex?  It's where the gamma interacts.
    # There can be multiple gammas, so take the one that comes first.

    found_vertex = False
    vertex = None


    vertex_set = io_manager.get_data("bbox3d", "vertex")
    vertex_collection = larcv.BBoxCollection3D()
    vertex_collection.meta(meta)

    # # Create a tree of particles to help sort out the primary gammas and the vertex.
    # # On the first pass through, just gathering up the nodes:
    # nodes = {}
    # primary = None
    # for particle in this_particles:
    #     node = anytree.AnyNode(id=particle['particle_indx'], particle=particle)
    #     # Tag the primary:
    #     if particle['primary'] == 1:
    #         node.parent = None
    #         primary = node
    #     else:
    #         nodes.update({particle['particle_indx'] : node})

    # print(nodes)

    # # Here, sort the list out:
    # orphans = {}
    # while len(nodes) > len(orphans):
    #     # Go through the list assigning nodes to their parent.
    #     # The primary was never added to the list ofnodes so we shouldn't hit it.
    #     node_indx, top_node = nodes.popitem()
    #     target_indx = top_node.particle['mother_indx']
    #     if target_indx == primary.particle['particle_indx']:
    #         primary.children = primary.children + (top_node,)
    #     elif target_indx in nodes.keys():
    #         nodes[target_indx].children = nodes[target_indx].children + (top_node,)
    #     else:
    #         print("Failed to find mother for node ", node)
    #         orphans[node_indx] = top_node
    #     # for other_nodes in nodes:
    #     #     if other_nodes.particle['particle_indx'] == target_indx:
    #     #         other_nodes
    #     #         nodes.pop(node)
    #     #         break
    #     print("Current length of nodes is: ", len(nodes))


    # print(primary.children)

    # Now, store the particles:
    for i, particle in enumerate(this_particles):

        if particle['particle_name'] == b'e+' and particle['initial_volume'] == b'ACTIVE':
            positron = True
            # print("Positron? ", particle)

        # Criteria to be the 2.6 MeV gamma, the final point of which we use as the vertex:
        # particle_name == "gamma"
        # Parent's name == "Pb208[2614.552]" OR kinetic_energy = 2.6145043



        if particle['particle_name'] == b'gamma' \
            and numpy.abs(particle['kin_energy'] - 2.6145043) < 0.01 \
            and not found_vertex:
            # print(particle.dtype)
            vertex = [particle['final_x'], particle['final_x'], particle['final_x']]
            # vertex = particle['final_vertex']

            # Check that the vertex is in the fiducial volume:
            if vertex[2] < 510.0 and vertex[2] > 20.0:
                if numpy.sqrt(vertex[0]**2 + vertex[1]**2) < 180.:


                    found_vertex = True
                    # print("Vertex Candidate: ", particle)
                    vertex_bbox = larcv.BBox3D(
                        (vertex[0], vertex[1], vertex[2]),
                        (0., 0., 0.)
                    )
                    vertex_collection.append(vertex_bbox)

        if b'Pb208' in particle['particle_name']:
            pdg_code = 30000000
        elif particle['particle_name'] in pdg_lookup.keys():
            pdg_code = pdg_lookup[particle['particle_name']]
        else:
            pdg_code = -123456789

        # print(particle.dtype)
        p = larcv.Particle()
        p.id(i) # id
        p.track_id(particle['particle_id'])
        p.nu_current_type(particle['primary']) # Storing primary info in nu_current_type
        p.pdg_code(pdg_code)
        p.parent_track_id(particle['mother_id'])
        # p.position(*particle['initial_vertex'])
        p.position(
            particle['initial_x'],
            particle['initial_y'],
            particle['initial_z'],
            particle['initial_t']
        )
        p.end_position(
            particle['final_x'],
            particle['final_y'],
            particle['final_z'],
            particle['final_t']
        )
        # p.end_position(*particle['final_vertex'])
        p.creation_process(particle['creator_proc'])
        p.energy_init(particle['kin_energy'])
        # p.momentum(*particle['momentum'])
        p.momentum(
            particle['initial_momentum_x'],
            particle['initial_momentum_y'],
            particle['initial_momentum_z']
        )

        particle_set.append(p)


    vertex_set.append(vertex_collection)

    return positron


def slice_into_event(_pmaps, event_number, _keys):
    # What does this correspond to in the raw file?
    selection = { key : _pmaps[key]['event'] == event_number for key in _keys }
    this_pmaps = { key : _pmaps[key][selection[key]] for key in _keys}

    return this_pmaps

def store_pmaps(io_manager, this_pmaps, db_lookup):

    # SiPM locations range from -235 to 235 mm in X and Y (inclusive) every 10mm
    # That's 47 locations in X and Y.


    # First, we note the time of S1, which will tell us Z locations
    s1_e = this_pmaps["S1"]["ene"]
    if len(s1_e) == 0: return
    s1_peak = numpy.argmax(s1_e)
    # This will be in nano seconds
    s1_t    = this_pmaps['S1']['time'][s1_peak]



    s2_times = this_pmaps['S2']['time']
    waveform_length = len(s2_times)

    # This is more sensors than we need, strictly.  Not all of them are filled.

    # For each sensor in the raw waveforms, we need to take the sensor index,
    # look up the X/Y,
    # convert to index, and deposit in the dense data

    # We loop over the waveforms in chunks of (s2_times)


    # Figure out the total number of sensors:
    n_sensors = int(len(this_pmaps["S2Si"]) / waveform_length)

    # print(n_sensors)



    # Get the energy, and use it to select only active hits
    energy      = this_pmaps["S2Si"]["ene"]
    # The energy is over all sipms:
    energy_selection   = energy != 0.0

    # # Make sure we're selecting only active sensors:
    active_selection   = numpy.take(db_lookup["active"], this_pmaps["S2Si"]["nsipm"]).astype(bool)



    # # Merge the selections:
    # selection = numpy.logical_and(energy_selection, active_selection)
    # selection = active_selection

    # Each sensor has values, some zero, for every tick in the s2_times.
    # The Z values are constructed from these, so stack this vector up
    # by the total number of unique sensors
    # print("s2_times: ", s2_times, len(s2_times))
    # print("n_sensors: ", n_sensors)
    # print("selection: ", selection)
    # ticks       = numpy.tile(s2_times, n_sensors)[selection]
    ticks       = numpy.tile(s2_times, n_sensors)
    # print(ticks)

    # x and y are from the sipm lookup tables, and then filter by active sites
    # x_locations = numpy.take(db_lookup["x_lookup"], this_pmaps["S2Si"]["nsipm"])[selection]
    # y_locations = numpy.take(db_lookup["y_lookup"], this_pmaps["S2Si"]["nsipm"])[selection]


    x_locations = numpy.take(db_lookup["x_lookup"], this_pmaps["S2Si"]["nsipm"])
    y_locations = numpy.take(db_lookup["y_lookup"], this_pmaps["S2Si"]["nsipm"])

    # Filter the energy to active sites
    energy      = energy
    # energy      = energy[selection]

    # Convert to physical coordinates
    z_locations = ((ticks - s1_t) / 1000).astype(numpy.int32)


    # Put them into larcv:
    event_sparse3d = io_manager.get_data("sparse3d", "pmaps")

    meta = get_NEW_meta()

    st = larcv.SparseTensor3D()
    st.meta(meta)


    for x, y, z, e in zip(x_locations, y_locations, z_locations, energy) :
        if e > 0:
            index = meta.position_to_index([x, y, z])

            if index >= meta.total_voxels():
                print(f"Skipping voxel at original coordinates ({x}, {y}, {z}, index {index}) as it is out of bounds")
                continue
            st.emplace(larcv.Voxel(index, e), False)

    event_sparse3d.set(st)

    return




def energy_corrected(energy, z_min, z_max):
    Z_corr_factor = 2.76e-4

    return energy/(1. - Z_corr_factor*(z_max-z_min))

def convert_to_larcv(image_tables, mc_tables, output_name, db_lookup):

    # print(image_tables.keys())
    # print(mc_tables.keys())

    if mc_tables is not None:
        is_mc = True
    else:
        is_mc = False


    # # Format output name:
    # output_name = input_lr_file.name.replace(".h5", "_larcv.h5")
    # output      = output_directory /  pathlib.Path(output_name)

    # Create an output larcv file:
    io_manager = larcv.IOManager(larcv.IOManager.kWRITE)
    io_manager.set_out_file(str(output_name))
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
        mc_extents   = mc_tables["/MC/extents/"]
        mc_hits      = mc_tables["/MC/hits/"]
        mc_particles = mc_tables["/MC/particles/"]

    events  = image_tables["/Run/events/"]
    run     = image_tables["/Run/runInfo/"]
    summary = image_tables["/Summary/Events/"]

    # event no is events[i_evt][0]
    # run no is run[i_evt][0]
    # We'll set all subrun info to 0, it doesn't matter.

    this_run = run[0][0]


    event_numbers = events['evt_number']
    # event_energy  = summary['evt_energy']

    # lr_hits = evtfile.root.DECO.Events.read()
    lr_hits = image_tables["/DECO/Events/"]

    keys = {"S1", "S1Pmt", "S2", "S2Pmt", "S2Si"}
    pmap_tables = {key : image_tables["/PMAPS/" + key + "/"] for key in keys}


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


        # Get the pmaps:
        this_pmaps = slice_into_event(pmap_tables, event_no, keys)

        store_pmaps(io_manager, this_pmaps, db_lookup)

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
            # print("Event number: ", event_no, "(positron: ", positron, ")")

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
        energy = energy_corrected(energy, this_summary['evt_z_min'][0], this_summary['evt_z_max'][0])
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
    # evtfile.close()

    return


if __name__ == '__main__':
    main()

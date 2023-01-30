import parsl
import os
from parsl.app.app import python_app, bash_app
# from parsl.configs.local_threads import config


from parsl.data_provider.files import File
from parsl.dataflow.memoization import id_for_memo


@id_for_memo.register(File)
def id_for_memo_File(f, output_ref=False):
    if output_ref:
        # logger.debug("hashing File as output ref without content: {}".format(f))
        return f.url
    else:
        # logger.debug("hashing File as input with content: {}".format(f))
        assert f.scheme == "file"
        filename = f.filepath
        try:
            stat_result = os.stat(filename)

            return [f.url, stat_result.st_size, stat_result.st_mtime]
        except:
            return [f.url, 0, 0]

# parsl.set_stream_logger() # <-- log everything to stdout

print(parsl.__version__)


from parsl.config import Config

# from libsubmit.providers.local.local import Local
from parsl.providers import PBSProProvider, LocalProvider
from parsl.channels import LocalChannel
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.launchers import MpiExecLauncher, GnuParallelLauncher
from parsl.addresses import address_by_hostname
from parsl.monitoring.monitoring import MonitoringHub
from parsl.utils import get_all_checkpoints

user_opts = {
    "cpus_per_node" : 10,
    "run_dir" : "/data/datasets/NEXT/NEW-simulation/runinfo",
    "strategy" : "simple"
}

def create_config(user_opts):

    checkpoints = get_all_checkpoints(user_opts["run_dir"])
    print("Found the following checkpoints: ", checkpoints)

    config = Config(
            # executors=[
                # HighThroughputExecutor(
                #     label="htex",
                #     heartbeat_period=15,
                #     heartbeat_threshold=120,
                #     worker_debug=True,
                #     max_workers=user_opts["cpus_per_node"],
                #     cores_per_worker=1,
                #     address=address_by_hostname(),
                #     cpu_affinity="alternating",
                #     prefetch_capacity=0,
                #     provider=LocalProvider(
                #         launcher=SingleNodeLauncher(debug=False),
                #     ),
                # ),
            executors=[ThreadPoolExecutor(
                label='threads', 
                managed=True, 
                max_threads=2, 
                storage_access=None, 
                thread_name_prefix='', 
                working_dir=None)
            ],
            checkpoint_files = checkpoints,
            run_dir=user_opts["run_dir"],
            checkpoint_mode = 'task_exit',
            strategy=user_opts["strategy"],
            retries=0,
            app_cache=True,
    )

    return config

config = create_config(user_opts)

print(config)
parsl.clear()
parsl.load(config)



import pathlib


@python_app(cache=True)
def create_config_file(inputs, outputs, run, event_start, output_filename):
    """
    Read the templates and put specific data into the output files
    By convention, inputs[0] is the init template and inputs[1] is the mac template
    outputs[0] is the init file, outputs[1] is the mac file.
    """

    # read the templates:
    with open(inputs[0], 'r') as _f:
        init_template = "".join(_f.readlines())
    with open(inputs[1], 'r') as _f:
        mac_template = "".join(_f.readlines())


    this_macro = mac_template.format(
        event       = event_start,
        output_file = output_filename.rstrip(".h5"),
        seed        = event_start+1
    )


    # Write the macro:
    with open(outputs[1].filepath, 'w') as _f:
        _f.write(this_macro)

    this_init = init_template.format(
        mac_file = outputs[1].url
    )

    with open(outputs[0].filepath, 'w') as _f:
        _f.write(this_init)

    return

@bash_app(cache=True)
def nexus_simulation(inputs, outputs, n_events, workdir, stdout, stderr):
    """
    inputs[0] should be the mac file
    outputs[0] is the output file name
    """
    script = """

source /home/cadams/NEXT/setup_nexus.sh

cd {workdir}
nexus -n {n_events} -c {mac}

rm GammaEnergy.root

    """.format(
            workdir  = workdir,
            n_events = n_events,
            mac      = inputs[0]
        )

    return script

@bash_app(cache=True)
def ic(inputs, outputs, workdir, city, config, stdout, stderr):
    """
    inputs[0] should be the input file
    outputs[0] is the output file name
    """

    script = """

# Set up IC with this stuff:
source /home/cadams/miniconda/bin/activate
conda activate IC-3.8-2022-04-13
export ICTDIR=/home/cadams/NEXT/IC/
export ICDIR=$ICTDIR/invisible_cities
export PYTHONPATH=$ICTDIR

cd {workdir}

export PATH=$ICTDIR/bin:$PATH

city {city}  -i {input} -o {output} --event-range=all {config}    

    """.format(
        city   = city, 
        config = config, 
        workdir = workdir,
        input  = inputs[0].url, 
        output = outputs[0].url)
    print(script)
    return script


@bash_app(cache=True)
def larcv(inputs, outputs, workdir, script, stdout, stderr):
    """
    inputs[0] should be the input file
    outputs[0] is the output file name
    """

    script = """

cd {workdir}

python {script}  -lri {input} -o {output}

    """.format(
        script  = script, 
        workdir = workdir,
        input   = inputs[0].url, 
        output  = outputs[0].url)
    print(script)
    return script



def simulate_and_reco_file(top_dir, run, subrun, event_offset, n_events, templates):

    # This simulates one file's worth of events
    # through to where it can be reconstructed

    local_outdir = top_dir / pathlib.Path(f"r{run}/s{subrun}")
    local_outdir.mkdir(exist_ok=True, parents=True)

    # Subdirs in the outputs:
    config_dir = local_outdir / pathlib.Path("configs/")
    sim_dir    = local_outdir / pathlib.Path("sim/")
    log_dir    = local_outdir / pathlib.Path("logs/")

    # Make sure those exist:
    config_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)


    # First up is nexus config generation.

    # Parse the basename of the input templates:
    template_basenames = [ os.path.basename(f) for f in templates]

    # Change the basename for this run / subrun:
    template_basenames = [ 
        t.replace(".mac", f".r{run}_s{subrun}.mac") for t in template_basenames
    ]

    # The input templates can be data futures:
    inputs = [ File(str(f)) for f in input_templates ]


    # These are the output files for the config and macro:
    outputs = [
        File(str(config_dir / pathlib.Path(t))) for t in template_basenames
    ]



    # Specific name of the output file:
    output_template_base = os.path.basename(templates[0]).replace("init.mac","")
    output_file_format = output_template_base + f"r{run}_s{subrun}_nexus.h5"

    output_file = str(sim_dir / pathlib.Path(output_file_format))

    nexus_config_future = create_config_file(
        inputs          = inputs,
        outputs         = outputs,
        run             = run,
        event_start     = event_offset,
        output_filename = output_file

    )

    output_file = File(output_file)


    nexus_future = nexus_simulation(
        inputs   = [nexus_config_future.outputs[0]], 
        outputs  = [output_file,], 
        n_events = n_events,
        workdir  = str(log_dir),
        stdout   = str(log_dir) + "/nexus.stdout", 
        stderr   = str(log_dir) + "/nexus.stderr")

    latest_future = nexus_future

    ic_template_dir = os.path.dirname(templates[0]) + "/IC/"

    for city in ["detsim", "hypathia", "penthesilea", "esmeralda", "beersheba"]:
        latest_output = File(output_file.url.replace("nexus", city))
        latest_future = ic(
            inputs  = [latest_future.outputs[0],] , 
            outputs = [latest_output, ], 
            city    = city,
            config  = f"{ic_template_dir}/{city}.conf",
            workdir = str(log_dir),
            stdout  = str(log_dir) + f"/{city}.stdout", 
            stderr  = str(log_dir) + f"/{city}.stderr"
        )



    # Add a larcv step:
    larcv_script = pathlib.Path("to_larcv.py").resolve()
    larcv_output = File(output_file.url.replace("nexus", "larcv"))
    print(larcv_output)
    larcv_future = larcv(
        inputs  = [latest_future.outputs[0],],
        outputs = [larcv_output, ],
        workdir = str(log_dir),
        script  = str(larcv_script),
        stdout  = str(log_dir) + f"/larcv.stdout", 
        stderr  = str(log_dir) + f"/larcv.stderr"
    )


    return larcv_future


if __name__ == '__main__':

    # Where are the templates?
    template_dir = pathlib.Path("/home/cadams/NEXT/sample-generation/config_templates/NEW/Tl208/")
    
    # The input templates, which are not memoized:
    input_templates = [
        template_dir / pathlib.Path("NEW_MC208_NN.init.mac"),
        template_dir / pathlib.Path("NEW_MC208_NN.config.mac"),
    ]


    # Where to put the outputs?
    output_dir = pathlib.Path(f"/data/datasets/NEXT/NEW-simulation/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # This simulates a file with nexus and detsim
    # Will add diomira eventually but not there yet
    sim_future = simulate_and_reco_file(
        run       = 0,
        subrun    = 0,
        n_events  = 100000,
        event_offset = 0,
        top_dir   = output_dir,
        templates = input_templates
    )

    print(sim_future.result())

    # # This reconstructs that file:
    # input_filename = sim_future.outputs[0].filename
    # two_up = os.path.dirname(os.path.dirname(input_filename))


    # # Reco_future should take
    # reco_future = reconstruct_chain(two_up, sim_future.outputs[0])

    # # futures = spawn_chain(run=0, n_subrun=1, event_offset=200, n_events=10000)

    # # # Make parsl wait:
    # # for future in futures:
    # #     print(future.result())
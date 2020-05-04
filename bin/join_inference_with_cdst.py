import argparse
import pathlib

import tables
import pandas
import numpy


parser = argparse.ArgumentParser()


parser = argparse.ArgumentParser(
    description     = 'Merge inference result files with high level info from tables.',
    formatter_class = argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--inference-file',
    type    = pathlib.Path,
    default = "",
    help    = 'The inference numpy file.')

parser.add_argument('--cdst',
    type    = pathlib.Path,
    default = "",
    help    = 'The original CDST file')

args = parser.parse_args()

# Open the cdst file:
evtfile = tables.open_file(str(args.cdst), 'r')

# Open the inference file:
arrays = numpy.load(str(args.inference_file), allow_pickle=True)


inference_dict = {}
for i in range(len(arrays)):
    key = arrays[i][0]
    inference_dict[key] = arrays[i][1]

print(inference_dict.keys())

events = inference_dict['event']


unique_events = numpy.unique(events)

# Read in the summary and ensure all events are here:
summary = evtfile.root.Summary.Events.read()
tracks  = evtfile.root.Tracking.Tracks.read()

unique_events = numpy.asarray([e for e in unique_events if e in summary['event']])

print(unique_events)

energy  = summary['evt_energy']

unique_mask = numpy.asarray([numpy.argwhere(events == u)[0][0] for u in unique_events ])
for key in inference_dict:
    inference_dict[key] = inference_dict[key][unique_mask]


# Need to map out the location of indexes in the original table
# based on the event in the inference file

# print(summary['event'])
# print(unique_events)

# print(numpy.argwhere(summary['event'] == unique_events[0]))



index = numpy.asarray([ numpy.argwhere(summary['event'] == e)[0][0] for e in unique_events ])

# Create a dataframe:
#
df = pandas.DataFrame({
    'run'                : inference_dict['run'],
    'subrun'             : inference_dict['subrun'],
    'event'              : inference_dict['event'],
    'entry'              : inference_dict['entries'],
    'label'              : inference_dict['label'],
    'pred'               : inference_dict['pred'],
    'score_signal'       : inference_dict['softmax'][:,1],
    'score_background'   : inference_dict['softmax'][:,0],
    'evt_ntrks'          : summary['evt_ntrks'][index],
    'evt_energy'         : summary['evt_energy'][index],
    'evt_z_avg'          : summary['evt_z_avg'][index],
    'evt_r_avg'          : summary['evt_r_avg'][index],
    'evt_z_min'          : summary['evt_z_min'][index],
    'evt_r_min'          : summary['evt_r_min'][index],
    'evt_r_max'          : summary['evt_r_max'][index],
    'evt_z_max'          : summary['evt_z_max'][index],
    'eblob1'             : tracks['eblob1'][index],
    'eblob2'             : tracks['eblob2'][index],
})


# Last, we need to save the dataframe.
basename = str(args.inference_file.name).replace(".npy","")
dataframe_file_name = basename + "_joined.h5"
df.to_hdf(dataframe_file_name, key=basename)

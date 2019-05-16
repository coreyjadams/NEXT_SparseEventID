from larcv import larcv
import numpy

io = larcv.IOManager()
io.add_in_file("next_new_classification_train.h5")
io.initialize()

voxel_counts3d = numpy.zeros((io.get_n_entries(), 1))

for i in range(io.get_n_entries()):
    io.read_entry(i)
    image3d = larcv.EventSparseTensor3D.to_sparse_tensor(io.get_data("sparse3d", "voxels"))
    voxel_counts3d[i] = image3d.as_vector().front().as_vector().size()

    if i % 100 == 0:
        print("On entry ", i, " of ", io.get_n_entries())

    # if i > 10000:
        # break

print ("Average Voxel Occupation: ")
print("{av:.2f} +/- {rms:.2f} ({max} max)".format(
    av  = numpy.mean(voxel_counts3d[:]), 
    rms = numpy.std(voxel_counts3d[:]), 
    max = numpy.max(voxel_counts3d[:])
    )
)

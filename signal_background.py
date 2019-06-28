from larcv import larcv
import numpy

io = larcv.IOManager()
io.add_in_file("/ccs/home/deltutto/data/next_new_classification_train.h5")
io.initialize()
signal, background = 0, 0

for i in range(io.get_n_entries()):
    io.read_entry(i)
    particle = larcv.EventParticle.to_particle(io.get_data("particle", "label"))
    if particle.as_vector().size() is not 1:
      print('Got a problem here, more than one particle per event?')
    if particle.as_vector().front().pdg_code() > 0.5:
      signal += 1
    else:
      background += 1

    if i % 100 == 0:
        print("On entry ", i, " of ", io.get_n_entries())

    # if i > 10000:
        # break

print ('Number of signal events =', signal)
print ('Number of background events =', background)

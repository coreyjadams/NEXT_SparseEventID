from . import trainer_eventID, trainier_cycleGAN


# Joint training for cycleGAN + EventID

class trainer_joint(object):

    def __init__(self, args):
        self._eventID  = trainer_eventID.trainer_eventID(args)
        self._cycleGAN = trainier_cycleGAN.trainier_cycleGAN(args)


    def initialize_io(self):
        self._eventID.initialize_io()
        self._cycleGAN.initialize_io()

    def init_network(self):
        self._eventID.init_network()
        self._cycleGAN.init_network()


    def init_optimizer(self):
        self._eventID.init_optimizer()
        self._cycleGAN.init_optimizer()

    def init_saver(self):
        self._eventID.init_saver()
        self._cycleGAN.init_saver()

    def initialize(self, io_only=False):

        self.initialize_io()


        if io_only:
            return


        self.init_network()

        self.init_optimizer()

        self.init_saver()

        eventID_state  = self._eventIDGAN.restore_model()
        cycleGAN_state = self._cycleGAN.restore_model()

        if cycleGAN_state is not None and eventID_state is not None:
            self._eventID.load_state(eventID_state)
            self._cycleGAN.load_state(cycleGAN_state)
        else:
            self._global_step = 0

        self._eventID.set_log_keys()
        self._cycleGAN.set_log_keys()

        self._eventID.model_to_device()
        self._cycleGAN.model_to_device()



    
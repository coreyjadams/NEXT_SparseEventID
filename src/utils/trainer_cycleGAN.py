import datetime
import itertools

import numpy
import torch

from .trainercore import trainercore

from ..networks import generator, discriminator

class trainer_cycleGAN(trainercore):

    def __init__(self, args):
        trainercore.__init__(self, args)
        self._rank = 0

        # This variable controls printouts from this class.
        # In a distributed case it's set to false except on one rank
        self._print = True

    def set_log_keys(self):
        self._log_keys = ['generator_loss', 'discriminator_loss']


    def initialize_io(self):

        if self.args.training:
            self._data_epoch_size = self.larcv_fetcher.prepare_cycleGAN_sample(
                "real", self.args.data_file, self.args.minibatch_size)
            self._sim_epoch_size = self.larcv_fetcher.prepare_cycleGAN_sample(
                "sim", self.args.sim_file, self.args.minibatch_size)

    def init_network(self):

        self.init_generators()

        if self._print:
            n_trainable_parameters = 0
            for var in self.generators['real_to_sim'].parameters():
                n_trainable_parameters += numpy.prod(var.shape)
            print("Total number of trainable parameters per generator: {}".format(n_trainable_parameters))

        self.init_discriminators()

        if self._print:
            n_trainable_parameters = 0
            for var in self.discriminators['real_data'].parameters():
                n_trainable_parameters += numpy.prod(var.shape)
            print("Total number of trainable parameters per discriminator: {}".format(n_trainable_parameters))

        self.generator_criterion     = torch.nn.L1Loss()
        self.discriminator_criterion = torch.nn.MSELoss()

    def init_generators(self):

        self.generators = {}

        self.generators['real_to_sim'] = generator.ResidualGenerator(self.args)
        self.generators['sim_to_real'] = generator.ResidualGenerator(self.args)

    def init_discriminators(self):

        self.discriminators = {}

        self.discriminators['real_data'] = discriminator.Discriminator(self.args)
        self.discriminators['sim_data'] = discriminator.Discriminator(self.args)




    def init_optimizer(self):

        # Create an optimizer:
        if self.args.optimizer.lower() == "adam":

            self.generator_opt = torch.optim.Adam(
                        itertools.chain(self.generators['real_to_sim'].parameters(),
                                        self.generators['sim_to_real'].parameters()),
                        lr=self.args.learning_rate,
                        weight_decay=self.args.weight_decay
                      )

            self.discriminator_opt = torch.optim.Adam(
                        itertools.chain(self.discriminators['real_data'].parameters(),
                                        self.discriminators['sim_data'].parameters()),
                        lr=self.args.learning_rate,
                        weight_decay=self.args.weight_decay
                      )

        else:
            self.generator_opt = torch.optim.RMSprop(
                        itertools.chain(self.generators['real_to_sim'].parameters(),
                                        self.generators['sim_to_real'].parameters()),
                        lr=self.args.learning_rate,
                        weight_decay=self.args.weight_decay
                      )

            self.discriminator_opt = torch.optim.RMSprop(
                        itertools.chain(self.discriminators['real_data'].parameters(),
                                        self.discriminators['sim_data'].parameters()),
                        lr=self.args.learning_rate,
                        weight_decay=self.args.weight_decay
                      )



        device = self.get_device()




    def get_model_save_dict(self):
        '''Return the save dict for the current models

        Expected to vary between cycleGAN and eventID
        '''

                # save the model state(s) into the file path:
        state_dict = {
            'gen_real_to_sim' : self.generators['real_to_sim'].state_dict(),
            'gen_sim_to_real' : self.generators['sim_to_real'].state_dict(),
            'disc_real_data'  : self.discriminators['real_data'].state_dict(),
            'disc_sim_data'   : self.discriminators['sim_data'].state_dict(),
            'global_step'     : self._global_step,
            'gen_optimizer'   : self.generator_opt.state_dict(),
            'disc_optimizer'  : self.discriminator_opt.state_dict(),
        }

        return state_dict

    def load_state(self, state):
        self.generators['real_to_sim'].load_state_dict(state['gen_real_to_sim'])
        self.generators['sim_to_real'].load_state_dict(state['gen_sim_to_real'])
        self.discriminators['real_data'].load_state_dict(state['disc_real_data'])
        self.discriminators['sim_data'].load_state_dict(state['disc_sim_data'])
        self.generator_opt.load_state_dict(state['gen_optimizer'])
        self.discriminator_opt.load_state_dict(state['disc_optimizer'])
        self._global_step = state['global_step']

        # If using GPUs, move the model to GPU:
        if self.args.compute_mode == "GPU":
            for state in self.generator_opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

            for state in self.discriminator_opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        return True

    def model_to_device(self):

        if self.args.compute_mode == "CPU":
            pass
        if self.args.compute_mode == "GPU":
            for key in self.generators:     self.generators[key].cuda()
            for key in self.discriminators: self.discriminators[key].cuda()


    def _calculate_accuracy(self, logits, minibatch_data):
        ''' Calculate the accuracy.

        '''

        # Compare how often the input label and the output prediction agree:

        # Compute the accuracy of the discriminator:

        # We want to know how often each discriminator

        # return accuracy
        return

    def _compute_metrics(self, discriminator_score, loss):

        # Call all of the functions in the metrics dictionary:
        metrics = {}

        for key in loss:
            metrics[key]     = loss[key].data
        # accuracy = self._calculate_accuracy(discriminator_score)
        # metrics['accuracy'] = accuracy

        return metrics

    def compute_cycle_loss(self, original, cycled):

        cycle_loss = self.generator_criterion(original,cycled)

        return cycle_loss


    def compute_generator_loss(self, fake_score):
        # This is not the "traditional" GAN loss but rather
        # The "more stable" version from the cycle gan paper

        generator_loss = self.discriminator_criterion(fake_score, torch.ones_like(fake_score))

        return generator_loss

    def compute_discriminator_loss(self, real_score, fake_score):
        # This is not the "traditional" GAN loss but rather
        # The "more stable" version from the cycle gan paper
        discriminator_loss  = self.discriminator_criterion(real_score, torch.ones_like(real_score))
        discriminator_loss += self.discriminator_criterion(fake_score, torch.zeros_like(real_score))

        return discriminator_loss

    def train_step(self):



        # For a train step, we fetch data, run a forward and backward pass, and
        # if this is a logging step, we compute some logging metrics.

        # Make sure all nets are trainable:
        for key in self.generators:     self.generators[key].train()
        for key in self.discriminators: self.discriminators[key].train()

        global_start_time = datetime.datetime.now()

        # Reset the gradient values for this step:
        self.generator_opt.zero_grad()
        self.discriminator_opt.zero_grad()

        # Fetch the next batch of data with larcv
        io_start_time = datetime.datetime.now()
        real_minibatch_data = self.larcv_fetcher.fetch_next_cycleGAN_batch("real")
        sim_minibatch_data  = self.larcv_fetcher.fetch_next_cycleGAN_batch("sim")
        io_end_time = datetime.datetime.now()

        real_minibatch_data = self.larcv_fetcher.to_torch_cycleGAN(real_minibatch_data)
        sim_minibatch_data  = self.larcv_fetcher.to_torch_cycleGAN(sim_minibatch_data)

        # print("real_minibatch_data['image'].shape: ", real_minibatch_data['image'].shape)
        # print("sim_minibatch_data['image'].shape: ",  sim_minibatch_data['image'].shape)

        # Cycle gan has a complicated training sequence compared to most networks.

        # First, run forward on the real data to generate simulation:
        fake_simulation = self.generators['real_to_sim'](real_minibatch_data['image'])

        # And, run forward on the fake simulation to get back to real data:
        cycled_real_data = self.generators['sim_to_real'](fake_simulation)
        # print("cycled_real_data.shape: ", cycled_real_data.shape)

        # print("Cycled real images")
        # Next, run forward on the simulated data to generate real data:
        fake_real_data = self.generators['sim_to_real'](sim_minibatch_data['image'])

        # And, run forward on the fake data to get back to simulation:
        cycled_simulation = self.generators['real_to_sim'](fake_real_data)
        # print("cycled_simulation.shape: ", cycled_simulation.shape)

        # print("Cycled fake images")


        # Now, all networks have done their forward pass.
        # We need to compute the adversarial losses:

        discriminator_score = {}
        discriminator_score['real_data'] = self.discriminators['real_data'](real_minibatch_data['image'])
        discriminator_score['fake_data'] = self.discriminators['real_data'](fake_real_data)

        discriminator_score['real_sim']  = self.discriminators['sim_data'](sim_minibatch_data['image'])
        discriminator_score['fake_sim']  = self.discriminators['sim_data'](fake_simulation)


        # Next, we begin computing loss values:
        loss = {}

        # We don't need

        # Compute the cycle loss between the two categories:
        loss['real_cycle_loss'] = self.compute_cycle_loss(real_minibatch_data['image'], cycled_real_data)
        loss['sim_cycle_loss']  = self.compute_cycle_loss(sim_minibatch_data['image'],  cycled_simulation)

        loss['total_cycle_loss'] = loss['real_cycle_loss'] + loss['sim_cycle_loss']

        # We next compute the GAN loss for the generators:
        loss['gan_real_to_sim'] = self.compute_generator_loss(discriminator_score['fake_sim'])
        loss['gan_sim_to_real'] = self.compute_generator_loss(discriminator_score['fake_data'])
        loss['gan_loss'] = loss['gan_real_to_sim'] + loss['gan_sim_to_real']

        # Compute the gradients for the network parameters:
        loss['generator_loss'] = self.args.cycle_lambda *loss['total_cycle_loss'] + loss['gan_loss']
        loss['generator_loss'].backward(retain_graph=True)


        # Next, compute GAN Loss values for the discriminator:
        loss['discriminator_real_data'] = self.compute_discriminator_loss(discriminator_score['real_data'], discriminator_score['fake_data'])
        loss['discriminator_sim_data']  = self.compute_discriminator_loss(discriminator_score['real_sim'], discriminator_score['fake_sim'])
        loss['discriminator_loss'] = loss['discriminator_sim_data'] + loss['discriminator_real_data']

        loss['discriminator_loss'].backward()
        # print("Completed backward pass")

        # Compute any necessary metrics:
        metrics = self._compute_metrics(discriminator_score, loss)



        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = self.args.minibatch_size / self._seconds_per_global_step
        except:
            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0

        metrics['io_fetch_time'] = (io_end_time - io_start_time).total_seconds()

        # print("Calculated metrics")


        step_start_time = datetime.datetime.now()
        # Apply the parameter update:
        self.generator_opt.step()
        self.discriminator_opt.step()
        # print("Updated Weights")
        global_end_time = datetime.datetime.now()

        metrics['step_time'] = (global_end_time - step_start_time).total_seconds()


        self.log(metrics, saver="train")

        # print("Completed Log")

        self.summary(metrics, saver="train")

        # print("Summarized")

        # Compute global step per second:
        self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()

        # Increment the global step value:
        self.increment_global_step()



        return metrics

    def val_step(self, n_iterations=1):

        # First, validation only occurs on training:
        if not self.args.training: return

        pass

    def ana_step(self, iteration=None):

        pass

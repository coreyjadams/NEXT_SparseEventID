import datetime
import numpy

import torch
import tensorboardX

from .trainercore import trainercore
from ..networks import sparseresnet3d

class trainer_eventID(trainercore):

    def __init__(self, args):
        trainercore.__init__(self, args)
        self._rank = 0

        # This variable controls printouts from this class.
        # In a distributed case it's set to false except on one rank
        self._print = True

    def set_log_keys(self):
        self._log_keys = ['loss', 'accuracy']


    def initialize_io(self):

        if self.args.training:
            self._epoch_size = self.larcv_fetcher.prepare_eventID_sample("train", self.args.file, self.args.minibatch_size)

            if self.args.aux_file is not None:
                self._epoch_size = self.larcv_fetcher.prepare_eventID_sample("test", self.args.aux_file, self.args.minibatch_size)

    def init_network(self):

        self._net = sparseresnet3d.ResNet(self.args)


        if self.args.training:
            self._net.train(True)


        if self._print:
            n_trainable_parameters = 0
            for var in self._net.parameters():
                n_trainable_parameters += numpy.prod(var.shape)
            print("Total number of trainable parameters in this network: {}".format(n_trainable_parameters))

    def init_saver(self):

        trainercore.init_saver(self)

        if self.args.aux_file is not None and self.args.training:
            self._aux_saver = tensorboardX.SummaryWriter(self.args.log_directory + "/test/")
        elif self.args.aux_file is not None and not self.args.training:
            self._aux_saver = tensorboardX.SummaryWriter(self.args.log_directory + "/val/")

        else:
            self._aux_saver = None



    def init_optimizer(self):

        # Create an optimizer:
        if self.args.optimizer == "SDG":
            self._opt = torch.optim.SGD(self._net.parameters(), lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay)
        else:
            self._opt = torch.optim.Adam(self._net.parameters(), lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay)





        device = self.get_device()
        #
        # if self.args.weight_sig is None or self.args.weight_bkg is None:
        #     print ('You have requested to balance loss but have not set the weight for signal of background. I will ignore your request.')
        #     self.args.balance_loss = False

        #
        # if self.args.balance_loss:
        #
        #     weights = [self.args.weight_bkg, self.args.weight_sig]
        #     class_weights = torch.tensor(weights, device=device)
        #     self._criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        #
        # else:


        # For the criterion, get the relative ratios of the classes:
        all_labels = self.larcv_fetcher.eventID_labels('train')
        labels,counts = numpy.unique(all_labels, return_counts=True)
        print(counts)
        weights = (numpy.sum(counts) - counts) / numpy.sum(counts)
        print(weights)
        weights = 2 * torch.tensor(weights, device=device).float()

        self._criterion = torch.nn.CrossEntropyLoss(weights)


    def get_model_save_dict(self):
        '''Return the save dict for the current models

        Expected to vary between cycleGAN and eventID
        '''

                # save the model state(s) into the file path:
        state_dict = {
            'global_step' : self._global_step,
            'state_dict'  : self._net.state_dict(),
            'optimizer'   : self._opt.state_dict(),
        }

        return state_dict

    def load_state(self, state):

        self._net.load_state_dict(state['state_dict'])
        self._opt.load_state_dict(state['optimizer'])
        self._global_step = state['global_step']

        # If using GPUs, move the model to GPU:
        if self.args.compute_mode == "GPU":
            for state in self._opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        return True

    def model_to_device(self):

        if self.args.compute_mode == "CPU":
            pass
        if self.args.compute_mode == "GPU":
            self._net.cuda()
            # for state in self._opt.state.values:
            #     for k, v in state.items():
            #         if torch.is_tensor(v):
            #             state[k] = v.cuda()
            # This moves the optimizer to the GPU:


    def _calculate_loss(self, inputs, logits):
        ''' Calculate the loss.

        returns a single scalar for the optimizer to use.
        '''
        pass


        values, target = torch.max(inputs['label'], dim = 1)
        loss = self._criterion(logits, target=target)
        return loss


    def _calculate_accuracy(self, logits, minibatch_data):
        ''' Calculate the accuracy.

        '''

        # Compare how often the input label and the output prediction agree:


        values, indices = torch.max(minibatch_data['label'], dim = 1)
        values, predict = torch.max(logits, dim=1)
        correct_prediction = torch.eq(predict,indices)
        accuracy = torch.mean(correct_prediction.float())

        return accuracy

    def _compute_metrics(self, logits, minibatch_data, loss):

        # Call all of the functions in the metrics dictionary:
        metrics = {}

        metrics['loss']     = loss.data
        accuracy = self._calculate_accuracy(logits, minibatch_data)
        metrics['accuracy'] = accuracy

        return metrics



    def train_step(self):



        # For a train step, we fetch data, run a forward and backward pass, and
        # if this is a logging step, we compute some logging metrics.

        self._net.train()

        global_start_time = datetime.datetime.now()

        # Reset the gradient values for this step:
        self._opt.zero_grad()

        # Fetch the next batch of data with larcv
        io_start_time = datetime.datetime.now()
        minibatch_data = self.larcv_fetcher.fetch_next_eventID_batch("train")
        io_end_time = datetime.datetime.now()

        minibatch_data = self.larcv_fetcher.to_torch_eventID(minibatch_data)


        # Run a forward pass of the model on the input image:
        logits = self._net(minibatch_data['image'])

        # print("Completed Forward pass")
        # Compute the loss based on the logits
        loss = self._calculate_loss(minibatch_data, logits)
        # print("Completed loss")

        # Compute the gradients for the network parameters:
        loss.backward()
        # print("Completed backward pass")

        # Compute any necessary metrics:
        metrics = self._compute_metrics(logits, minibatch_data, loss)



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
        self._opt.step()
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

        # Second, validation can not occur without a validation dataloader.
        if self.args.aux_file is None: return

        # perform a validation step
        # Validation steps can optionally accumulate over several minibatches, to
        # fit onto a gpu or other accelerator

        # self._net.eval()

        with torch.no_grad():

            if self._global_step != 0 and self._global_step % self.args.aux_iteration == 0:


                # Fetch the next batch of data with larcv
                # (Make sure to pull from the validation set)
                minibatch_data = self.larcv_fetcher.fetch_next_eventID_batch('test')


                # Convert the input data to torch tensors
                minibatch_data = self.larcv_fetcher.to_torch_eventID(minibatch_data)

                # Run a forward pass of the model on the input image:
                logits = self._net(minibatch_data['image'])

                # # Here, we have to map the logit keys to aux keys
                # for key in logits.keys():
                #     new_key = 'aux_' + key
                #     logits[new_key] = logits.pop(key)



                # Compute the loss
                loss = self._calculate_loss(minibatch_data, logits)

                # Compute the metrics for this iteration:
                metrics = self._compute_metrics(logits, minibatch_data, loss)


                self.log(metrics, saver="test")
                self.summary(metrics, saver="test")

                return metrics

    def ana_step(self, iteration=None):

        # First, validation only occurs on training:
        if self.args.training: return

        # perform a validation step

        # Set network to eval mode
        self._net.eval()
        # self._net.train()

        # Fetch the next batch of data with larcv
        minibatch_data = self.fetch_next_batch(metadata=True)


        # Convert the input data to torch tensors
        minibatch_data = self.to_torch(minibatch_data)

        # Run a forward pass of the model on the input image:
        with torch.no_grad():
            logits = self._net(minibatch_data['image'])

        if self.args.LABEL_MODE == 'all':
            softmax = torch.nn.Softmax(dim=-1)(logits)
        else:
            softmax = { key : torch.nn.Softmax(dim=-1)(logits[key]) for key in logits }

        # print('label_neut', minibatch_data['label_neut'])
        # print('label_npi', minibatch_data['label_npi'])
        # print('label_cpi', minibatch_data['label_cpi'])
        # print('label_prot', minibatch_data['label_prot'])
        # print(softmax)

        # Call the larcv interface to write data:
        if self.args.OUTPUT_FILE is not None:
            if self.args.LABEL_MODE == 'all':
                writable_logits = numpy.asarray(softmax.cpu())
                self._larcv_interface.write_output(data=writable_logits[0], datatype='meta', producer='all',
                    entries=minibatch_data['entries'], event_ids=minibatch_data['event_ids'])
            else:
                for key in softmax:
                    writable_logits = numpy.asarray(softmax[key].cpu())
                    self._larcv_interface.write_output(data=writable_logits[0], datatype='meta', producer=key,
                        entries=minibatch_data['entries'], event_ids=minibatch_data['event_ids'])

        # If the input data has labels available, compute the metrics:
        if (self.args.LABEL_MODE == 'all' and 'label' in minibatch_data) or \
           (self.args.LABEL_MODE == 'split' and 'label_neut' in minibatch_data):
            # Compute the loss
            loss = self._calculate_loss(minibatch_data, logits)

            # Compute the metrics for this iteration:
            metrics = self._compute_metrics(logits, minibatch_data, loss)

            if iteration is not None:
                metrics.update({'it.' : iteration})


            self.log(metrics, saver="test")
            # self.summary(metrics, saver="test")

            return metrics

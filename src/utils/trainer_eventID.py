import time, datetime, os
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

        # This is for inference to keep track, at the end of an epoch say,
        # The performance on data

        self.inference_results = {}


    def set_log_keys(self):
        self._log_keys = ['loss', 'accuracy']


    def initialize_io(self):

        self._epoch_size = {}

        if self.args.training:
            self._epoch_size["train"] = self.larcv_fetcher.prepare_eventID_sample(
                "train", self.args.train_file, self.args.minibatch_size)

            if self.args.test_file is not None:
                self._epoch_size["test"] = self.larcv_fetcher.prepare_eventID_sample(
                    "test", self.args.test_file, self.args.minibatch_size)
        else:
            # Inference mode
            if self.args.sim_file is not None and self.args.sim_file != "none":
                self._epoch_size["sim"] = self.larcv_fetcher.prepare_eventID_sample(
                    "sim", self.args.sim_file, self.args.minibatch_size)

            if self.args.data_file is not None and self.args.data_file != "none":
                self._epoch_size["data"] = self.larcv_fetcher.prepare_eventID_sample(
                    "data", self.args.data_file, self.args.minibatch_size)


    def init_network(self):

        self._net = sparseresnet3d.ResNet(self.args)


        if self.args.training:
            self._net.train(True)


        if self._print:
            n_trainable_parameters = 0
            for var in self._net.parameters():
                n_trainable_parameters += numpy.prod(var.shape)
            self.print("Total number of trainable parameters in this network: {}".format(n_trainable_parameters))

    def init_saver(self):

        trainercore.init_saver(self)

        if self.args.training and self.args.test_file is not None:
            self._aux_saver = tensorboardX.SummaryWriter(self.args.log_directory + "/test/")

        else:
            self._aux_saver = None



    def init_optimizer(self):

        if not self.args.training:
            return

        # get the initial learning_rate:
        initial_learning_rate = self.lr_calculator(self._global_step)

        # IMPORTANT: the scheduler in torch is a multiplicative factor,
        # but I've written it as learning rate itself.  So set the LR to 1.0

        # Create an optimizer:
        if self.args.optimizer == "SDG":
            self._opt = torch.optim.SGD(self._net.parameters(), lr=1.0,
                weight_decay=self.args.weight_decay)
        else:
            self._opt = torch.optim.Adam(self._net.parameters(), lr=1.0,
                weight_decay=self.args.weight_decay)



        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self._opt, self.lr_calculator, last_epoch=-1)




        device = self.get_device()

        # For the criterion, get the relative ratios of the classes:
        all_labels = self.larcv_fetcher.eventID_labels('train')
        labels,counts = numpy.unique(all_labels, return_counts=True)
        weights = (numpy.sum(counts) - counts) / numpy.sum(counts)
        weights = 2 * torch.tensor(weights, device=device).float()

        self.print("Computed weights as ", weights)

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
            'scheduler'   : self.lr_scheduler.state_dict(),
        }

        return state_dict

    def load_state(self, state):

        self._net.load_state_dict(state['state_dict'])
        if self.args.training:
            self._opt.load_state_dict(state['optimizer'])
            self.lr_scheduler.load_state_dict(state['scheduler'])
            self._global_step = state['global_step']

        # If using GPUs, move the model to GPU:
        if self.args.compute_mode == "GPU":
            if self.args.training:
                for state in self._opt.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

        return True

    def to_device(self):

        if self.args.compute_mode == "CPU":
            pass
        if self.args.compute_mode == "GPU":
            self._net.cuda()


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

        # print(self._opt.device)
        step_start_time = datetime.datetime.now()
        # Apply the parameter update:
        self._opt.step()
        # print("Updated Weights")
        global_end_time = datetime.datetime.now()

        self.lr_scheduler.step()


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
        if self.args.test_file is None: return

        # perform a validation step
        # Validation steps can optionally accumulate over several minibatches, to
        # fit onto a gpu or other accelerator

        self._net.eval()

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

    def sum_over_images(self, images):

        if type(images) is tuple:
            locations = images[0][:,0:3]
            batch     = images[0][:,3]
            values    = images[1]
            n_images = len(torch.unique(batch))
            energy = torch.zeros(size=(n_images,)).float()
            for b in range(n_images):
                energy[b] = torch.sum(values[batch == b])
            # indexes = torch.
        else:
            pass

        return energy

    def ana_epoch(self, target):

        if target not in self._epoch_size:
            return
        self.print("Starting")
        self.inference_results[target] = {
            'energy'  : None,
            'label'   : None,
            'pred'    : None,
            'entries' : None,
            'softmax' : None,
            'run'     : None,
            'subrun'  : None,
            'event'   : None,
        }

        n_events = self._epoch_size[target]

        n_processed = 0
        self.print(n_events)
        start = time.time()

        self._net.eval()


        while n_processed < n_events:


            # perform a validation step

            # Set network to eval mode
            self._net.eval()
            # self._net.train()

            # Fetch the next batch of data with larcv
            io_start_time = datetime.datetime.now()
            minibatch_data = self.larcv_fetcher.fetch_next_eventID_batch(target)
            io_end_time = datetime.datetime.now()


            # Convert the input data to torch tensors
            minibatch_data = self.larcv_fetcher.to_torch_eventID(minibatch_data)

            # Run a forward pass of the model on the input image:
            with torch.no_grad():
                logits = self._net(minibatch_data['image'])
                softmax = torch.softmax(logits, dim=-1)

            prediction = torch.argmax(logits, dim=-1)

            # What is the energy of each event?
            # Sum the voxels:
            energy = self.sum_over_images(minibatch_data['image'])

            keys = ["energy", "pred", "softmax"]

            if "label" in minibatch_data:
                label      = torch.argmax(minibatch_data['label'], dim=-1)
                keys.append("label")

            if len(energy) != self.args.minibatch_size:
                # Found an event that is "too short"
                self.print("size mismatch!")

            for key in keys:
                if key == "energy":  tensor = energy
                if key == "label":   tensor = label
                if key == "softmax": tensor = softmax
                if key == "pred":    tensor = prediction


                if self.inference_results[target][key] is None:
                    self.inference_results[target][key] = tensor
                else:
                    self.inference_results[target][key] = torch.cat([self.inference_results[target][key], tensor])


            run = numpy.asarray([ minibatch_data['event_ids'][i].run() for i in range(self.args.minibatch_size)])
            subrun = numpy.asarray([ minibatch_data['event_ids'][i].subrun() for i in range(self.args.minibatch_size)])
            event = numpy.asarray([ minibatch_data['event_ids'][i].event() for i in range(self.args.minibatch_size)])

            if self.inference_results[target]['entries'] is None:
                self.inference_results[target]['entries'] = minibatch_data['entries']
            else:
                self.inference_results[target]['entries'] = numpy.concatenate([self.inference_results[target]['entries'], minibatch_data['entries']])


            if self.inference_results[target]['run'] is None:
                self.inference_results[target]['run'] = run
            else:
                self.inference_results[target]['run'] = numpy.concatenate([self.inference_results[target]['run'], run])

            if self.inference_results[target]['subrun'] is None:
                self.inference_results[target]['subrun'] = subrun
            else:
                self.inference_results[target]['subrun'] = numpy.concatenate([self.inference_results[target]['subrun'], subrun])

            if self.inference_results[target]['event'] is None:
                self.inference_results[target]['event'] = event
            else:
                self.inference_results[target]['event'] = numpy.concatenate([self.inference_results[target]['event'], event])


            n_processed += self.args.minibatch_size

        end = time.time()
        self.print (f"Processed {n_processed} images at {n_processed / (end-start):.2f} Img/s")



        # Save the inference results:
        for key in self.inference_results[target].keys():
            if key not in keys:
                continue

            if key != "entries":
                self.inference_results[target][key] = self.inference_results[target][key].cpu().numpy()
            # # If the batch size doesn't divide the dataset, it will get reused a little.
            # # This clips that
            # self.print(key, self.inference_results[target][key].shape)
            # self.inference_results[target][key] = self.inference_results[target][key][0:n_events]
            # print(key, self.inference_results[target][key].shape)


        if target == "sim":
            output_filename = os.path.basename(self.args.sim_file).replace(".h5", "_inference_results")
        if target == "data":
            output_filename = os.path.basename(self.args.data_file).replace(".h5", "_inference_results")

        print(output_filename)

        numpy.save(output_filename,
                list(self.inference_results[target].items()),
                list(self.inference_results[target].keys()))


    def make_analysis(self):
        if self.inference_results is None:
            raise(Exception)


        min_bin = 1500
        max_bin = 2500
        n_bins  = 50

        # if "sim" in self.inference_results:

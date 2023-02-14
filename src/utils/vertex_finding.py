import torch
import pytorch_lightning as pl

# torch.set_float32_matmul_precision('high')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import datetime

from . training_utils import init_optimizer, format_log_message

from src import logging

logger = logging.getLogger("NEXT")

class lightning_trainer(pl.LightningModule):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args, encoder, head, transforms,
                 image_meta,
                 image_key   = "pmaps",
                 lr_scheduler=None): 
        super().__init__()

        self.args         = args
        self.transforms   = transforms
        self.encoder      = encoder
        self.head         = head
        self.image_key    = image_key
        self.lr_scheduler = lr_scheduler

        self.image_size   = torch.tensor(image_meta['size'][0])
        self.image_origin = torch.tensor(image_meta['origin'][0])

        self.log_keys = ["loss"]


    def forward(self, batch):


        t = self.transforms[0](batch)

        representation = self.encoder(t)

        logits = self.head(representation)


        return logits


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.

        image = batch[self.image_key]

        prediction = self(image)

        reference_shape = prediction.shape[2:]
        vertex_labels = self.compute_vertex_labels(batch, reference_shape)

        anchor_loss, regression_loss = self.vertex_loss(vertex_labels, prediction)

        loss = anchor_loss + 0.1*regression_loss

        prediction_dict = self.predict_event(prediction)

        accuracy_dict = self.calculate_accuracy(prediction_dict, batch, vertex_labels)

        metrics = {
            'loss/loss' : loss,
            'loss/anchor_loss' : anchor_loss,
            'loss/regression_loss' : regression_loss,
            'opt/lr' : self.optimizers().state_dict()['param_groups'][0]['lr']
        }


        metrics.update(accuracy_dict)

        # self.log()
        self.print_log(metrics, mode="train")
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):

        image = batch[self.image_key]

        prediction = self(image)

        reference_shape = prediction.shape[2:]
        vertex_labels = self.compute_vertex_labels(batch, reference_shape)

        anchor_loss, regression_loss = self.vertex_loss(vertex_labels, prediction)

        loss = anchor_loss + 0.1*regression_loss

        prediction_dict = self.predict_event(prediction)

        accuracy_dict = self.calculate_accuracy(prediction_dict, batch, vertex_labels)

        metrics = {
            'loss/loss' : loss,
            'loss/anchor_loss' : anchor_loss,
            'loss/regression_loss' : regression_loss,
            'opt/lr' : self.optimizers().state_dict()['param_groups'][0]['lr']
        }


        metrics.update(accuracy_dict)

        # self.log()
        self.print_log(metrics, mode="val")
        self.log_dict(metrics)
        return



    def print_log(self, metrics, mode=""):

        if self.global_step % self.args.mode.logging_iteration == 0:

            message = format_log_message(
                log_keys = self.log_keys, 
                metrics  = metrics, 
                batch_size = self.args.run.minibatch_size, 
                global_step = self.global_step, 
                mode = mode
            )

            logger.info(message)

    def exit(self):
        pass

    def unravel_index(self,index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))

    def predict_event(self, prediction):
        
        batch_size = prediction.shape[0]
        image_size = prediction.shape[2:]

        class_prediction = prediction[:,0]
        regression_prediction = prediction[:,1:]


        max_val, max_index = torch.max(
            torch.reshape(class_prediction, (batch_size, -1)), dim=1)


        # Declare it a signal event if it's more than 0.5 for a vertex:
        signal = max_val.to(torch.float32)


        prediction_dict = {
            "label" : signal,
            "class" : class_prediction,
        }

        # Take the maximum location and use that to infer the vertex prediction.
        # Need to unravel the index since it's flattened ...
        vertex_anchor = self.unravel_index(max_index, image_size)
        batch_index = torch.arange(batch_size)
        selected_boxes = regression_prediction[batch_index,:,vertex_anchor[0], vertex_anchor[1],vertex_anchor[2]]

        # Convert the selected anchors + regression coordinates into a vertex in 3D:
        vertex_anchor = torch.stack(vertex_anchor,dim=1)
        vertex = self.image_origin + (vertex_anchor+selected_boxes)*self.anchor_size 

        prediction_dict['vertex'] = vertex

        return prediction_dict

    def calculate_accuracy(self, prediction, batch, vertex_labels):

        label_prediction = (prediction['label'] < 0.5).to(torch.float32)

        event_accuracy = (label_prediction == batch['label']).to(torch.float32)

        # Compute the label false positive rate:
        # label prediction has 1 == positive vertex
        has_vertex = batch['label'] == 0

        false_postive  = torch.mean(event_accuracy[has_vertex])
        false_negative = torch.mean(event_accuracy[torch.logical_not(has_vertex)])

        # Reduce the global event accuracy:
        event_accuracy = torch.mean(event_accuracy,dim=0)


        vertex_truth = batch["vertex"]
        vertex_pred  = prediction["vertex"]

        vertex_resolution = has_vertex.reshape((-1,1)) * ( vertex_pred - vertex_truth)

        n_vertex = torch.sum(has_vertex)
        vertex_resolution = torch.sum(vertex_resolution, dim=0) / (n_vertex - 0.0001)

        # IF there is a vertex, what fraction of the time is it detected?
        batch_size = has_vertex.shape[0]
        _, true_vertex_loc = torch.max(vertex_labels["class"].reshape((batch_size,-1)), dim=1)
        _, pred_vertex_loc = torch.max(prediction   ["class"].reshape((batch_size,-1)), dim=1)

        vertex_anchor_acc = true_vertex_loc == pred_vertex_loc
        vertex_anchor_acc = torch.sum(has_vertex * vertex_anchor_acc.to(torch.float32)) 
        vertex_anchor_acc = vertex_anchor_acc / (torch.sum(has_vertex) + 0.0001)



        return {
            "acc/label"  : event_accuracy,
            "acc/vertex_decection" : vertex_anchor_acc,
            "acc/vertex_x" : vertex_resolution[0] ,
            "acc/vertex_y" : vertex_resolution[1] ,
            "acc/vertex_z" : vertex_resolution[2] ,
            "acc/vertex"   : torch.sqrt(torch.sum(vertex_resolution**2)),
            "acc/false_pos": false_postive,
            "acc/false_neg": false_negative,
        }


    def compute_vertex_labels(self, batch, reference_shape):

        # Basic properties infered from the input:
        vertex_label = batch['vertex']
        target_device = vertex_label.device
        batch_size = vertex_label.shape[0]

        self.image_size   = self.image_size.to(target_device, dtype=torch.float32)
        self.image_origin = self.image_origin.to(target_device, dtype=torch.float32)

        # the size of each anchor box can be found first:
        if not hasattr(self, "anchor_size"):
            labels_spatial_size = torch.tensor(reference_shape).to(target_device)
            self.anchor_size = torch.tensor(self.image_size / labels_spatial_size)

        class_shape = (batch_size,) + reference_shape
        regression_shape = (batch_size, 3,) + reference_shape

        # Start with the reference shape to define the labels
        class_labels = torch.zeros(size = class_shape, device=target_device, dtype=torch.float32)
        regression_labels = vertex_label.to(target_device)

        # The first goal is to figure out which binary labels to turn on, if any.
        # We take the 3D vertex location and figure out which pixel it aligns with in X/Y/Z.

        # Identify where the vertex is with the 0,0,0 index at 0,0,0 coordinate::
        relative_vertex = regression_labels - self.image_origin


        # Map the relative location onto the anchor grid
        anchor_index = (relative_vertex // self.anchor_size).to(torch.int64)

        active_anchors = vertex_label[:,2] != 0.0

        # Select the target anchor indexes:
        target_anchors = anchor_index[active_anchors]

        # Set the class labels
        class_labels[active_anchors,target_anchors[:,0],target_anchors[:,1],target_anchors[:,2]] = 1.

        # We need to map the regression label to a relative point in the anchor window.
        anchor_start_point = self.anchor_size * anchor_index + self.image_origin
        regression_labels = (regression_labels - anchor_start_point) / self.anchor_size

        regression_labels = regression_labels.reshape(regression_labels.shape + (1,1,1,))


        # Lastly, we do an event-wide weight based on the presence of a vertex:
        weight = 0.5*torch.ones((batch_size,), device=target_device)
        has_vertex = batch['label'] == 0
        weight[has_vertex] = 5

        return {
            "class"      : class_labels,
            "regression" : regression_labels,
            "weight"     : weight
        }

    def vertex_loss(self, vertex_labels, prediction):

        class_prediction = prediction[:,0,:,:,:]
        class_labels = vertex_labels["class"]
        # print("Class labels: ", class_labels)

        # Compute the loss:
        class_loss = torch.nn.functional.binary_cross_entropy(
            class_prediction, class_labels, reduction="none")
        # print("Class loss: ", class_loss)
        # print("Class pred: ", class_prediction)
        focus = (class_prediction - class_labels)**4
        # print("focus: ", focus)
        # Sum over images; average over batch dimensions
        batch_size = class_loss.shape[0]
        anchor_loss = torch.reshape(class_loss*focus, (batch_size,-1))
        # Sum over entire images, but not the batch:
        anchor_loss = torch.sum(anchor_loss, dim=1)

        # We want to take the mean over the batch, but weighing the two categories
        # differently.  The dataset is about 90% bkg and 10% signal.
        # We want to use weights so that w_bkg * 0.9 + w_sig * 0.1 = 1
        anchor_loss = torch.mean(vertex_labels['weight']*anchor_loss)

        # For the regression loss, it's actually easy to calculate.
        # Compute the difference between the regression point and the prediction point
        # on every anchor, and then scale by the label for that anchor (signal/bkg)

        regression_prediction = prediction[:,1:,:,:,:]

        regression_loss = (vertex_labels['regression'] - regression_prediction)**2

        target_shape = regression_loss.shape
        # Scale, but put a 1 in the shape to broadcast over x/y/z
        target_shape = (target_shape[0],1) + target_shape[2:]
        regression_loss = regression_loss * class_labels.reshape((target_shape))

        # Sum over all anchors (most are 0) and batches
        regression_loss = torch.sum(regression_loss)
        weight = torch.sum(class_labels) + 0.0001

        return anchor_loss, regression_loss / weight


    def configure_optimizers(self):
        learning_rate = 1.0
        # learning_rate = self.args.mode.optimizer.learning_rate
        opt = init_optimizer(self.args.mode.optimizer.name, self.parameters())

        lr_fn = lambda x : self.lr_scheduler[x]

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn, last_epoch=-1)

        return [opt],[{"scheduler" : lr_scheduler, "interval": "step"}]


def create_lightning_module(args, datasets, transforms=None, lr_scheduler=None, batch_keys=None):

    # Going to build up the lightning module here.

    # Take the first dataset:
    example_ds = next(iter(datasets.values()))


    image_shape = example_ds.dataset.image_size(args.data.image_key)
    image_meta = example_ds.dataset.image_meta(args.data.image_key)
    # vertex_meta = create_vertex_meta(args, example_ds.image_meta, example_ds.image_size())

    # Next, create the network:
    from src.networks.yolo_head import build_networks
    encoder, yolo_head = build_networks(args, image_shape)

    model = lightning_trainer(
        args,
        encoder,
        yolo_head,
        transforms,
        image_meta,
        args.data.image_key,
        lr_scheduler,
    )
    return model

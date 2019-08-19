from torch import nn
from copy import deepcopy
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import tqdm
import os
import numpy as np
import time


from source.storage_utils import save_to_stats_pkl_file, load_from_stats_pkl_file, \
    save_statistics, load_statistics


class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data,
                 test_data, weight_decay_coefficient, use_gpu, learning_rate, continue_from_epoch=-1):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(ExperimentBuilder, self).__init__()
        if torch.cuda.is_available() and use_gpu:  # checks whether a cuda gpu is available and whether the gpu flag is True
            self.device = torch.device('cuda')  # sets device to be cuda
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # sets the main GPU to be the one at index 0 (on multi gpu machines you can choose which one you want to use by using the relevant GPU ID)
            print("use GPU")
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU

        self.experiment_name = experiment_name
        self.model = network_model
        self.model.to(self.device)  # sends the model from the cpu to the gpu
        self.model.reset_parameters()  # re-initialize network parameters
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = optim.Adam(self.parameters(), amsgrad=False,
                                    weight_decay=weight_decay_coefficient, lr=learning_rate)
        # Generate the directory names
        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))

        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_acc = 0.

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory
            os.mkdir(self.experiment_logs)  # create the experiment log directory
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU

        if continue_from_epoch != -1:  # if continue from epoch is not -1 then
            self.best_val_model_idx, self.best_val_model_acc = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = continue_from_epoch
        else:
            self.starting_epoch = 0

    def run_train_iter(self, x, y):
        """
        Receives the inputs and targets for the model and runs a training iteration. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)
#         print("Before y: ", y)
        y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels
#         print("After y: ", y)
        x, y = torch.Tensor(x).float().to(device=self.device), torch.Tensor(y).long().to(
            device=self.device)  # send data to device as torch tensors
        out = self.model.forward(x, y)  # forward the data in the model
#         print("out: ", out)
#         loss = F.cross_entropy(input=out, target=y)  # compute loss
        loss = self.criterion(out, y)
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss

        self.optimizer.step()  # update network parameters
        _, predicted = torch.max(out.data, 1)  # get argmax of predictions
#         print("predicted: ", predicted)
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
#         print("out[:,1]: ", out[:,1])
#         print("y:", y)
#         fpr, tpr, thresholds = metrics.roc_curve(y, out[:, 1])
#         auc = metrics.auc(fpr, tpr)
#         auc = 0
        return loss.data, accuracy, predicted
#         return loss.data, accuracy

    def run_evaluation_iter(self, x, y):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.eval()  # sets the system to validation mode
        y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels
        x, y = torch.Tensor(x).float().to(device=self.device), torch.Tensor(y).long().to(
            device=self.device)  # convert data to pytorch tensors and send to the computation device
        out = self.model.forward(x, y)  # forward the data in the model
#         loss = F.cross_entropy(out, y)  # compute loss
        loss = self.criterion(out, y)
        _, predicted = torch.max(out.data, 1)  # get argmax of predictions
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
#         fpr, tpr, thresholds = metrics.roc_curve(y, out[:, 1])
#         auc = metrics.auc(fpr, tpr)
#         auc = 0
        return loss.data, accuracy, predicted
#         return loss.data, accuracy

    def run_test_iter(self, x, y):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.eval()  # sets the system to validation mode
        y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels
        x, y = torch.Tensor(x).float().to(device=self.device), torch.Tensor(y).long().to(
            device=self.device)  # convert data to pytorch tensors and send to the computation device
        out = self.model.forward(x, y)  # forward the data in the model
#         loss = F.cross_entropy(out, y)  # compute loss
        loss = self.criterion(out, y)
        _, predicted = torch.max(out.data, 1)  # get argmax of predictions
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
#         fpr, tpr, thresholds = metrics.roc_curve(y, out[:, 1])
#         auc = metrics.auc(fpr, tpr)
#         auc = 0
        return loss.data, accuracy, out.data
#         return loss.data, accuracy

    def save_model(self, model_save_dir, model_save_name, model_idx, best_validation_model_idx,
                   best_validation_model_acc):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        state = dict()
        state['network'] = self.state_dict()  # save network parameter and other variables.
        state['best_val_model_idx'] = best_validation_model_idx  # save current best val idx
        state['best_val_model_acc'] = best_validation_model_acc  # save current best val acc
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state['best_val_model_idx'], state['best_val_model_acc']

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        total_losses = {"train_acc": [], "train_loss": [], "val_acc": [],
                        "val_loss": []}  # initialize a dict to keep the per-epoch metrics
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            current_epoch_losses = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}

            with tqdm.tqdm(total=self.train_data.num_batches) as pbar_train:  # create a progress bar for training
#                 print("In Ex_builder train_data_type: ", type(self.train_data))
                for idx, (x, y) in enumerate(self.train_data):  # get data batches
                    loss, accuracy, out = self.run_train_iter(x=x, y=y)  # take a training iter step
                    current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                    current_epoch_losses["train_acc"].append(accuracy)  # add current iter acc to the train acc list
#                     current_epoch_losses["train_auc"].append(auc)  # add current iter auc to the train auc list
                    pbar_train.update(1)
                    pbar_train.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))

            with tqdm.tqdm(total=self.val_data.num_batches) as pbar_val:  # create a progress bar for validation
                for x, y in self.val_data:  # get data batches
                    loss, accuracy, out = self.run_evaluation_iter(x=x, y=y)  # run a validation iter
                    current_epoch_losses["val_loss"].append(loss)  # add current iter loss to val loss list.
                    current_epoch_losses["val_acc"].append(accuracy)  # add current iter acc to val acc lst.
#                     current_epoch_losses["val_auc"].append(auc)  # add current iter auc to the train auc list
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_val.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))
#                     print("y:", y)
#                     print("out:",out)
            val_mean_accuracy = np.mean(current_epoch_losses['val_acc'])
            if val_mean_accuracy > self.best_val_model_acc:  # if current epoch's mean val acc is greater than the saved best val acc then
                self.best_val_model_acc = val_mean_accuracy  # set the best val model acc to be current epoch's val accuracy
                self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx

            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(
                    value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=total_losses, current_epoch=i)  # save statistics to stats file.

            # load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx=epoch_idx,
                            best_validation_model_idx=self.best_val_model_idx,
                            best_validation_model_acc=self.best_val_model_acc)

        print("Generating test set evaluation metrics")
        self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx,
                        # load best validation model
                        model_save_name="train_model")
        current_epoch_losses = {"test_acc": [], "test_loss": [], "test_auc": [], "test_f1_score": []}  # initialize a statistics dict
        X_test = 0
        Y_pred = 0
        Y_target = 0
        Y_flag = 0
        with tqdm.tqdm(total=self.test_data.num_batches) as pbar_test:  # ini a progress bar
            for x, y in self.test_data:  # sample batch
                loss, accuracy, out = self.run_test_iter(x=x,
                                                          y=y)  # compute loss and accuracy by running an evaluation step
                current_epoch_losses["test_loss"].append(loss)  # save test loss
                current_epoch_losses["test_acc"].append(accuracy)  # save test accuracy
                pbar_test.update(1)  # update progress bar status
                pbar_test.set_description(
                    "loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))  # update progress bar string output
#                 print("y:", y)
#                 print("out:",out)
#                 np.concatenate(Y_pred, out)
                if Y_flag == 0: 
                    Y_flag = 1
                    X_test = x
                    Y_pred = out
                    Y_target = torch.tensor(np.argmax(y, axis=1))
                else:
#                     X_test = torch.cat((X_test, x), 0)
                    X_test = np.concatenate((X_test, x), axis=0)
                    Y_pred = torch.cat((Y_pred, out), 0)
                    Y_target = torch.cat((Y_target, torch.tensor(np.argmax(y, axis=1))), 0)
        fpr, tpr, thresholds = metrics.roc_curve(Y_target, Y_pred[:,1])
        auc = metrics.auc(fpr, tpr)
        f1_score = metrics.f1_score(Y_target, np.argmax(Y_pred, axis=1))
        current_epoch_losses["test_auc"].append(auc)
        current_epoch_losses["test_f1_score"].append(f1_score)
        test_losses = {key: [np.mean(value)] for key, value in
                       current_epoch_losses.items()}  # save test set metrics in dict format
        save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0)

        return total_losses, test_losses, Y_pred, Y_target, X_test

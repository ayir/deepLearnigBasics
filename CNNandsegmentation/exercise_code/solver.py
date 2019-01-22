from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-3,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        
        for epoch in range(num_epochs):
            train_losses = []
            for i, (images, labels) in enumerate(train_loader,1):
#                 outputs = model(images)
#                 loss = self.loss_func(outputs, labels)
#                 self.train_loss_history.append(loss.item())
                
                optim.zero_grad()
                outputs = model(images)
                loss=0
                for s in range(images.size()[0]):
                    loss += self.loss_func(outputs[s].view(23, -1).transpose(1, 0), labels[s].view(-1))
                train_losses.append(loss.data[0] / images.size()[0])
                loss.backward()
                optim.step()

                total = labels.size(0)

#                 _, predicted = torch.max(outputs.data, 1)
#                 correct = (predicted == labels).sum().item()
#                 self.train_acc_history.append(correct / total)
                _, preds = torch.max(outputs, 1)
                targets_mask = labels >= 0
                train_acc = np.mean((preds == labels)[targets_mask].data.cpu().numpy())
                
#                 print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
#                       .format(epoch + 1, num_epochs, i + 1, iter_per_epoch, loss.item(),
#                               (correct / total) * 100))
#             for i, (images, labels) in enumerate(val_loader):
#                 outputs = model(images)
#                 loss = self.loss_func(outputs, labels)
#                 self.val_loss_history.append(loss.item())
                
#                 total = labels.size(0)

#                 _, predicted = torch.max(outputs.data, 1)
#                 correct = (predicted == labels).sum().item()
#                 self.val_acc_history.append(correct / total)
               
                print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1,
                                                           num_epochs,
                                                           train_acc,
                                                           train_losses[-1]))    
            
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')

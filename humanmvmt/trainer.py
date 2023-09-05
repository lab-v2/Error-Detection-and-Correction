import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim


class Trainer:
    def __init__(self, model = None, train_dataloader = None, val_dataloader = None, saved_model_path = None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.has_mps else "cpu"))

    def fit(self, init_lr=1e-3, batch_size=64, epochs=10, saved_model_path=None):

        # Define Training Hyperparameters
        self.init_lr = init_lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_steps = len(self.train_dataloader.dataset) // batch_size
        if self.val_dataloader is not None:
            self.val_steps = len(self.val_dataloader.dataset) // batch_size

        # load the model and set it to evaluation mode
        if saved_model_path is not None:
            self.model = torch.load(saved_model_path).to(self.device)
        else:
            self.model = self.model.to(self.device)

        # Loss & Optimizer
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.parameters(), lr = self.init_lr)

        if self.val_dataloader is not None:
            history = {
                "train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": []
            }
        else:
            history = {
                "train_loss": [],
                "train_acc": [],
            }

        # measure how long training is going to take
        start_time = time.time()

        # Train Network
        for epoch in range(self.epochs):
            print(f'Epoch {epoch+1}/{self.epochs}')
            # set the model in training mode
            self.model.train()
        
            # initialize the total training and validation loss
            total_train_loss = 0
            total_val_loss = 0

            # initialize the number of correct predictions in the training
            # and validation step
            train_correct = 0
            train_new_correct = 0
            val_correct = 0

            # loop over the training set
            pbar = tqdm(self.train_dataloader, desc='Training')
            batches = 0
            count = 0
            for (x, y) in pbar:

                # send the input to the device
                (x, y) = (x.to(self.device), y.to(self.device))

                # perform a forward pass and calculate the training loss
                pred = self.model(x)
                #print("pred",pred.shape)
                #print("label:", label.shape)
                loss = criterion(pred, y)

                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # add the loss to the total training loss so far and
                # calculate the number of correct train predictions
                with torch.no_grad():
                    self.model.eval()
                    count += np.product(list(y.size()))
                    batches+=1
                    total_train_loss += loss.item()
                    pred1 = self.model(x)
                    train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    train_new_correct += (pred1.argmax(1) == y).type(torch.float).sum().item()
            
                    if batches == 1: #compute only once
                        if len(y.size())>1:
                            corrector = np.product(list(y.size())[1:])
                        else:
                            corrector = 1

                    batch_train_loss = total_train_loss/batches
                    train_accuracy = train_correct / (batch_size*batches*corrector)
                    train_new_acc = train_new_correct / count
                    pbar.set_postfix({'train_loss': batch_train_loss, 
                                      'train_acc': train_accuracy,
                                      'train_new_acc': train_new_acc})
                self.model.train()

            
            if self.val_dataloader is not None:
                # switch off autograd for evaluation
                with torch.no_grad():
                    # set the model in evaluation mode
                    self.model.eval()

                    # loop over the validation set
                    pbar_val = tqdm(self.val_dataloader, desc='Validation')
                    val_batches = 0
                    for (x, y) in pbar_val:
                        # send the input to the device
                        (x, y) = (x.to(self.device), y.to(self.device))

                        # make the predictions and calculate the validation loss
                        pred = self.model(x)
                        val_batches+=1
                        total_val_loss += criterion(pred, y).item()

                        # calculate the number of correct val predictions
                        val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()        

                        batch_val_loss = total_val_loss/batches
                        val_accuracy = train_correct / (batch_size*batches*corrector)
                        pbar.set_postfix({'train_loss': batch_val_loss, 
                                        'train_acc': val_accuracy})


            # calculate the average training and validation loss
            avg_train_loss = total_train_loss / self.train_steps
            if self.val_dataloader is not None:
                avg_val_loss = total_val_loss / self.val_steps

            train_correct = train_correct / (len(self.train_dataloader.dataset)*corrector)
            if self.val_dataloader is not None:
                val_correct = val_correct / (len(self.val_dataloader.dataset)*corrector)

            # update our training history
            history["train_loss"].append(avg_train_loss)
            history["train_acc"].append(train_correct)
            if self.val_dataloader is not None:
                history["val_loss"].append(avg_val_loss)
                history["val_acc"].append(val_correct)

            # print the model training and validation information
            if self.val_dataloader is not None:
                print("[INFO] EPOCH: {}/{} --> Train loss: {:.6f}, Train accuracy: {:.6f} --> Val loss: {:.6f}, Val accuracy: {:.4f}".\
                      format(epoch + 1, self.epochs, avg_train_loss, train_correct, avg_val_loss, val_correct))
            else:
                print("[INFO] EPOCH: {}/{} --> Train loss: {:.6f}, Train accuracy: {:.6f}".\
                      format(epoch + 1, self.epochs,avg_train_loss, train_correct))
            
        # finish measuring how long training took
        end_time = time.time()
        print("[INFO] Total Training Time: {:.2f}s".format(end_time - start_time))

        if saved_model_path:
            torch.save(self.model, saved_model_path)
        else:
            return history, self.model


    def predict(self, test_dataloader, saved_model_path = None):
        pred_list = []
        pred_out = []
        act_list = []
        cntr=0

        # load the model and set it to evaluation mode
        if saved_model_path is not None:
            self.model = torch.load(saved_model_path).to(self.device)
        
        self.model.eval()

        # switch off autograd
        with torch.no_grad():
            
            # loop over the test set
            count = 0
            train_correct = 0
            for (features, label) in test_dataloader:

                # send the input to the device and make predictions on it
                features = features.to(self.device)
                pred = self.model(features)

                train_correct += (pred.argmax(1).cpu() == label).type(torch.float).sum().item()
                count += 1
                # find the class label index with the largest corresponding probability
                batch_preds = pred.argmax(axis=1).cpu().numpy()
                pred_list.extend(batch_preds)
                act_list.extend(label)
                pred_out.extend(pred.cpu().numpy())
                break
            print("acc:", train_correct / count)
        return pred_list, act_list, pred_out

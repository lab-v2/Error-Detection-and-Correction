import os
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, prefix_model_name=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.prefix_model_name = prefix_model_name

    def early_stop(self, validation_loss, model, epoch):
        if validation_loss < self.min_validation_loss:
            print("Current validation loss %f is less than the min validation loss %f in epoch %d, save the model"
                  %(validation_loss, self.min_validation_loss, epoch))
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(model, "./Models/" + self.prefix_model_name + "_best" )
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Trainer:
    def __init__(self, model = None, train_dataloader = None, val_dataloader = None, saved_model_path = None, prob = None, prefix_model_name = None, need_prob = False):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.has_mps else "cpu"))
        self.prob = prob
        self.saved_model_path = saved_model_path
        self.prefix_model_name = prefix_model_name
        self.need_prob = need_prob

    def fit(self, init_lr=1e-3, batch_size=64, epochs=10, saved_model_path=None ):

        # Define Training Hyperparameters
        self.saved_model_path = saved_model_path
        self.init_lr = init_lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_steps = len(self.train_dataloader.dataset) // batch_size
        if self.val_dataloader is not None:
            self.val_steps = len(self.val_dataloader.dataset) // batch_size

        # load the model and set it to evaluation mode
        if saved_model_path is not None:
            if os.path.isfile(saved_model_path):
                self.model = torch.load(saved_model_path).to(self.device)
            else:
                print("Warning: saved model path file didn't exists, training from the begining...\n")
        else:
            self.model = self.model.to(self.device)

        # Loss & Optimizer
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.parameters(), lr = self.init_lr, weight_decay = 0.000001)
        #optimizer = optim.Adam(self.model.parameters(), lr = 0.0001, weight_decay = 0.0000001)
        #optimizer = optim.Adam(self.model.parameters(), lr = self.init_lr)

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

        early_stopper = EarlyStopper(patience=50, min_delta=10, prefix_model_name = self.prefix_model_name)
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

                #compute probability
                if self.need_prob:
                    probability = self.prob.compute_probability(x)
                    probability = torch.from_numpy(probability).to(self.device)

                # send the input to the device
                (x, y) = (x.to(self.device), y.to(self.device))

                # perform a forward pass and calculate the training loss
                if self.need_prob:
                    orig_out, pred = self.model(x, probability)
                else:
                    orig_out, pred = self.model(x)

                #print("pred",pred.shape)
                #print("label:", y.shape)
                #print(pred.argmax(1))
                #print(y)
                #assert 1 == 2
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

                    if self.need_prob:
                        probability = self.prob.compute_probability(x)
                        probability = torch.from_numpy(probability).to(self.device)


                    # perform a forward pass and calculate the training loss
                    if self.need_prob:
                        orig_out, pred1 = self.model(x, probability)
                    else:
                        orig_out, pred1 = self.model(x)

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
                    corrector = 1
                    for (x, y) in pbar_val:
                        #compute probability
                        if self.need_prob:
                            probability = self.prob.compute_probability(x)
                            probability = torch.from_numpy(probability).to(self.device)

                        # send the input to the device
                        (x, y) = (x.to(self.device), y.to(self.device))

                        # make the predictions and calculate the validation loss
                        if self.need_prob:
                            orig_out, pred = self.model(x, probability)
                        else:
                            orig_out, pred = self.model(x)
                        val_batches+=1
                        total_val_loss += criterion(pred, y).item()

                        # calculate the number of correct val predictions
                        val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()        

                        batch_val_loss = total_val_loss/val_batches
                        val_accuracy = val_correct / (batch_size*val_batches*corrector)
                        pbar.set_postfix({'train_loss': batch_val_loss, 
                                        'train_acc': val_accuracy})


            # calculate the average training and validation loss
            avg_train_loss = total_train_loss / self.train_steps
            if self.val_dataloader is not None:
                avg_val_loss = total_val_loss / self.val_steps

            train_correct = train_correct / (len(self.train_dataloader.dataset)*len(self.train_dataloader.dataset[0]))
            if self.val_dataloader is not None:
                val_correct = val_correct / (len(self.val_dataloader.dataset)* 5)
                #print(self.val_dataloader.dataset[0][0].shape)
                #val_correct = val_correct / (len(self.val_dataloader.dataset)*len(self.val_dataloader.dataset[0]))

            # update our training history
            history["train_loss"].append(avg_train_loss)
            history["train_acc"].append(train_correct)
            if self.val_dataloader is not None:
                history["val_loss"].append(avg_val_loss)
                history["val_acc"].append(val_correct)

            # print the model training and validation information
            if self.val_dataloader is not None:
                print("[INFO] EPOCH: {}/{} --> Val loss: {:.6f}, Val accuracy: {:.4f}".\
                      format(epoch + 1, self.epochs, avg_val_loss, val_correct))
            else:
                print("[INFO] EPOCH: {}/{} --> Train loss: {:.6f}, Train accuracy: {:.6f}".\
                      format(epoch + 1, self.epochs,avg_train_loss, train_correct))
            if early_stopper.early_stop(total_val_loss, self.model, epoch):             
                break 
        # finish measuring how long training took
        end_time = time.time()
        print("[INFO] Total Training Time: {:.2f}s".format(end_time - start_time))
        torch.save(self.model, "./Models/" + self.prefix_model_name + "_final" )
        #if saved_model_path:
        #else:
        #    return history, self.model


    def predict(self, test_dataloader, saved_model_path = None):
        pred_list = []
        pred_out = []
        act_list = []
        orig_outs = []
        cntr=0
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.has_mps else "cpu"))
        print(self.device)

        # load the model and set it to evaluation mode
        if saved_model_path is not None:
            #self.model = torch.load(saved_model_path).to(self.device)
            self.model = torch.load(saved_model_path,map_location=torch.device('cpu')).to(self.device)
        
        self.model.eval()

        # switch off autograd
        with torch.no_grad():
            
            # loop over the test set
            count = 0
            train_correct = 0
            raw_datas = []
            for (features, label) in tqdm(test_dataloader, desc = "Testing:"):

                #compute probability
                if self.need_prob:
                    probability = self.prob.compute_probability(features)
                    probability = torch.from_numpy(probability).to(self.device)

                # send the input to the device and make predictions on it
                features = features.to(self.device)
                if self.need_prob:
                    orig_out, pred = self.model(features, probability)
                else:
                    orig_out, pred = self.model(features)
                #print("pred0:",pred[0])
                #print("pred1:",pred[0,:,0])
                #print("predmax:",pred.argmax(1)[0])
                #assert 1 == 2
                #pred_before, pred = self.model(features)
                pred = pred.cpu()

                train_correct += (pred.argmax(1) == label).type(torch.float).sum().item()
                count += 1
                # find the class label index with the largest corresponding probability
                batch_preds = pred.argmax(axis=1).cpu().numpy()
                pred_list.extend(batch_preds)
                act_list.extend(label.cpu().numpy())
                pred_out.extend(pred.cpu().numpy())
                orig_outs.extend(orig_out.cpu().numpy())
                #max_values = features.cpu().numpy().max(axis=(3))
                max_values = features.cpu().numpy()
                raw_datas.extend(max_values)
            print("acc:", train_correct / count)
        return pred_list, act_list, pred_out, orig_outs, raw_datas

import os
import pickle
import sys
import multiprocessing as mp
import mdtraj as md
import numpy as np
from . import exmax, nnutils, utils, data_processing
import copy
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data as torch_data


class Dataset(torch_data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, train_inds, labels, data):
        'Initialization'
        self.labels = labels
        self.train_inds = train_inds
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.train_inds)

  def __getitem__(self, index):
        'Generates one sample of data'
        #If data needs to be loaded
        ID = self.train_inds[index]
        if type(self.data) is str:
            # Load data and get label
            X = torch.load(self.data + "/ID-%s" % ID + '.pt')
        else: 
            X = torch.from_numpy(self.data[ID]).type(torch.FloatTensor)
        y = self.labels[ID]
        
        return X, y, ID


class Trainer:

    def __init__(self,job):
        """Object to train your DiffNet.
        
        Parameters:
        -----------
        job : dict 
            Dictionary with all training parameters. See training_dict.txt
            for all keys. All keys are required. See train_submit.py for an
            example.
        """
        self.job = job

    def set_training_data(self, job, train_inds, test_inds, labels, data):
        """Construct generators out of the dataset for training, validation,
        and expectation maximization.

        Parameters
        ----------
        job : dict
            See training_dict.txt for all keys.
        train_inds : np.ndarray
            Indices in data that are to be trained on
        test_inds : np.ndarray
            Indices in data that are to be validated on
        labels : np.ndarray,
            classification labels used for training
        data : np.ndarray, shape=(n_frames,3*n_atoms) OR str to path
            All data
        """

        batch_size = job['batch_size']
        cpu_cores = job['em_n_cores']
        test_batch_size = job['test_batch_size']
        em_batch_size = job['em_batch_size']
        subsample = job['subsample']
        data_dir = job["data_dir"]

        n_train_inds = len(train_inds)  # number of data points in the training set
        random_inds = np.random.choice(np.arange(n_train_inds),int(n_train_inds/subsample),replace=False)   # replace=False: A value in np.arange(n_train_inds) cannot be selected multiple times
        sampler=torch_data.SubsetRandomSampler(random_inds)

        params_t = {'batch_size': batch_size,
                  'shuffle':False,
                  'num_workers': cpu_cores,
                  'sampler': sampler}

        params_v = {'batch_size': test_batch_size,
                  'shuffle':True,
                  'num_workers': cpu_cores}

        params_e = {'batch_size': em_batch_size,
                  'shuffle':True,
                  'num_workers': cpu_cores}

        n_snapshots = len(train_inds) + len(test_inds)

        training_set = Dataset(train_inds, labels, data)
        training_generator = torch_data.DataLoader(training_set, **params_t)

        validation_set = Dataset(test_inds, labels, data)
        validation_generator = torch_data.DataLoader(validation_set, **params_v)

        em_set = Dataset(train_inds, labels, data)
        em_generator = torch_data.DataLoader(em_set, **params_e)

        return training_generator, validation_generator, em_generator
    
    def em_parallel(self, net, em_generator, train_inds, em_batch_size,
                    indicators, em_bounds, em_n_cores, label_str, epoch):
        """Use expectation maximization to update all training classification
           labels.

        Parameters
        ----------
        net : nnutils neural network object
            Neural network 
        em_generator : Dataset object
            Training data
        train_inds : np.ndarray
            Indices in data that are to be trained on
        em_batch_size : int
            Number of examples that are have their classification labels
             updated in a single round of expectation maximization.
        indicators : np.ndarray, shape=(len(data),)
            Value to indicate which variant each data frame came from.
        em_bounds : np.ndarray, shape=(n_variants,2)
            A range that sets what fraction of conformations you
            expect a variant to have biochemical property. Rank order
            of variants is more important than the ranges themselves.
        em_n_cores : int
            CPU cores to use for expectation maximization calculation

        Returns
        -------
        new_labels : np.ndarray, shape=(len(data),)
            Updated classification labels for all training examples
        """
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        
        n_em = np.ceil(train_inds.shape[0]*1.0/em_batch_size) # number of EM batches
        print(f"    Number of samples in the traing set: {train_inds.shape[0]}")
        print(f"    em_batch_size={em_batch_size} -> {int(n_em)} EM batches")

        freq_output = np.floor(n_em/10.0)  # at most print 10 times
        
        train_inds = []
        inputs = []
        i = 0
        
        # To save DiffNet labels before each EM update
        pred_labels = -1 * np.ones(indicators.shape[0])   # just for initialization (shape: (n_frames,)), reminder: indicators is something like [0, 0, 0, ..., 3, 3, 3] (case of 4 variants)
        for local_batch, local_labels, t_inds in em_generator:
            # shape of local_batch: (em_batch_size, n_feat)
            # shape of local_labels: (em_batch_size, 1)
            # shape of t_inds: (em_batch_size)
            t_inds = np.array(t_inds)   # indices of the samples in the training set used here
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            if hasattr(net, "decode"):
                if hasattr(net, "reparameterize"):
                    x_pred, latent, logvar, class_pred = net(local_batch)
                else:
                    x_pred, latent, class_pred = net(local_batch)
            else:   # classify_ae is the only case
                class_pred = net(local_batch)

            # class_pred: for a batch; pred_labels: for the whole dataset
            cur_labels = class_pred.cpu().detach().numpy() # shape: (em_batch_size, 1)
            pred_labels[t_inds] = cur_labels.flatten()     # shape after flatting: (em_batch_size,)
            inputs.append([cur_labels, indicators[t_inds], em_bounds])  # for apply_examx
            """
            if i % freq_output == 0:
                print("      %d/%d" % (i, n_em))
            i += 1
            """
            train_inds.append(t_inds)

        pred_label_fn = os.path.join(self.job['outdir'],"tmp_labels_%s_%s.npy" % (label_str,epoch))
        np.save(pred_label_fn,pred_labels)
        pool = mp.Pool(processes=em_n_cores)
        res = pool.map(self.apply_exmax, inputs)  # should be a list of  arrays
        pool.close()
        
        # train_inds = np.concatenate(np.array(train_inds))
        train_inds = np.concatenate(train_inds)
        # NumPy array such as A = np.array([2,1,3], [2, 2]) would cause the following warning:
        # VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences
        # (which is a list-or-tuple of lists-or-tuples-or ndarrays with different
        # lengths or shapes) is deprecated. If you meant to do this, you must
        # specify 'dtype=object' when creating the ndarray
        # Since not all the batches have the same number of samples, this warning could happen
        # a lot. Concatening a list instead of a numpy array could avoid this warning. 

        new_labels = -1 * np.ones((indicators.shape[0], 1))
        new_labels[train_inds] = np.concatenate(res)
        return new_labels

    def apply_exmax(self, inputs):
        """Apply expectation maximization to a batch of data.

        Parameters
        ----------
        inputs : list
            list where the 0th index is a list of current classification
            labels of length == batch_size. 1st index is a corresponding
            list of variant simulation indicators. 2nd index is em_bounds.
            
        Returns
        -------
        Updated labels -- length == batch size
        """
        # It seems that "RuntimeWarning: divide by zero encountered in true_divide" only 
        # occurs when some labels obtaiend from expectation_range_CUBIC are NaN. 
        # Here we temporarily ignore them to make the logging cleaner.
        np.seterr(divide='ignore', invalid='ignore')
        
        cur_labels, indicators, em_bounds = inputs
        n_vars = em_bounds.shape[0]    # number of variants

        for i in range(n_vars):
            inds = np.where(indicators == i)[0]
            lower = np.int(np.floor(em_bounds[i, 0] * inds.shape[0]))
            upper = np.int(np.ceil(em_bounds[i, 1] * inds.shape[0]))
            cur_labels[inds] = exmax.expectation_range_CUBIC(cur_labels[inds], lower, upper).reshape(cur_labels[inds].shape)

        bad_inds = np.where(np.isnan(cur_labels))
        if len(bad_inds[0]) > 0:
            for i in bad_inds:
                print(f"    {len(bad_inds[0])} out of {len(cur_labels)} classification labels are NaN.")
        cur_labels[bad_inds] = 0   # if the label is nan, correct it to 0
        
        try:
            assert((cur_labels >= 0.).all() and (cur_labels <= 1.).all())
        except AssertionError:
            neg_inds = np.where(cur_labels<0)[0]
            pos_inds = np.where(cur_labels>1)[0]
            bad_inds = neg_inds.tolist() + pos_inds.tolist()
            
            if len(neg_inds) > 0:
                print(f"    {len(neg_inds)} out of {len(cur_labels)} classification labels are smaller than 0.")
                info_str = ''
                if len(neg_inds) < 3:
                    info_list = [f"({indicators[neg_inds[i]]}, {cur_labels[neg_inds[i]][0]:.7f})" for i in range(len(neg_inds))]
                    print(f"    Here are the paris of (variant_idx, label): {'  '.join(info_list)}")
                else:
                    start = "    Here are the paris of (variant_idx, label):"
                    for i in range(len(neg_inds)):
                        info = f"({indicators[neg_inds[i]]}, {cur_labels[neg_inds[i]][0]:.7f})"
                        if i % 4 == 0:
                            info_str += f'\t{info}'
                        elif i % 4 == 3:
                            info_str += f'  {info}\n'
                        else:
                            info_str += f'  {info}'
                    info_str = start + '\n' + info_str
                print(info_str)

            if len(pos_inds) > 0:
                print(f"    {len(pos_inds)} out of {len(cur_labels)} classification labels are larger than 1.")
                info_str = ''
                if len(pos_inds) < 3: 
                    info_list = [f"({indicators[pos_inds[i]]}, {cur_labels[pos_inds[i]][0]:.7f})" for i in range(len(pos_inds))]
                    print(f"    Here are the paris of (variant_idx, label): {'  '.join(info_list)}")    
                else:
                    start = "    Here are the paris of (variant_idx, label):"
                    for i in range(len(pos_inds)):
                        info = f"({indicators[pos_inds[i]]}, {cur_labels[pos_inds[i]][0]:.7f})"
                        if i % 4 == 0:
                            info_str += f'\t{info}'
                        elif i % 4 == 3:
                            info_str += f'  {info}\n'
                        else:
                            info_str += f'  {info}'
                    info_str = start + '\n' + info_str
                print(info_str)
            
            """
            for iis in bad_inds:
                print("      ", indicators[iis], cur_labels[iis])
            print("      #bad neg, pos", len(neg_inds), len(pos_inds))
            """

            #np.save("tmp.npy", tmp_labels)
            cur_labels[neg_inds] = 0.0
            cur_labels[pos_inds] = 1.0
            #sys.exit(1)
            
            # WTH: I don't thinkg cur_labels is different from cur_labels.reshape((cur_labels.shape[0], 1)) ... (They also )
        return cur_labels.reshape((cur_labels.shape[0], 1))

    def train(self, data, training_generator, validation_generator, em_generator,
              targets, indicators, train_inds, test_inds,net, label_str,
              job, lr_fact=1.0):
        """Core method for training

        Parameters
        ----------
        data : np.ndarray, shape=(n_frames,3*n_atoms) OR str to path
            Training data
        training_generator: Dataset object
            Generator to sample training data
        validation_generator: Dataset object
            Generator to sample validation data
        em_generator: Dataset object
            Generator to sample training data in batches for expectation
            maximization
        targets : np.ndarray, shape=(len(data),)
            classification labels used for training
        indicators : np.ndarray, shape=(len(data),)
            Value to indicate which variant each data frame came from.
        train_inds : np.ndarray
            Indices in data that are to be trained on
        test_inds : np.ndarray
            Indices in data that are to be validated on
        net : nnutils neural network object
            Neural network
        label_str: int
            For file naming. Indicates what iteration of training we're
            on. Training goes through several iterations where neural net
            architecture is progressively built deeper.
        job : dict
            See training_dict.tx for all keys.
        lr_fact : float
            Factor to multiply the learning rate by.

        Returns
        -------
        best_nn : nnutils neural network object
            Neural network that has the lowest reconstruction error
            on the validation set.
        targets : np.ndarry, shape=(len(data),)
            Classification labels after training.
        """
        job = self.job
        do_em = job['do_em']
        n_epochs = job['n_epochs']
        lr = job['lr'] * lr_fact
        subsample = job['subsample']
        batch_size = job['batch_size']
        batch_output_freq = job['batch_output_freq']
        epoch_output_freq = job['epoch_output_freq']
        test_batch_size = job['test_batch_size']
        em_bounds = job['em_bounds']
        nntype = job['nntype']
        em_batch_size = job['em_batch_size']
        em_n_cores = job['em_n_cores']
        outdir = job['outdir'] 
        w_loss = job['w_loss']

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        n_test = test_inds.shape[0]
        lam_cls = w_loss[0]    # does not change the value at all, possibly just a place convenient ofr changing the weight of the classification error
        lam_corr = w_loss[1]   # not used at all.

        n_batch = np.ceil(train_inds.shape[0]*1.0/subsample/batch_size)

        optimizer = optim.Adam(net.parameters(), lr=lr)
        bce = nn.BCELoss()  # BCE = Binary Cross Entropy (classificaiton loss function)
        training_loss_full = []
        test_loss_full = []
        epoch_test_loss = []
        best_loss = np.inf
        best_nn = None 
        ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
        
        loss_term = [[], [], [], []]  # for L1, MSE, BCE, corr_penalty (validation)

        for epoch in range(n_epochs):
            print(f'\n    \U0001F92F Going through the {ordinal(epoch + 1)} epoch ...')
            # go through mini batches
            running_loss = 0
            i = 0   # batch index
            for local_batch, local_labels, _ in training_generator:
                if use_cuda:
                    local_labels = local_labels.type(torch.cuda.FloatTensor)
                else:
                    local_labels = local_labels.type(torch.FloatTensor)
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                optimizer.zero_grad()    
                x_pred, latent, class_pred = net(local_batch)  # same as net.forward(local_batch), shape of latent: (batch_size, n_features)
                loss = nnutils.my_mse(local_batch, x_pred)
                loss += nnutils.my_l1(local_batch, x_pred)
                if class_pred is not None:
                    local_labels = local_labels.reshape(-1, 1)  # for gaussian and bimodal label spreading
                    loss += bce(class_pred, local_labels).mul_(lam_cls)
                
                #Minimize correlation between latent variables
                n_feat = net.sizes[-1]
                my_c00 = torch.einsum('bi,bo->io', (latent, latent)).mul(1.0/local_batch.shape[0])  # covariance, np.einsum('bi,bo->io', A, A) = np.matmul(A.T, A) 
                my_mean = torch.mean(latent, 0)   # axis = 0, average over frames 
                my_mean = torch.einsum('i,o->io', (my_mean, my_mean)) 
                ide = np.identity(n_feat)
                if use_cuda:
                    ide = torch.from_numpy(ide).type(torch.cuda.FloatTensor)
                else:
                    ide = torch.from_numpy(ide).type(torch.FloatTensor)
                #ide = Variable(ide)
                #ide = torch.from_numpy(np.identity(n_feat))
                #ide = ide.to(device)
                zero_inds = np.where(1-ide.cpu().numpy()>0)
                corr_penalty = nnutils.my_mse(ide[zero_inds], my_c00[zero_inds]-my_mean[zero_inds])
                loss += corr_penalty
                loss.backward()   # compute the gradient of the curent tensor
                optimizer.step()  # move to next iteration of GD
                running_loss += loss.item()   # .item(): returns the value of the tensor as a standard Python number 

                if i%batch_output_freq == 0:
                    train_loss = running_loss
                    if i != 0:
                        train_loss /= batch_output_freq  # average loss
                    training_loss_full.append(train_loss)

                    test_loss = 0
                    k=0
                    for local_batch, local_labels, _ in validation_generator:
                        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                        x_pred, latent, class_pred = net(local_batch)
                        loss = nnutils.my_mse(local_batch,x_pred)
                        k += loss.item()
                        test_loss += loss.item() * local_batch.shape[0] # mult for averaging across samples, as in train_loss
                    #print("        ", test_loss)
                    test_loss /= n_test # n_test: number of samples in the test set
                    test_loss_full.append(test_loss)
                    # print("    [%s %d, %5d/%d] train loss: %0.6f    test loss: %0.6f" % (label_str, epoch, i, n_batch, train_loss, test_loss))
                    print(f'    [epoch: {epoch+1}/{n_epochs}, batch: {i+1}/{int(n_batch)}]  train loss (L_tot): {train_loss:.5f}   test loss (MSE): {test_loss:.5f}')
                    running_loss = 0

                    if test_loss < best_loss:
                        best_loss = test_loss
                        best_nn = copy.deepcopy(net)
                
                i += 1

            if do_em and hasattr(nntype, "classify"):
                print("\n    Applying the expectation maximization (EM) algorithm ...")
                targets = self.em_parallel(net, em_generator, train_inds,
                                    em_batch_size, indicators, em_bounds,
                                    em_n_cores, label_str, epoch)
                training_generator, validation_generator, em_generator = \
                    self.set_training_data(job, train_inds, test_inds, targets, data) 

            if epoch % 1 == 0:  # print the following information every epoch
                L1 = nnutils.my_l1(local_batch, x_pred)
                MSE = nnutils.my_mse(local_batch, x_pred)
                BCE = bce(class_pred, local_labels).mul_(lam_cls)
                L_total = L1 + MSE + BCE + corr_penalty

                loss_term[0].append(L1.item())
                loss_term[1].append(MSE.item())
                loss_term[2].append(BCE.item())
                loss_term[3].append(corr_penalty.item())

                print(f"\n    \U0001F4DD (End of {ordinal(epoch + 1)} epoch) Total validation loss of DiffNets: {L_total:.4f}")
                print(f"        Reconstruction loss: {L1 + MSE:.4f}")
                print(f"            L1 norm: {L1:.4f}")
                print(f"            MSE: {MSE:.4f}")
                print(f"        Classification error: {BCE:.4f}")
                print(f"        Correlation penalty: {corr_penalty:.4f}")
            
            if epoch % epoch_output_freq == 0:
                epoch_test_loss.append(test_loss)
                out_fn = os.path.join(outdir, "epoch_test_loss_%s.npy" % label_str)
                np.save(out_fn, epoch_test_loss)

                out_fn = os.path.join(outdir, "training_loss_%s.npy" % label_str)
                np.save(out_fn, training_loss_full)
                
                out_fn = os.path.join(outdir, "test_loss_%s.npy" % label_str)
                np.save(out_fn, test_loss_full)
            # nets need be on cpu to load multiple in parallel, e.g. with multiprocessing
                net.cpu()
                out_fn = os.path.join(outdir, "nn_%s_e%d.pkl" % (label_str, epoch))
                pickle.dump(net, open(out_fn, 'wb'))
                if use_cuda:
                    net.cuda()
                if hasattr(nntype, "classify"):
                    out_fn = os.path.join(outdir, "tmp_targets_%s_%s.npy" % (label_str,epoch))
                    np.save(out_fn, targets)
            

            # save best net every epoch
            best_nn.cpu()
            out_fn = os.path.join(outdir, "nn_best_%s.pkl" % label_str)
            pickle.dump(best_nn, open(out_fn, 'wb'))
            if use_cuda:
                best_nn.cuda()
        
        out_fn = os.path.join(outdir, f"all_loss_term_{label_str}.npy")
        np.save(out_fn, np.array(loss_term))

        return best_nn, targets    

    def get_targets(self,act_map,indicators,label_spread=None):
        """Convert variant indicators into classification labels.

        Parameters
        ----------
        act_map : np.ndarray, shape=(n_variants,)
            Initial classification labels to give each variant.
        indicators : np.ndarray, shape=(len(data),)
            Value to indicate which variant each data frame came from.

        Returns
        -------
        targets : np.ndarry, shape=(len(data),1)
            Classification labels for training.
        """
        targets = np.zeros((len(indicators), 1))
        if label_spread == 'gaussian':
            targets = np.array([np.random.normal(act_map[i],0.1) for i in indicators])
            zero_inds = np.where(targets < 0)[0]
            targets[zero_inds] = 0
            one_inds = np.where(targets > 1)[0]
            targets[one_inds] = 1
        elif label_spread == 'uniform':
            targets = np.vstack([np.random.uniform() for i in targets])
        elif label_spread == 'bimodal':
            targets = np.array([np.random.normal(0.8, 0.1) if np.random.uniform() < act_map[i]
                                else np.random.normal(0.2, 0.1) for i in indicators])
            zero_inds = np.where(targets < 0)[0]
            targets[zero_inds] = 0
            one_inds = np.where(targets > 1)[0]
            targets[one_inds] = 1
        else:
            targets[:, 0] = act_map[indicators]    
        return targets

    def split_test_train(self,n,frac_test):
        """Split data into training and validation sets.

        Parameters
        ----------
        n : int
            number of data points
        frac_test : float between 0 and 1
            Fraction of dataset to reserve for validation set

        Returns
        -------
        train_inds : np.ndarray
            Indices in data that are to be trained on
        test_inds : np.ndarray
            Indices in data that are to be validated on
        """
        n_test = int(n*frac_test)
       
        inds = np.arange(n)
        np.random.shuffle(inds)
        train_inds = inds[:-n_test]
        test_inds = inds[-n_test:]

        return train_inds, test_inds
    
    def run(self, data_in_mem=False):
        """Wrapper for running the training code

        Parameters
        ----------
        data_in_mem: boolean
            If true, load all training data into memory. Training faster this way.
        
        Returns
        -------
        net : nnutils neural network object
            Trained DiffNet
        """
        job = self.job 
        data_dir = job['data_dir']
        outdir = job['outdir']
        n_latent = job['n_latent']
        layer_sizes = job['layer_sizes']
        nntype = job['nntype']
        frac_test = job['frac_test']
        act_map = job['act_map']
        
        use_cuda = torch.cuda.is_available()
        print(f'Is CUDA used or not? {use_cuda}')
        
        # Step 1: Assign classification labels for all simulation frames
        print('\nAssigning classification labels for all simulation frames ...')
        indicator_dir = os.path.join(data_dir, "indicators")
        indicators = utils.load_npy_dir(indicator_dir, "*.npy")
        indicators = np.array(indicators, dtype=int)
        
        if 'label_spreading' in job.keys():
            targets = self.get_targets(act_map,indicators,
                                       label_spread=job['label_spreading'])
        else:
            targets = self.get_targets(act_map,indicators)
        n_snapshots = len(indicators)   # number of frames of all trajectories in total
        np.save(os.path.join(outdir, 'initial_targets.npy'), targets)

        # Step 2: Split the dataset and construct data generators
        print('Splitting the dataset into training/validation sets ...')
        train_inds, test_inds = self.split_test_train(n_snapshots,frac_test)
        if data_in_mem:
            xtc_dir = os.path.join(data_dir,"aligned_xtcs")
            top_fn = os.path.join(data_dir, "master.pdb")
            master = md.load(top_fn)
            data = utils.load_traj_coords_dir(xtc_dir, "*.xtc", master.top)
        else:
            data = os.path.join(data_dir, "data")

        print("Constructing data generators for training, validation, and expectation maximization ... ")
        training_generator, validation_generator, em_generator = \
            self.set_training_data(job, train_inds, test_inds, targets, data)

        print(f"Total number of examples in the original dataset: {targets.shape[0]}")

        wm_fn = os.path.join(data_dir, "wm.npy")
        uwm_fn = os.path.join(data_dir, "uwm.npy")
        cm_fn = os.path.join(data_dir, "cm.npy")
        wm = np.load(wm_fn)
        uwm = np.load(uwm_fn)
        cm = np.load(cm_fn).flatten()   # WTH: flatten() seems unnecessary since the cm array is of the shape of (3 * n_atoms,).

        n_train = train_inds.shape[0]
        n_test = test_inds.shape[0]
        out_fn = os.path.join(outdir, "train_inds.npy")
        np.save(out_fn, train_inds)
        out_fn = os.path.join(outdir, "test_inds.npy")
        np.save(out_fn, test_inds)
        print(f"    Number of samples in the training set: {n_train}")
        print(f"    Number of samples in the validation set: {n_test}")
        
        n_batch = np.ceil(n_train*1.0/job['subsample']/job['batch_size'])
        print(f'    (subsample={job["subsample"]}, batch_size={job["batch_size"]}) -> {int(n_batch)} batches')

        if hasattr(nntype, 'split_inds'):
            inds1 = job['inds1']  # region of interest, encoder A
            inds2 = job['inds2']  # the rest of the system, encoder B
            print('\nConstructing a split autoencoder ...')
            print('    Encoder A: region of interest, Encoder B: rest of the system')
            print('    Atoms (index, {resSeq}{resName}, {atom type}) in Encoder A:')
            
            info_str = ''
            master = md.load(os.path.join(data_dir, "master.pdb"))
            u_top = master.topology.to_dataframe()[0]
            for i in range(len(inds1)):
                info = f"[{i}, {u_top.iloc[i]['resSeq']}{u_top.iloc[i]['resName']}, {u_top.iloc[i]['name']}]"
                if i % 3 == 0:
                    info_str += f'\t{info}'
                elif i % 3 == 2:
                    info_str += f'\t{info}\n'
                else:
                    info_str += f'\t{info}'
            print(info_str)

            old_net = nntype(layer_sizes[0:2],inds1,inds2,wm,uwm)
        else:
            print('Constructing a non-split autoencoder ...')
            old_net = nntype(layer_sizes[0:2],wm,uwm)
        old_net.freeze_weights()
        
        print('\nNote: Below are the weights of each kind of loss function:')
        print(f'    Reconstruction loss: 1.0')
        print(f'    Classification error: {job["w_loss"][0]}')
        print(f'    Correlation penalty: {job["w_loss"][1]}')

        ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
        for cur_layer in range(2,len(layer_sizes)):
            if hasattr(nntype, 'split_inds'):
                net = nntype(layer_sizes[0:cur_layer+1],inds1,inds2,wm,uwm)
            else:
                net = nntype(layer_sizes[0:cur_layer+1],wm,uwm)
            net.freeze_weights(old_net)
            if use_cuda:
                net.cuda()
            print(f'\nNow training the {ordinal(cur_layer - 1)} out of the {len(layer_sizes) - 2} hidden layers ...')
            net, targets = self.train(data, training_generator, 
                               validation_generator, em_generator,
                               targets, indicators, train_inds,
                               test_inds, net, str(cur_layer), job)
            #Might make sense to make this optional
            training_generator, validation_generator, em_generator = \
                self.set_training_data(job, train_inds, test_inds, targets, data)          
            old_net = net

        #Polishing
        net.unfreeze_weights()
        if use_cuda:
            net.cuda()
        print('\n Now polishing the network ...')
        net, targets = self.train(data, training_generator, validation_generator,
                               em_generator, targets, indicators, train_inds,
                               test_inds, net, "polish", job, lr_fact=0.1)
        return net

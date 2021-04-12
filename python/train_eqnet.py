#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.callbacks import TerminateOnNaN, EarlyStopping, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf

from aux_networks import QuantizationAutoencoder, EstimationEncoder
from aux_networks import sample_balanced_wmse
from aux_networks import sample_wmse, sample_wmse_numpy

from aux_matlab import decode_matlab_file
from pymatbridge import Matlab

import numpy as np
import hdf5storage
import os

import joblib
from sklearn.cluster import MiniBatchKMeans

# GPU allocation
K.clear_session()
tf.reset_default_graph()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";
# Set a global seed
global_seed = 1000
# Tensorflow memory allocation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
K.tensorflow_backend.set_session(tf.Session(config=config))

### Parameters and initializations
# Scenario parameters
num_tx, num_rx = 2, 2
mod_size       = 6 * num_tx # K = log2(M) in the paper, bits per QAM symbol

# Autoencoder parameters
ae_cfg = { # Architecture
            'latent_dim': 3 * num_tx,
            'num_layers': 6,
            'hidden_dim': 4 * mod_size * np.ones((12,), dtype=np.int32),
            'common_layer': 'relu',
            'latent_layer': 'tanh',
            'weight_reg': 0.,
            'noise_sigma': 1e-3,
            'global_eps': 1e-6,
            # Training
            'batch_size': 8192 * 4,
            'num_rounds': 12,
            'num_epochs_1': 100,
            'num_epochs_2': 800,
            'freeze_enc': True,
            'learning_rate': 1e-3}
# Encoder parameters
enc_cfg = { # Architecture
            'latent_dim': ae_cfg['latent_dim'],
            'num_layers': 7,
            'num_blocks': 1,
            'hidden_dim': 16 * mod_size * np.ones((7,), dtype=np.int32),
            'common_layer': 'relu',
            'latent_layer': 'linear',
            'conj_inputs': False,
            'weight_reg': 0.,
            'global_eps': ae_cfg['global_eps'],
            # Training
            'batch_size': 8192 * 4,
            'num_epochs': 2000,
            'learning_rate': 1e-3,
            'latent_loss': 'mean_absolute_error'}

# Quantizer parameters
bits_per_dim = [5, 6]
# Are we training the quantizer?
train_quantizer = True
# Are we training the encoder?
train_encoder = True

# Inference parameters
inf_batch_size = 65536

# Define all used seeds
train_seed = 1234
test_seed  = 4321

### Training
# Which algorithm and which channel model
training_channel = 'rayleigh'
training_alg     = 'ml'
# Target file for training/validation data
train_file = 'matlab/data/extended_mimo%dby%d_mod%d_seed%d.mat' % (
        num_tx, num_rx, mod_size//num_tx, train_seed)
# train_file = 'matlab/data/extended_%s_%s_mimo%dby%d_mod%d_seed%d.mat' % (
#         training_channel, training_alg, 
#         num_tx, num_rx, mod_size//num_tx, train_seed)
# Load and remove high SNR points
contents   = hdf5storage.loadmat(train_file)
train_snr  = np.arange(len(np.squeeze(contents['snr_range'])))
ref_llr    = np.asarray(contents['ref_llr'])[train_snr]
ref_y      = np.asarray(contents['ref_y'])[train_snr]
ref_h      = np.asarray(contents['ref_h'])[train_snr]
ref_n      = np.asarray(contents['ref_n'])[train_snr]
# Reshape, convert to soft bits and seeded shuffle
# Downselect low-mid SNR values and permute
llr_train = np.moveaxis(ref_llr, -1, -3)
y_train   = np.moveaxis(ref_y, -1, -2)
h_train   = np.moveaxis(ref_h, -1, -3)
n_train   = np.repeat(ref_n[..., None], llr_train.shape[2], axis=1)
# Apply conjugate operator to features
if enc_cfg['conj_inputs']:
    y_train = np.matmul(np.conj(np.swapaxes(h_train, -2, -1)), y_train[..., None])[..., 0]
    h_train = np.matmul(np.conj(np.swapaxes(h_train, -2, -1)), h_train)
# Reshape
llr_train = np.reshape(llr_train, llr_train.shape[:-2] + (-1,))
llr_train = np.reshape(llr_train, (-1, mod_size))
llr_train = np.tanh(llr_train / 2)
y_train   = np.reshape(y_train, (-1, num_rx))
h_train   = np.reshape(h_train, h_train.shape[:-2] + (-1,))
h_train   = np.reshape(h_train, (-1, num_rx*num_tx))
n_train   = np.reshape(n_train, (-1, 1))
# Convert complex to reals
y_train   = y_train.view(np.float64)
h_train   = h_train.view(np.float64)
# Shuffle
np.random.seed(global_seed)
shuffled_idx = np.random.permutation(len(llr_train))
llr_train = llr_train[shuffled_idx]
y_train   = y_train[shuffled_idx]
h_train   = h_train[shuffled_idx]
n_train   = n_train[shuffled_idx]
# Split into training/validation
llr_train, llr_val, y_train, y_val, h_train, h_val, n_train, n_val = \
 train_test_split(llr_train, y_train, h_train, n_train, test_size=0.2)
 
# Test data
# Which algorithm and which channel model
testing_channel = 'fading'
testing_alg     = 'ml'
filename = 'matlab/data/extended_mimo%dby%d_mod%d_seed%d.mat' % (
        num_tx, num_rx, mod_size//num_tx, test_seed)
# filename = 'matlab/data/extended_%s_%s_mimo%dby%d_mod%d_seed%d.mat' % (
#         testing_channel, testing_alg,
#         num_tx, num_rx, mod_size//num_tx, test_seed)
contents   = hdf5storage.loadmat(filename)
ref_llr    = np.asarray(contents['ref_llr'])
ref_y      = np.asarray(contents['ref_y'])
ref_h      = np.asarray(contents['ref_h'])
ref_n      = np.asarray(contents['ref_n'])
ref_bits   = contents['ref_bits']
snr_range  = contents['snr_range'][0]
num_snr, num_codewords, _, _, _ = ref_llr.shape
# Reshape, convert to soft bits and seeded shuffle
llr_test = np.moveaxis(ref_llr, -1, -3)
llr_test = np.reshape(llr_test, llr_test.shape[:-2] + (-1,))
llr_test = np.reshape(llr_test, (-1, mod_size))
llr_test = np.tanh(llr_test / 2)
y_test   = np.moveaxis(ref_y, -1, -2)
h_test   = np.moveaxis(ref_h, -1, -3)
n_test   = np.repeat(ref_n[..., None], ref_llr.shape[-1], axis=1)
# Apply conjugate operator to features
if enc_cfg['conj_inputs']:
    y_test = np.matmul(np.conj(np.swapaxes(h_test, -2, -1)), y_test[..., None])
    h_test = np.matmul(np.conj(np.swapaxes(h_test, -2, -1)), h_test)
y_test   = np.reshape(y_test, (-1, num_rx))
h_test   = np.reshape(h_test, h_test.shape[:-2] + (-1,))
h_test   = np.reshape(h_test, (-1, num_rx*num_tx))
n_test   = np.reshape(n_test, (-1, 1))

# Convert complex to reals
y_test   = y_test.view(np.float64)
h_test   = h_test.view(np.float64)

# Start Matlab engine
eng = Matlab()
eng.start()
# Move to right path
eng.run_code('cd /home/yanni/marius/deep-llr-quantization-master/deep-llr-quantization-master/')

# How many runs
num_runs = 1
# Global metrics
local_seed_collect  = np.zeros((num_runs,))
bler_ae, ber_ae     = np.zeros((num_runs, num_snr)), np.zeros((num_runs, num_snr))
bler_aeq, ber_aeq   = np.zeros((num_runs, len(bits_per_dim), num_snr)), np.zeros((num_runs, len(bits_per_dim), num_snr))
bler_enc, ber_enc   = np.zeros((num_runs, num_snr)), np.zeros((num_runs, num_snr))
bler_encq, ber_encq = np.zeros((num_runs, len(bits_per_dim), num_snr)), np.zeros((num_runs, len(bits_per_dim), num_snr))

# Create a global result directory
global_dir = 'models_paper_ResubmitFig2Runs/freeze%d_cyclic%d/mimo%dby%d_\
conj%d_mod%d_aelayers%d_latent%d_blocks%d' % (
        ae_cfg['freeze_enc'], ae_cfg['cyclic_lr'],
        num_tx, num_rx, enc_cfg['conj_inputs'],
        mod_size//num_tx,
        ae_cfg['num_layers'], ae_cfg['latent_dim'],
        enc_cfg['num_blocks'])
if not os.path.exists(global_dir):
    os.makedirs(global_dir)
# Optionally, the target result directory if we want to use a pretrained representation
target_dir = 'models_paper/freeze%d_cyclic%d/mimo%dby%d_mod%d_aelayers%d_latent%d_blocks%d' % (
        ae_cfg['freeze_enc'], ae_cfg['cyclic_lr'],
        num_tx, num_rx, mod_size//num_tx, 
        ae_cfg['num_layers'], ae_cfg['latent_dim'],
        3)
target_seed = 1763986680

# For each run
for run_idx in range(num_runs):
    # Initial weight seeding - this allows for completely reproducible results
    local_seed = np.random.randint(low=0, high=2**31-1)
    np.random.seed(local_seed)
    # Store seeds
    local_seed_collect[run_idx] = local_seed
    # Create a local folder
    local_dir = global_dir + '/seed%d' % local_seed
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    # If desired, skip first stage
    if not train_quantizer:
        # Instantiate blank autoencoder
        ae, ae_list, enc, dec, dec_list = QuantizationAutoencoder(mod_size, ae_cfg['latent_dim'], 
                                                              ae_cfg['num_layers'], ae_cfg['hidden_dim'],
                                                              ae_cfg['common_layer'],
                                                              ae_cfg['latent_layer'],
                                                              ae_cfg['weight_reg'], local_seed,
                                                              False, ae_cfg['noise_sigma'])
        # Load weights
        ae.load_weights(target_dir + '/seed%d/finetune_best.h5' % target_seed)
    else:
        ### Stage 1 - Periodically update the weights wk ###
        # For each round
        for round_idx in range(ae_cfg['num_rounds']):
            # Clear session
            K.clear_session()
            
            # Initial weights
            if round_idx == 0:
                # Initial weight tensor
                loss_np, weight_values = np.ones((mod_size,)), np.ones((mod_size,))
            
            # Normalize and update weights
            loss_weights = K.expand_dims(K.variable(loss_np / np.sum(loss_np)))
            
            # Instantiate blank autoencoder
            ae, ae_list, enc, dec, dec_list = QuantizationAutoencoder(mod_size, ae_cfg['latent_dim'], 
                                                                  ae_cfg['num_layers'], ae_cfg['hidden_dim'],
                                                                  ae_cfg['common_layer'],
                                                                  ae_cfg['latent_layer'],
                                                                  ae_cfg['weight_reg'], local_seed,
                                                                  False, ae_cfg['noise_sigma'])
            # Local optimizer
            optimizer = Adam(lr=ae_cfg['learning_rate'], amsgrad=True)
            # Compile with custom weighted loss function
            ae.compile(optimizer=optimizer, loss=sample_balanced_wmse(eps=ae_cfg['global_eps'],
                                                                      weights=loss_weights))
            
            # Load last round weights and optimizer state
            if round_idx > 0:
                ae._make_train_function()
                ae.optimizer.set_weights(weight_values)
                ae.load_weights(global_dir + '/tmp_weights_seed%d.h5' % local_seed)
                
            # Create list of callbacks
            callbacks = [TerminateOnNaN()]
            # Train
            history = ae.fit(x=llr_train, y=llr_train, batch_size=ae_cfg['batch_size'],
                             epochs=ae_cfg['num_epochs_1'],
                             validation_data=(llr_val, llr_val), verbose=2,
                             callbacks=callbacks)
            
            # Write incrementally
            hdf5storage.savemat(local_dir + '/results.mat',
                                {'val_loss': history.history['val_loss']},
                                truncate_existing=True)
            
            # Evaluate on validation data
            rec_val = ae.predict(llr_val, batch_size=inf_batch_size)
            loss_np = sample_wmse_numpy(llr_val, rec_val, eps=ae_cfg['global_eps']) # This is sufficient
            # Print errors
            print('Per-output error is:' + str(loss_np))
            
            # Save weights and optimizer state
            symbolic_weights = getattr(ae.optimizer, 'weights')
            weight_values = K.batch_get_value(symbolic_weights)
            ae.save_weights(global_dir + '/tmp_weights_seed%d.h5' % local_seed)
        
        # Freeze encoder and finetune decoders
        if ae_cfg['freeze_enc']:
            enc.trainable = False
            
        # Recompile with slower learning rate and WMSE
        optimizer = Adam(lr=ae_cfg['learning_rate']/2, amsgrad=True)
        ae.compile(optimizer=optimizer, loss=sample_wmse(ae_cfg['global_eps']))
        
        # Early stop
        earlyStop = EarlyStopping(monitor='val_loss', patience=100, min_delta=1e-5,
                                  restore_best_weights=True)
        # Save best weights
        bestModel = ModelCheckpoint(local_dir + '/finetune_best.h5',
                                    verbose=0, save_best_only=True, 
                                    save_weights_only=True, period=1)
        
        # Train (fully parallel)
        history = ae.fit(x=llr_train, y=llr_train, batch_size=ae_cfg['batch_size'],
                         epochs=ae_cfg['num_epochs_2'],
                         validation_data=(llr_val, llr_val), verbose=2,
                         callbacks=[earlyStop, bestModel, TerminateOnNaN()])
        # Save last weights
        ae.save_weights(local_dir + '/finetune_last.h5')
        # Load best weights
        ae.load_weights(local_dir + '/finetune_best.h5')
        
        # Test performance
        rec_test = ae.predict(llr_test, batch_size=inf_batch_size)
        bler_ae[run_idx], ber_ae[run_idx], _ = decode_matlab_file(eng, 'ldpc', 
               rec_test, ref_bits, num_snr, num_codewords)
        
        # Get all latent validation data and test data
        latent_val  = enc.predict(llr_val, batch_size=inf_batch_size)
        latent_test = enc.predict(llr_test, batch_size=inf_batch_size)
    
        ### Train quantizers
        for idx, num_bits in enumerate(bits_per_dim):
            # Quantized representation
            latent_q = np.zeros(latent_test.shape)
            
            # One quantizer per dimension
            for dim_idx in range(ae_cfg['latent_dim']):
                # Fit
                kmeans = MiniBatchKMeans(n_clusters=2**num_bits, verbose=2,
                                         batch_size=8192, n_init=1000, max_no_improvement=1000)
                kmeans.fit(np.reshape(latent_val[:, dim_idx], (-1, 1)))
                
                # Save trained model to file
                joblib.dump(kmeans, local_dir + '/kmeans_dimension%d_bits%d.sav' % (
                        dim_idx, num_bits))
                
                # Extract codebook
                codebook = kmeans.cluster_centers_
                # Predict codebook index
                codebook_idx = kmeans.predict(np.reshape(latent_test[:,dim_idx], (-1, 1)))
                # Assign values from codebook
                latent_q[:, dim_idx] = np.squeeze(codebook[codebook_idx])
                
            # Test performance
            rec_test = dec.predict(latent_q, batch_size=inf_batch_size)
            bler_aeq[run_idx, idx], ber_aeq[run_idx, idx], _ = decode_matlab_file(eng, 'ldpc',
                    rec_test, ref_bits, num_snr, num_codewords)
    
    # Get all latent validation data and test data
    latent_val  = enc.predict(llr_val, batch_size=inf_batch_size)
    latent_test = enc.predict(llr_test, batch_size=inf_batch_size)
        
    ### Train sample encoder
    if not train_encoder:
        # Instantiate blank encoder
        sample_encoder = EstimationEncoder(mod_size, num_rx, num_tx,
                                               enc_cfg['num_blocks'],
                                               enc_cfg['latent_dim'], 
                                               enc_cfg['num_layers'], enc_cfg['hidden_dim'],
                                               enc_cfg['common_layer'], enc_cfg['latent_layer'],
                                               enc_cfg['weight_reg'],
                                               local_seed, verbose=False)
        # Load weights
        sample_encoder.load_weights(target_dir + '/seed%d/estimator_best.h5' % target_seed)
    else:
        # Get latent representation of training data
        latent_train = enc.predict(llr_train, batch_size=inf_batch_size)
        
        # Instantiate blank encoder
        sample_encoder = EstimationEncoder(mod_size, num_rx, num_tx,
                                               enc_cfg['num_blocks'],
                                               enc_cfg['latent_dim'], 
                                               enc_cfg['num_layers'], enc_cfg['hidden_dim'],
                                               enc_cfg['common_layer'], enc_cfg['latent_layer'],
                                               enc_cfg['weight_reg'],
                                               local_seed, verbose=False)
        
        # Reduce LR
        slowdown = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100,
                                     verbose=1)
        # Save best weights
        bestModel = ModelCheckpoint(local_dir + '/estimator_best.h5',
                                    verbose=0, save_best_only=True, save_weights_only=True, period=1)
        
        # Local optimizer
        optimizer = Adam(lr=enc_cfg['learning_rate'], amsgrad=True)
        # Compile with custom weighted loss function
        sample_encoder.compile(optimizer=optimizer, loss=enc_cfg['latent_loss'])
        
        # Create list of callbacks
        callbacks = [slowdown, bestModel, TerminateOnNaN()]
        
        # Train
        history = sample_encoder.fit(x=[y_train, h_train, n_train], y=latent_train,
                                     batch_size=enc_cfg['batch_size'],
                                     epochs=enc_cfg['num_epochs'],
                                     validation_data=([y_val, h_val, n_val], latent_val), verbose=2,
                                     callbacks=callbacks)
        # Save encoder loss
        hdf5storage.savemat(local_dir + '/estimator_history.mat',
                            {'history': history.history})
        # Load best weights
        sample_encoder.load_weights(local_dir + '/estimator_best.h5')
        
    ### Test sample encoder
    # Without quantization
    latent_test = sample_encoder.predict([y_test, h_test, n_test], batch_size=inf_batch_size)
    # Decode
    rec_test = dec.predict(latent_test, batch_size=inf_batch_size)
    bler_enc[run_idx], ber_enc[run_idx], _ = decode_matlab_file(eng, 'ldpc',
            rec_test, ref_bits, num_snr, num_codewords)
    
    # Test under quantization
    for idx, num_bits in enumerate(bits_per_dim):
        # Quantized representation
        latent_q = np.zeros(latent_test.shape)
        
        # One quantizer per dimension
        for dim_idx in range(ae_cfg['latent_dim']):
            if not train_quantizer:
                # Load pretrained model
                kmeans = joblib.load(target_dir + '/seed%d/kmeans_dimension%d_bits%d.sav' % (
                        target_seed, dim_idx, num_bits))
            else:
                kmeans = joblib.load(local_dir + '/kmeans_dimension%d_bits%d.sav' % (
                        dim_idx, num_bits))
            
            # Extract codebook
            codebook = kmeans.cluster_centers_
            # Predict codebook index
            codebook_idx = kmeans.predict(np.reshape(latent_test[:, dim_idx], (-1, 1)))
            # Assign values from codebook
            latent_q[:, dim_idx] = np.squeeze(codebook[codebook_idx])
            
        # Test performance
        rec_test = dec.predict(latent_q, batch_size=inf_batch_size)
        bler_encq[run_idx, idx], ber_encq[run_idx, idx], _ = decode_matlab_file(eng, 'ldpc',
                 rec_test, ref_bits, num_snr, num_codewords)
    
    # Store local results
    hdf5storage.savemat(local_dir + '/results.mat', {'bler_ae': bler_ae[run_idx],
                                                     'ber_ae': ber_ae[run_idx],
                                                     'bler_aeq': bler_aeq[run_idx],
                                                     'ber_aeq': ber_aeq[run_idx],
                                                     'bler_enc': bler_enc[run_idx],
                                                     'ber_enc': ber_enc[run_idx],
                                                     'bler_encq': bler_encq[run_idx],
                                                     'ber_encq': ber_encq[run_idx],
                                                     'val_loss': history.history['val_loss']
                                                     }, truncate_existing=True)
    # Store global results incrementally
    hdf5storage.savemat(global_dir + '/results_global%d.mat' % global_seed, {'bler_ae': bler_ae,
                                                 'ber_ae': ber_ae,
                                                 'bler_aeq': bler_aeq,
                                                 'ber_aeq': ber_aeq,
                                                 'bler_enc': bler_enc,
                                                 'ber_enc': ber_enc,
                                                 'bler_encq': bler_encq,
                                                 'ber_encq': ber_encq,
                                                 'local_seed_collect': local_seed_collect
                                                 }, truncate_existing=True)
    
# Close MATLAB engine
eng.stop()
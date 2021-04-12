# -*- coding: utf-8 -*-

from keras.layers import Input, Dense, GaussianNoise, Concatenate, Lambda
from keras.layers import Add, Activation
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.regularizers import l2
from keras import backend as K

import numpy as np

# Per sample and branch weighted MSE loss function
def sample_balanced_wmse(eps, weights):
    # Core function
    def loss(y_true, y_pred):
        return K.dot(1/(K.abs(y_true) + eps) * K.square(y_pred - y_true), weights)
    
    # Return function
    return loss

# Per sample weighted MSE loss function
def sample_wmse(eps):
    # Core function
    def loss(y_true, y_pred):
        return K.mean(1/(K.abs(y_true) + eps) * K.square(y_pred - y_true), axis=-1)
    
    # Return function
    return loss

# Numpy implementation of previous function
def sample_wmse_numpy(y_true, y_pred, eps):
    # Return value
    return np.mean(1/(np.abs(y_true) + eps) * np.square(y_pred - y_true), axis=0)

# Simple scalar quantizer for the latent space
def scalar_quantizer(input_latent, dim_bits, min_clip, max_clip):
    # Generate quantization array
    qValues = np.expand_dims(np.linspace(start=min_clip, stop=max_clip, num=2**dim_bits), axis=0)
    
    # Output
    input_shape = input_latent.shape
    
    # Serialize input
    input_ser = np.expand_dims(input_latent.flatten(), axis=1)
    
    # L2 distance matrix
    distMatrix = np.abs(input_ser - qValues) ** 2
    # Nearest index
    nearestIdx = np.argmin(distMatrix, axis=1)
    # Convert to quantized value
    out_q = qValues[0, nearestIdx]
    # Reshape
    out_q = np.reshape(out_q, input_shape)
    
    return out_q

# Nearest neighbour scalar quantizer
def codebook_quantizer(inputs, codebook):
    # Distance matrix
    dist_matrix = np.square(inputs[:, None] - codebook[None, :])
    # Nearest idex
    nearest_idx = np.argmin(dist_matrix, axis=-1)
    
    # Convert to quantized value
    out_q = codebook[nearest_idx]
    
    return out_q

# Instantiate a multiple-decoder autoencoder with the outputs merged
def QuantizationAutoencoder(mod_size, latent_dim, num_layers, hidden_dim,
                        common_layer, latent_layer, weight_reg, local_seed, 
                        verbose=False, noise_sigma=1e-3, passthrough=False):
    # NN parameters
    input_dim = mod_size
    
    # Local seed
    np.random.seed(local_seed)
    # Generate integers to seed each non-sigma layer
    seed_array = np.random.randint(low=0, high=2**31, size=2*num_layers)
    # Initializers
    weight_init = []
    for layer_idx in range(2*num_layers):
        weight_init.append(glorot_uniform(seed=seed_array[layer_idx]))
    
    # Weights regularizers
    l2_reg = weight_reg
    
    # Input layer
    input_bits = Input(shape=(input_dim,))
    # Universal encoder
    encoded = Dense(hidden_dim[0], activation=common_layer,
                    kernel_initializer=weight_init[0], kernel_regularizer=l2(l2_reg),
                    name='enc_layer0')(input_bits)
    for layer_idx in range(1, num_layers-1):
        encoded = Dense(hidden_dim[layer_idx], activation=common_layer,
                        kernel_initializer=weight_init[layer_idx], kernel_regularizer=l2(l2_reg),
                        name='enc_layer%d' % (layer_idx))(encoded)
    # Final layer is tanh activated
    encoded = Dense(latent_dim, activation=latent_layer,
                    kernel_initializer=weight_init[num_layers-1], kernel_regularizer=l2(l2_reg),
                    name='enc_layer%d' % (num_layers-1))(encoded)
    # If passthrough is enabled, quantize in forward pass
    if passthrough:
        encoded       = Lambda(lambda x: K.stop_gradient(K.sign(x) - x) + x)(encoded)
        encoded_noisy = encoded
    # Otherwise, add noise
    else:
        encoded_noisy = GaussianNoise(stddev=noise_sigma, name='noise_layer')(encoded)
    
    # List of decoders
    # Keep a list of outputs
    output_bit_list  = []
    for decoder_idx in range(mod_size):
        decoded = Dense(hidden_dim[-1], activation=common_layer, 
                        kernel_initializer=weight_init[num_layers], kernel_regularizer=l2(l2_reg),
                        name='dec_bit%d_layer0' % (decoder_idx))(encoded_noisy)
        for layer_idx in range(num_layers+1, 2*num_layers-1):
            decoded = Dense(hidden_dim[-(layer_idx-num_layers+1)], activation=common_layer,
                                       kernel_initializer=weight_init[layer_idx], kernel_regularizer=l2(l2_reg),
                                       name='dec_bit%d_layer%d' % (decoder_idx, layer_idx-num_layers))(decoded)
        # Final layer is tanh activated
        output_bit = Dense(1, activation='tanh',
                           kernel_initializer=weight_init[2*num_layers-1], kernel_regularizer=l2(l2_reg),
                           name='dec_bit%d_layer%d' % (decoder_idx, num_layers-1))(decoded)
        
        # Append to tensor list
        output_bit_list.append(output_bit)
        
    # Concatenate output tensors in single output
    output_bits = Concatenate(axis=-1)(output_bit_list)
    
    # this model maps end-to-end
    autoencoder = Model(input_bits, output_bits)
    # Save encoder model
    encoder = Model(input_bits, encoded)
    
    # Extract local decoder networks and local autoencoder networks
    decoder_list = []
    bit_list     = []
    # Global input
    local_decoder_input = Input(shape=(latent_dim,))
    for decoder_idx in range(mod_size):
        local_decoder = local_decoder_input
        # Stack layers by name starting from the latent representation
        for layer_idx in range(num_layers):
            local_decoder = autoencoder.get_layer(name='dec_bit%d_layer%d' % (decoder_idx, layer_idx))(local_decoder)
        # After stacking, save output
        bit_list.append(local_decoder)
        # Create local model
        local_decoder    = Model(inputs=local_decoder_input, outputs=local_decoder)
        # Append to list
        decoder_list.append(local_decoder)
        
    # Create local autoencoder models
    autoencoder_list = []
    for decoder_idx in range(mod_size):
        # One-output
        local_ae = Model(inputs=input_bits, outputs=output_bit_list[decoder_idx])
        autoencoder_list.append(local_ae)
    
    # Shadow concatentation
    bit_list = Concatenate(axis=-1)(bit_list)
    # Joint decoder
    decoder = Model(inputs=local_decoder_input, outputs=bit_list)
    
    # Print model summary
    if verbose:
        autoencoder.summary()
    
    return autoencoder, autoencoder_list, encoder, decoder, decoder_list

# Sample-to-latent direct encoder
def EstimationEncoder(mod_size, num_rx, num_tx, num_blocks,
                      latent_dim, num_layers, hidden_dim,
                      common_layer, latent_layer, weight_reg, 
                      local_seed, verbose=False):
    # Local seed
    np.random.seed(local_seed)
    # Generate integers to seed each non-sigma layer
    seed_array = np.random.randint(low=0, high=2**31, size=2*num_layers*num_blocks)
    # Initializers
    weight_init = []
    for layer_idx in range(2*num_layers*num_blocks):
        weight_init.append(glorot_uniform(seed=seed_array[layer_idx]))
    
    # Weights regularizers
    l2_reg = weight_reg
    
    # Signal inputs
    input_samples  = Input(shape=(2*num_rx,))        # Reals
    input_channels = Input(shape=(2*num_rx*num_tx,)) # Reals
    input_noise    = Input(shape=(1,))
    
    # All features get their own early layers
    encoded_samples = Dense(hidden_dim[0], activation=common_layer,
                            kernel_initializer=weight_init[0], kernel_regularizer=l2(l2_reg),
                            name='samples_layer0')(input_samples)
    encoded_channels = Dense(hidden_dim[0], activation=common_layer,
                             kernel_initializer=weight_init[0], kernel_regularizer=l2(l2_reg),
                             name='channels_layer0')(input_channels)
    encoded_noise = Dense(hidden_dim[0], activation=common_layer,
                             kernel_initializer=weight_init[0], kernel_regularizer=l2(l2_reg),
                             name='noise_layer0')(input_noise)
    
    # Merge
    encoded = Concatenate(axis=-1)([encoded_samples, encoded_channels, encoded_noise])
    
    # For each block
    for block_idx in range(num_blocks):
        # Collect residuals
        residuals = []
        
        # Hidden layers
        for layer_idx in range(block_idx*3, (block_idx+1)*3):
            encoded   = Dense(hidden_dim[layer_idx - block_idx*3], activation=common_layer,
                              kernel_initializer=weight_init[layer_idx], kernel_regularizer=l2(l2_reg),
                              name='enc_pre_layer%d' % (layer_idx))(encoded)
            residuals.append(Dense(hidden_dim[layer_idx - block_idx*3], activation=common_layer,
                             kernel_initializer=weight_init[layer_idx], kernel_regularizer=l2(l2_reg),
                             name='res_layer%d' % (layer_idx))(encoded))
        
        # Start adding residuals
        for layer_idx in range(block_idx*3, (block_idx+1)*3):
            encoded = Dense(hidden_dim[layer_idx - block_idx*3], activation='linear',
                            kernel_initializer=weight_init[layer_idx+3], kernel_regularizer=l2(l2_reg),
                            name='enc_post_layer%d' % (layer_idx))(encoded)
            encoded = Add()([encoded, residuals[layer_idx-block_idx*3]])
            # Activation
            encoded = Activation(common_layer)(encoded)
            
    # Final layer is linearly activated
    encoded = Dense(latent_dim, activation=latent_layer,
                    kernel_initializer=weight_init[-1], kernel_regularizer=l2(l2_reg),
                    name='output_layer')(encoded)
    
    # Save encoder model
    encoder = Model([input_samples, input_channels, input_noise],
                    encoded)
    
    # Print model summary
    if verbose:
        encoder.summary()
    
    return encoder

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.layers import Input, Dense, Concatenate, Lambda
from keras.layers import Activation, BatchNormalization, Dropout
from keras.layers import Reshape, Flatten
from keras.activations import softmax
from keras.models import Model
from keras import backend as K

import tensorflow as tf

# Return a model corresponding to a single iteration
def get_iteration_model(num_tx, num_rx, mod_size, constellation,
                        hidden_dim, is_last=False):

    # Calculate output dimension
    mod_order = 2 ** (mod_size // 2)
    output_dim = 2 * num_tx * mod_order
    
    # Constant tensor with constellation values (on single axis)
    tensor_constellation = tf.constant(constellation[None, None, None, ...], dtype=tf.float32)
    
    # Estimated x
    x_real = Input(shape=(num_tx,))
    x_imag = Input(shape=(num_tx,))
    # Projected y
    hy_real = Input(shape=(num_tx,))
    hy_imag = Input(shape=(num_tx,))
    # Projected x
    hx_real = Input(shape=(num_tx,))
    hx_imag = Input(shape=(num_tx,))
    
    # Concatenate previous estimates
    x_merged = Concatenate(axis=-1)([x_real, x_imag])
    
    # Input tensors for channel matrix
    h_real = Input(shape=(num_rx, num_tx,))
    h_imag = Input(shape=(num_rx, num_tx,))
    # Create complex tensor
    h_cplx = Lambda(lambda x: tf.complex(x[0], x[1]))([h_real, h_imag])
    # And canonical form
    h_dagger = Lambda(lambda x: tf.matmul(x, x, adjoint_a=True))(h_cplx)
    
    # Concatenate inputs
    input_features = Concatenate(axis=-1)([x_real, x_imag, hy_real, hy_imag, hx_real, hx_imag])
    
    # First layer: FC + BN + ReLU + DO
    hidden_features = Dense(hidden_dim[0], activation='linear')(input_features) 
    hidden_features = BatchNormalization(axis=-1)(hidden_features)
    hidden_features = Activation('relu')(hidden_features)
    hidden_features = Dropout(0.2)(hidden_features)
    
    # Second layer: FC + BN + ReLU
    hidden_features = Dense(hidden_dim[1], activation='linear')(hidden_features)
    hidden_features = BatchNormalization(axis=-1)(hidden_features)
    hidden_features = Activation('relu')(hidden_features)
    
    # Output layer
    output_features = Dense(output_dim, activation='linear')(hidden_features)
    # Reshape
    output_features = Reshape((2, num_tx, mod_order))(output_features)
    # Apply softmax per axis
    output_probs = Lambda(lambda x: softmax(x, axis=-1))(output_features)
    
    # If this is the last model, dead stop here
    if is_last:
        # Create model for one iteration
        core_model = Model(inputs=[x_real, x_imag, hy_real, hy_imag, hx_real, hx_imag, h_real, h_imag],
                           outputs=output_probs)
    else:
        # Probability to values
        output_values = Lambda(lambda x: tf.reduce_sum(tf.math.multiply(x, tensor_constellation), axis=-1))(output_probs)
        # Flatten (implicitly arranges them in real-imag order)
        output_values = Flatten()(output_values)
        
        # Pass previous estimates through T and C (see [30] for details)
        output_t = Dense(output_dim//mod_order, activation='sigmoid', use_bias=False)(x_merged)
        output_c = Dense(output_dim//mod_order, activation='sigmoid', use_bias=False)(x_merged)
        # Negate the C channel
        output_c = Lambda(lambda x: 1 - x)(output_c)
        
        # Build the highway output
        output_x = Lambda(lambda x: tf.math.multiply(x[0], x[1]) + tf.math.multiply(x[2], x[3]))(
                [output_values, output_t, x_merged, output_c])
        # Reshape and re-organize to real/imag
        output_x      = Reshape((2, num_tx))(output_x)
        output_x_real = Lambda(lambda x: x[:, 0, :])(output_x)
        output_x_imag = Lambda(lambda x: x[:, 1, :])(output_x)
        # Create complex version
        output_x_cplx = Lambda(lambda x: tf.complex(x[0], x[1]))([output_x_real, output_x_imag])
        
        # Multiply with canonical H
        # There's implicit expansion to column vector inside here, and then implicit flattening at the end
        output_hx = Lambda(lambda x: tf.matmul(x[0], tf.expand_dims(x[1], axis=-1))[..., 0])([h_dagger, output_x_cplx])
        
        # Split to real and imaginary
        output_hx_real = Lambda(lambda x: tf.real(x))(output_hx)
        output_hx_imag = Lambda(lambda x: tf.imag(x))(output_hx)
        
        # Create model for one iteration
        core_model = Model(inputs=[x_real, x_imag, hy_real, hy_imag, hx_real, hx_imag, h_real, h_imag],
                               outputs=[output_x_real, output_x_imag, hy_real, hy_imag,
                                        output_hx_real, output_hx_imag, h_real, h_imag])
    
    # Return the model
    return core_model


# Return an unrolled model
def get_complete_model(num_tx, num_rx, mod_size, constellation,
                       hidden_dim, num_iterations):

    # Get a sequence of models
    model_list = []
    for idx in range(num_iterations):
        # Get and add to list
        local_model = get_iteration_model(num_tx, num_rx, mod_size, constellation,
                                          hidden_dim, is_last=(idx==(num_iterations-1)))
        model_list.append(local_model)
    
    # Global real input tensors
    # Estimated x
    x_real = Input(shape=(num_tx,))
    x_imag = Input(shape=(num_tx,))
    # Projected y
    hy_real = Input(shape=(num_tx,))
    hy_imag = Input(shape=(num_tx,))
    # Projected x
    hx_real = Input(shape=(num_tx,))
    hx_imag = Input(shape=(num_tx,))
    # Channel matrix
    h_real = Input(shape=(num_rx, num_tx,))
    h_imag = Input(shape=(num_rx, num_tx,))
    
    # Pass through first iteration
    hidden_x_real, hidden_x_imag, hidden_hy_real, hidden_hy_imag,\
    hidden_hx_real, hidden_hx_imag, hidden_h_real, hidden_h_imag = \
    model_list[0]([x_real, x_imag, hy_real, hy_imag, hx_real, hx_imag, h_real, h_imag])
    
    # Pass through each iteration - except last
    for idx in range(1, num_iterations-1):
        hidden_x_real, hidden_x_imag, hidden_hy_real, hidden_hy_imag,\
        hidden_hx_real, hidden_hx_imag, hidden_h_real, hidden_h_imag  = \
        model_list[idx]([hidden_x_real, hidden_x_imag, hidden_hy_real, hidden_hy_imag,
                  hidden_hx_real, hidden_hx_imag, hidden_h_real, hidden_h_imag])
    
    # Pass through last iteration
    output_probs = model_list[-1]([hidden_x_real, hidden_x_imag, hidden_hy_real, hidden_hy_imag,
                                 hidden_hx_real, hidden_hx_imag, hidden_h_real, hidden_h_imag])
    
    # Create global model
    global_model = Model(inputs=[x_real, x_imag, hy_real, hy_imag, hx_real, hx_imag, h_real, h_imag],
                           outputs=[hidden_x_real, hidden_x_imag, hidden_hy_real, hidden_hy_imag,
                                    hidden_hx_real, hidden_hx_imag, hidden_h_real, hidden_h_imag])
    
    # Create a model which only outputs probabilities
    global_prob_model = Model(inputs=[x_real, x_imag, hy_real, hy_imag, hx_real, hx_imag, h_real, h_imag],
                              outputs=output_probs)
    
    # Return the model
    return global_model, global_prob_model

# Define a custom loss
def custom_loss(y_true, y_pred):
    return K.sum(K.categorical_crossentropy(y_true, y_pred, axis=-1))
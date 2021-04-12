#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.layers import Input, Concatenate, Lambda
from keras.layers import Layer
from keras.models import Model

import numpy as np
import tensorflow as tf

# Custom trainable layer with scalar variable
class Scalar(Layer):

    def __init__(self, output_dim=1, name='unknown', init_values='ones', **kwargs):
        self.output_dim = output_dim
        self.init_values = init_values
        self.name = name
        super(Scalar, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer
        self.kernel = self.add_weight(name=self.name,
                                      shape=(1, 1),
                                      initializer=self.init_values,
                                      trainable=True)
        super(Scalar, self).build(input_shape)

    def call(self, x):
        return x * tf.complex(self.kernel, 0.)

    def compute_output_shape(self, input_shape):
        return input_shape

# Return a model corresponding to a single iteration
def get_iteration_model(num_tx, num_rx, mod_size, constellation,
                        sigma_n, is_last=False, debug=True):
    
    # Constant tensor with constellation values (on single axis)
    tensor_constellation = tf.constant(constellation, dtype=tf.complex64)
    
    # Input x
    x_real = Input(shape=(num_tx,))
    x_imag = Input(shape=(num_tx,))
    # Input y
    y_real = Input(shape=(num_rx,))
    y_imag = Input(shape=(num_rx,))
    
    # Input v
    v_t = Input(shape=(1,))
    # Safety check for v
    v_t_safe = Lambda(lambda x: tf.maximum(x, 5e-13))(v_t)
    v_t_cplx = Lambda(lambda x: tf.complex(x, 0.))(v_t_safe)
    # Input tau
    tau_t = Input(shape=(1,))
    
    # Input tensors for channel matrix
    h_real = Input(shape=(num_rx, num_tx,))
    h_imag = Input(shape=(num_rx, num_tx,))
    # Create complex tensors
    x_cplx = Lambda(lambda x: tf.complex(x[0], x[1])[..., None])([x_real, x_imag])
    y_cplx = Lambda(lambda x: tf.complex(x[0], x[1])[..., None])([y_real, y_imag])
    h_cplx = Lambda(lambda x: tf.complex(x[0], x[1]))([h_real, h_imag])
    
    # Canonical forms of H
    h_inner = Lambda(lambda x: tf.matmul(x, x, adjoint_a=True))(h_cplx)
    h_outer = Lambda(lambda x: tf.matmul(x, x, adjoint_b=True))(h_cplx)
    # Compute trace(H'H)
    h_trace = Lambda(lambda x: tf.real(tf.linalg.trace(x)[..., None]))(h_inner)
    
    # Compute auxiliary residual
    y_residual = Lambda(lambda x: x[0] - tf.matmul(x[1], x[2]))([y_cplx, h_cplx, x_cplx])

    # Compute W-hat with auxiliaries
    ax_1 = Lambda(lambda x: tf.math.multiply((x[0])[..., None], x[1]))([v_t_cplx, h_outer])
    ax_2 = tf.constant(sigma_n * np.eye(num_rx), dtype=tf.complex64)
    
    W_hat_t = Lambda(lambda x: tf.cast(x[0], tf.complex64)[..., None] * tf.matmul(
            tf.cast(tf.linalg.adjoint(x[1]), tf.complex64),
            tf.linalg.inv(tf.math.add(tf.cast(x[2], tf.complex64), ax_2))))(
    [v_t_cplx, h_cplx, ax_1])
    
    # Compute trace separately
    aux_trace = Lambda(lambda x: tf.complex(tf.real(tf.linalg.trace(tf.matmul(x[0], x[1]))[..., None, None]), 0.))(
            [W_hat_t, h_cplx])
    # Compute W
    W_t = Lambda(lambda x: x[0] * num_tx / x[1])([W_hat_t, aux_trace])
    
    # Multiply W_t with y_residual
    encoded = Lambda(lambda x: tf.matmul(x[0], x[1]))([W_t, y_residual])
    
    # Multiply with trainable gamma
    encoded = Scalar(init_values='ones', name='gamma')(encoded)
    # Compute residual
    encoded = Lambda(lambda x: x[0] + x[1])([x_cplx, encoded])
    
    # Compute decoupled MMSE estimate
    mmse_list = []
    for symbol_idx in range(num_tx):
        # Get local entry
        encoded_local = Lambda(lambda x: x[:, symbol_idx, :])(encoded)
        # Compute gaussian pdf at all constellation symbols with mean given by current entry and variance by tau
        encoded_dist = Lambda(lambda x: tf.math.exp(-tf.math.square(tf.math.abs(
                tensor_constellation - tf.cast(x[0], tf.complex64))) / tf.cast(x[1], tf.float32) ))(
                [encoded_local, tau_t])
        
        # Cast distances to complex
        encoded_cplx_dist = Lambda(lambda x: tf.cast(x, tf.complex64))(encoded_dist)
        # Compute MMSE estimate
        encoded_mmse = Lambda(lambda x: tf.reduce_sum(tf.math.multiply(tensor_constellation, tf.cast(x, tf.complex64)),
                                                      axis=-1, keepdims=True) /
                              tf.reduce_sum(tf.cast(x, tf.complex64), axis=-1, keepdims=True))(encoded_cplx_dist)
        # Add to list
        mmse_list.append(encoded_mmse)
        
    # Concatenate
    mmse = Concatenate(axis=-1)(mmse_list)
    mmse = Lambda(lambda x: tf.expand_dims(x, axis=-1))(mmse)
    
    # Multiply residual with trainable epsilon
    encoded_weighted = Scalar(init_values='zeros', name='epsilon')(encoded)
    # Compute nonlinear estimator
    nonlinear_est = Lambda(lambda x: x[0] - x[1])([mmse, encoded_weighted])
    # Multiply with phi
    x_out = Scalar(init_values='ones', name='phi')(nonlinear_est)
    # Separate in real/imaginary
    x_out_real = Lambda(lambda x: tf.real(x[..., 0]))(x_out)
    x_out_imag = Lambda(lambda x: tf.imag(x[..., 0]))(x_out)
    
    # Create core model for one iteration
    if is_last:
        if debug:
            # Directly output x and everything else
            core_model = Model(inputs=[x_real, x_imag, y_real, y_imag, h_real, h_imag, v_t, tau_t],
                               outputs=[x_out_real, x_out_imag, y_residual, W_hat_t, W_t, mmse, nonlinear_est])
        else:
            # Directly output x
            core_model = Model(inputs=[x_real, x_imag, y_real, y_imag, h_real, h_imag, v_t, tau_t],
                               outputs=[x_out_real, x_out_imag])
    else:
        # Update v
        v_out = Lambda(lambda x: (tf.square(tf.cast(tf.linalg.norm(x[0], axis=-2), tf.float32)) - num_rx * sigma_n) / x[1])(
                [y_residual, h_trace])
        # Update tau
        # First, instantiate the theta layer
        theta_layer = Scalar(init_values='ones', name='theta')
        # Compute WH
        wh_prod = Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([W_t, h_cplx])
        # Scale with theta
        wh_scaled = theta_layer(wh_prod)
        # Compute C with auxiliaries
        ax_3 = tf.constant(np.eye(num_tx), dtype=tf.complex64)
        C_matrix = Lambda(lambda x: ax_3 - tf.cast(x, tf.complex64))(wh_scaled)
        # Part one of tau
        ax_4  = Lambda(lambda x: tf.cast(x, tf.complex64))(v_out)
        tau_1 = Lambda(lambda x: x[1] * tf.linalg.trace(tf.linalg.matmul(
                x[0], x[0], adjoint_b=True))[..., None] / num_tx)([C_matrix, ax_4])
        
        # Part two of tau - before scaling
        tau_2_unscaled = Lambda(lambda x: tf.linalg.trace(tf.linalg.matmul(x, x, adjoint_b=True))[..., None]
        * sigma_n/num_tx)(W_t)
        # Scale by passing twice through layer
        tau_2 = theta_layer(theta_layer(tau_2_unscaled))
        
        # Add two parts
        tau_out = Lambda(lambda x: x[0] + x[1])([tau_1, tau_2])
        # Get real part
        tau_out = Lambda(lambda x: tf.math.real(x))(tau_out)
        
        # Create model
        core_model = Model([x_real, x_imag, y_real, y_imag, h_real, h_imag, v_t, tau_t],
                           [x_out_real, x_out_imag, v_out, tau_out])
    
    # Return the model
    return core_model

# Return a list of processed tensors corresponding to a single iteration
def get_iteration_tensor_model(x_real, x_imag, y_real, y_imag, h_real, h_imag,
                               v_t, tau_t, num_tx, num_rx, mod_size, constellation,
                               sigma_n, is_last=False, debug=True):
    
    # Constant tensor with constellation values (on single axis)
    tensor_constellation = tf.constant(constellation, dtype=tf.complex64)
    
    # Safety check for v
    v_t_cplx = Lambda(lambda x: tf.complex(x, 0.))(v_t)
    
    # Create complex tensors
    x_cplx = Lambda(lambda x: tf.complex(x[0], x[1])[..., None])([x_real, x_imag])
    y_cplx = Lambda(lambda x: tf.complex(x[0], x[1])[..., None])([y_real, y_imag])
    h_cplx = Lambda(lambda x: tf.complex(x[0], x[1]))([h_real, h_imag])
    
    # Canonical forms of H
    h_inner = Lambda(lambda x: tf.matmul(x, x, adjoint_a=True))(h_cplx)
    h_outer = Lambda(lambda x: tf.matmul(x, x, adjoint_b=True))(h_cplx)
    # Compute trace(H'H)
    h_trace = Lambda(lambda x: tf.real(tf.linalg.trace(x)[..., None]))(h_inner)
    
    # Compute auxiliary residual
    y_residual = Lambda(lambda x: x[0] - tf.matmul(x[1], x[2]))([y_cplx, h_cplx, x_cplx])

    # Compute W-hat with auxiliaries
    ax_1 = Lambda(lambda x: tf.math.multiply((x[0])[..., None], x[1]))([v_t_cplx, h_outer])
    ax_2 = tf.constant(sigma_n * np.eye(num_rx), dtype=tf.complex64)
    
    W_hat_t = Lambda(lambda x: tf.cast(x[0], tf.complex64)[..., None] * tf.matmul(
            tf.cast(tf.linalg.adjoint(x[1]), tf.complex64),
            tf.linalg.inv(tf.math.add(tf.cast(x[2], tf.complex64), ax_2))))(
    [v_t_cplx, h_cplx, ax_1])
    
    # Compute trace separately
    aux_trace = Lambda(lambda x: tf.complex(tf.real(tf.linalg.trace(tf.matmul(x[0], x[1]))[..., None, None]), 0.))(
            [W_hat_t, h_cplx])
    # Compute W
    W_t = Lambda(lambda x: x[0] * num_tx / x[1])([W_hat_t, aux_trace])
    
    # Multiply W_t with y_residual
    encoded = Lambda(lambda x: tf.matmul(x[0], x[1]))([W_t, y_residual])
    
    # Multiply with trainable gamma
    encoded = Scalar(init_values='ones', name='gamma')(encoded)
    # Compute residual
    encoded = Lambda(lambda x: x[0] + x[1])([x_cplx, encoded])
    
    # Compute decoupled MMSE estimate
    mmse_list = []
    for symbol_idx in range(num_tx):
        # Get local entry
        encoded_local = Lambda(lambda x: x[:, symbol_idx, :])(encoded)
        # Compute gaussian pdf at all constellation symbols with mean given by current entry and variance by tau
        encoded_dist = Lambda(lambda x: tf.math.exp(-tf.math.square(tf.math.abs(
                tensor_constellation - tf.cast(x[0], tf.complex64))) / tf.cast(x[1], tf.float32) ))(
                [encoded_local, tau_t])
        
        # Cast distances to complex
        encoded_cplx_dist = Lambda(lambda x: tf.cast(x, tf.complex64))(encoded_dist)
        # Compute MMSE estimate
        encoded_mmse = Lambda(lambda x: tf.reduce_sum(tf.math.multiply(tensor_constellation, tf.cast(x, tf.complex64)),
                                                      axis=-1, keepdims=True) /
                              tf.reduce_sum(tf.cast(x, tf.complex64), axis=-1, keepdims=True))(encoded_cplx_dist)
        # Add to list
        mmse_list.append(encoded_mmse)
        
    # Concatenate
    mmse = Concatenate(axis=-1)(mmse_list)
    mmse = Lambda(lambda x: tf.expand_dims(x, axis=-1))(mmse)
    
    # Multiply residual with trainable epsilon
    encoded_weighted = Scalar(init_values='zeros', name='epsilon')(encoded)
    # Compute nonlinear estimator
    nonlinear_est = Lambda(lambda x: x[0] - x[1])([mmse, encoded_weighted])
    # Multiply with phi
    x_out = Scalar(init_values='ones', name='phi')(nonlinear_est)
    # Separate in real/imaginary
    x_out_real = Lambda(lambda x: tf.real(x[..., 0]))(x_out)
    x_out_imag = Lambda(lambda x: tf.imag(x[..., 0]))(x_out)
    
    # Create core model for one iteration
    if is_last:
        if debug:
            # Directly output x and everything else
            return x_out_real, x_out_imag, y_residual, W_hat_t, W_t, mmse, nonlinear_est
        else:
            # Directly output x
            return x_out_real, x_out_imag
    else:
        # Update v
        v_out = Lambda(lambda x: (tf.square(tf.cast(tf.linalg.norm(x[0], axis=-2), tf.float32)) - num_rx * sigma_n) / x[1])(
                [y_residual, h_trace])
        # Safety check
        v_out = Lambda(lambda x: tf.maximum(x, 5e-13))(v_out)
        # Update tau
        # First, instantiate the theta layer
        theta_layer = Scalar(init_values='ones', name='theta')
        # Compute WH
        wh_prod = Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([W_t, h_cplx])
        # Scale with theta
        wh_scaled = theta_layer(wh_prod)
        # Compute C with auxiliaries
        ax_3 = tf.constant(np.eye(num_tx), dtype=tf.complex64)
        C_matrix = Lambda(lambda x: ax_3 - tf.cast(x, tf.complex64))(wh_scaled)
        # Part one of tau
        ax_4  = Lambda(lambda x: tf.cast(x, tf.complex64))(v_out)
        tau_1 = Lambda(lambda x: x[1] * tf.linalg.trace(tf.linalg.matmul(
                x[0], x[0], adjoint_b=True))[..., None] / num_tx)([C_matrix, ax_4])
        
        # Part two of tau - before scaling
        tau_2_unscaled = Lambda(lambda x: tf.linalg.trace(tf.linalg.matmul(x, x, adjoint_b=True))[..., None]
        * sigma_n/num_tx)(W_t)
        # Scale by passing twice through layer
        tau_2 = theta_layer(theta_layer(tau_2_unscaled))
        
        # Add two parts
        tau_out = Lambda(lambda x: x[0] + x[1])([tau_1, tau_2])
        # Get real part
        tau_out = Lambda(lambda x: tf.math.real(x))(tau_out)
        
        # Create model
        return x_out_real, x_out_imag, v_out, tau_out

# Return an unrolled model
def get_complete_model(num_tx, num_rx, mod_size, constellation,
                       sigma_n, num_iterations):

    # Get a sequence of models
    model_list = []
    for idx in range(num_iterations):
        # Get and add to list
        local_model = get_iteration_model(num_tx, num_rx, mod_size, constellation,
                                          sigma_n, is_last=(idx==(num_iterations-1)),
                                          debug=False)
        model_list.append(local_model)
    
    # Global real input tensors
    # Input x
    x_real = Input(shape=(num_tx,))
    x_imag = Input(shape=(num_tx,))
    # Input y
    y_real = Input(shape=(num_rx,))
    y_imag = Input(shape=(num_rx,))
    # Input h
    h_real = Input(shape=(num_rx, num_tx,))
    h_imag = Input(shape=(num_rx, num_tx,))
    # Input v
    v = Input(shape=(1,))
    # Input tau
    tau = Input(shape=(1,))
    
    # Pass through first iteration
    if num_iterations == 1:
        # Return model directly
        return model_list[0]
    else:
        hidden_x_real, hidden_x_imag, hidden_v, hidden_tau = \
        model_list[0]([x_real, x_imag, y_real, y_imag, h_real, h_imag, v, tau])
    
        # Pass through each iteration - except last
        if num_iterations > 2:
            for idx in range(1, num_iterations-1):
                hidden_x_real, hidden_x_imag, hidden_v, hidden_tau = \
                model_list[idx]([hidden_x_real, hidden_x_imag, y_real, y_imag,
                          h_real, h_imag, hidden_v, hidden_tau])
        
        # Pass through last iteration
        output_x_real, output_x_imag = model_list[-1]([hidden_x_real, hidden_x_imag, y_real, y_imag,
                                               h_real, h_imag, hidden_v, hidden_tau])
    
    # Create global model
    global_model = Model(inputs=[x_real, x_imag, y_real, y_imag, h_real, h_imag, v, tau],
                         outputs=[output_x_real, output_x_imag])
    
    # Return the model
    return global_model

# Return an unrolled model
def get_complete_tensor_model(num_tx, num_rx, mod_size, constellation,
                              sigma_n, num_iterations):

    # Global real input tensors
    # Input x
    x_real = Input(shape=(num_tx,))
    x_imag = Input(shape=(num_tx,))
    # Input y
    y_real = Input(shape=(num_rx,))
    y_imag = Input(shape=(num_rx,))
    # Input h
    h_real = Input(shape=(num_rx, num_tx,))
    h_imag = Input(shape=(num_rx, num_tx,))
    # Input v
    v = Input(shape=(1,))
    # Input tau
    tau = Input(shape=(1,))
    
    # Pass through a sequence of models
    if num_iterations == 1:
        output_x_real, output_x_imag = get_iteration_tensor_model(x_real, x_imag, y_real, y_imag, 
                                                                  h_real, h_imag,
                                                                  v, tau, num_tx, num_rx, 
                                                                  mod_size, constellation,
                                                                  sigma_n, is_last=True, debug=False)
    else:
        # First hidden layer
        hidden_x_real, hidden_x_imag, hidden_v, hidden_tau = \
        get_iteration_tensor_model(x_real, x_imag, y_real, y_imag, 
                                   h_real, h_imag,
                                   v, tau, num_tx, num_rx, 
                                   mod_size, constellation,
                                   sigma_n, is_last=False, debug=False)
        # Other hidden layers
        for idx in range(1, num_iterations-1):
            # Hidden layers
            hidden_x_real, hidden_x_imag, hidden_v, hidden_tau = \
            get_iteration_tensor_model(hidden_x_real, hidden_x_imag, y_real, y_imag, 
                                       h_real, h_imag,
                                       hidden_v, hidden_tau, num_tx, num_rx, 
                                       mod_size, constellation,
                                       sigma_n, is_last=False, debug=False)
        # Output
        output_x_real, output_x_imag = get_iteration_tensor_model(hidden_x_real, hidden_x_imag, y_real, y_imag, 
                                                                  h_real, h_imag,
                                                                  hidden_v, hidden_tau, num_tx, num_rx, 
                                                                  mod_size, constellation,
                                                                  sigma_n, is_last=True, debug=False)
    
    
    global_model = Model(inputs=[x_real, x_imag, y_real, y_imag, h_real, h_imag, v, tau],
                         outputs=[output_x_real, output_x_imag])
    
    # Return the model
    return global_model
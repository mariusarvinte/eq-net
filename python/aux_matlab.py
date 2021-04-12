#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hdf5storage
import os
import numpy as np
# This instantiates local generators for decoupled randomness
from numpy.random import Generator, PCG64

# Perform a single-shot decode, while passing through the file system
def decode_matlab_file(eng, code_type, input_llr, ref_bits, num_snr, num_codewords, mode='soft'):
    # Static path
    # !!!: This is the full path of the decoder .m function
    decoder_path = 'InsertPathToMatlabDecoder.m'

    # Draw a random integer (independent of global seed)
    rg         = Generator(PCG64())
    random_idx = np.round(1e10 * rg.standard_normal()).astype(np.int)
    # Filenames
    filename_in  = 'scratch/in%d.mat' % random_idx
    filename_out = 'scratch/out%d.mat' % random_idx
    
    # Copy
    input_llr = np.copy(input_llr)
    # Restore and reshape
    if mode == 'soft':
        input_llr = 2 * np.arctanh(input_llr)
    input_llr = np.reshape(input_llr, (num_snr, num_codewords, -1))
    
    # Write to input file
    hdf5storage.savemat(filename_in, {'llr_input': input_llr,
                                      'code_type': code_type})
        
    # Create input dictionary
    args = {'filename_in': filename_in,
            'filename_out': filename_out}
    # Call decoder
    _ = eng.run_func(decoder_path, args)
    
    # Read output file
    contents = hdf5storage.loadmat(filename_out)
    rec_bits = contents['bits_out']
    
    # Convert to arrays
    rec_bits = np.asarray(rec_bits)

    # Compute error rates
    bler = np.mean(np.any(rec_bits != ref_bits, axis=-1), axis=-1)
    ber  = np.mean(np.mean(rec_bits != ref_bits, axis=-1), axis=-1)
    
    # Delete files
    os.remove(filename_in)
    os.remove(filename_out)
    
    return bler, ber, rec_bits

# Train an MI quantizer with an unrolled LLR vector
def train_mi_quantizer(eng, input_llr, mod_size, num_bits, num_tx, num_rx,
                       train=True):
    # Static paths
    trainer_path = '/home/yanni/marius/spawc2015-master/designQuantizerFile.m'
    
    # Draw a random integer (independent of global seed)
    rg         = Generator(PCG64())
    random_idx = np.round(1e10 * rg.standard_normal()).astype(np.int)
    # Filenames
    filename_in  = '/home/yanni/marius/spawc2015-master/scratch/in%d.mat' % random_idx
    filename_out = '/home/yanni/marius/spawc2015-master/quantizers/sota_mimo%dby%d_qam%d_bits%d.mat' % (
            num_rx, num_tx, mod_size, num_bits)
    
    # Training from scratch
    if train:
        # Copy
        input_llr = np.copy(input_llr)
        # Write to input file
        hdf5storage.savemat(filename_in, {'ref_llr': input_llr})
        
        # Create input dictionary
        args = {'filename_in': filename_in,
                'filename_out': filename_out,
                'mod_size': mod_size,
                'num_bits': num_bits}
        # Call decoder
        _ = eng.run_func(trainer_path, args)
        
        # Delete files
        os.remove(filename_in)
        
    # Load codebook
    contents = hdf5storage.loadmat(filename_out)
    codebook = contents['LLRs']
    
    return codebook
import numpy as np
import pandas as pd
import tensorflow as tf

import utils

def load_data_muchisimocks3D(tag_mocks, n_train, n_val, n_test,
                             batch_size=32, n_threads=8):       
    
    if n_threads is not None:
        tf.config.threading.set_inter_op_parallelism_threads(n_threads)
        tf.config.threading.set_intra_op_parallelism_threads(n_threads)
    
    print("Loading 3D data")
    dir_params = '../data/params'
    fn_params = f'{dir_params}/params_lh{tag_mocks}.txt'
    params_df = pd.read_csv(fn_params, index_col=0)
    param_names = params_df.columns.tolist()
    idxs_LH = params_df.index.tolist()

    dir_mocks = f'/scratch/kstoreyf/muchisimocks/muchisimocks_lib{tag_mocks}'
    tag_fields = '_deconvolved'
    tag_fields_extra = ''
    n_grid_orig = 512
    # TODO deal with bias vector better!
    bias_vector = np.array([1.,0.,0.,0.])

    fn_rands = f'{dir_params}/randints{tag_mocks}.npy'
    random_ints = np.load(fn_rands)

    fns_fields, theta = [], []
    for idx_LH in idxs_LH:
        fns_fields.append( f'{dir_mocks}/LH{idx_LH}/bias_fields_eul{tag_fields}_{idx_LH}{tag_fields_extra}.npy' )
        # TODO what is my error??
        param_vals = params_df.loc[idx_LH].values
        theta.append(param_vals)

    fns_fields = np.array(fns_fields)
    theta = np.array(theta)
    #y_err = np.array(y_err)
        
    frac_train=0.8
    frac_val=0.1
    frac_test=0.1
        
    idxs_train, idxs_val, idxs_test = utils.idxs_train_val_test(random_ints, 
                                   frac_train=frac_train, frac_val=frac_val, frac_test=frac_test)
    idxs_train = idxs_train[:n_train]
    idxs_val = idxs_val[:n_val]
    idxs_test = idxs_test[:n_test]

    theta_train, theta_val, theta_test = utils.split_train_val_test(theta, idxs_train, idxs_val, idxs_test)
    fns_train, fns_val, fns_test = utils.split_train_val_test(fns_fields, idxs_train, idxs_val, idxs_test)
    #y_err_train, y_err_val, y_err_test = utils.split_train_val_test(y_err, idxs_train, idxs_val, idxs_test)

    print("load_data_muchisimocks3D")
    print(len(fns_train), len(fns_val), len(fns_test))
    print(idxs_train[:10], idxs_val[:10], idxs_test[:10])

    ds_fns_train = tf.data.Dataset.from_tensor_slices(fns_train)
    ds_theta_train = tf.data.Dataset.from_tensor_slices(theta_train)
    dataset_train = tf.data.Dataset.zip((ds_fns_train, ds_theta_train))
    
    ds_fns_val = tf.data.Dataset.from_tensor_slices(fns_val)
    ds_theta_val = tf.data.Dataset.from_tensor_slices(theta_val)
    dataset_val = tf.data.Dataset.zip((ds_fns_val, ds_theta_val))
    
    ds_fns_test = tf.data.Dataset.from_tensor_slices(fns_test)
    ds_theta_test = tf.data.Dataset.from_tensor_slices(theta_test)
    dataset_test = tf.data.Dataset.zip((ds_fns_test, ds_theta_test))
        
    # Step 3: Process the dataset

    def process_fn(fn_fields, theta_val):
        print('processing', fn_fields)
        bias_terms_eul = np.load(fn_fields.numpy().decode("utf-8"))
        print("loaded")
        tracer_field = utils.get_tracer_field(bias_terms_eul, bias_vector, n_grid_norm=n_grid_orig)
        print(np.min(tracer_field), np.max(tracer_field))
        tracer_field = tf.convert_to_tensor(tracer_field, dtype=tf.float32)
        #theta_val = tf.convert_to_tensor(np.float32(theta_val), dtype=tf.float32)
        theta_val = tf.convert_to_tensor(theta_val, dtype=tf.float64)
        theta_val = tf.cast(theta_val, tf.float32)
        print(tracer_field.shape, theta_val.shape)
        return tracer_field, theta_val

    # Wrapper function for use with tf.py_function
    def process_fn_py(fn_fields, theta_val):
        tracer_field, theta_val = tf.py_function(func=process_fn, inp=[fn_fields, theta_val], Tout=(tf.float32, tf.float32))
        #image.set_shape([224, 224, 3])  # Set the shape explicitly to [224, 224, 3]
        return tracer_field, theta_val

    # Apply the processing function to the dataset
    # dataset_train = dataset_train.map(process_fn_py, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset_val = dataset_val.map(process_fn_py, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset_test = dataset_test.map(process_fn_py, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print("Set up processing")
    dataset_train = dataset_train.map(lambda fn_fields, label: tf.py_function(
            process_fn_py, [fn_fields, label], [tf.float32, tf.float32]), 
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_val = dataset_val.map(lambda fn_fields, label: tf.py_function(
            process_fn_py, [fn_fields, label], [tf.float32, tf.float32]), 
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_test = dataset_test.map(lambda fn_fields, label: tf.py_function(
            process_fn_py, [fn_fields, label], [tf.float32, tf.float32]), 
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Step 4: Configure dataset for performance
    # print("Shuffle")
    # dataset_train = dataset_train.shuffle(buffer_size=len(fns_train)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    # dataset_val = dataset_val.shuffle(buffer_size=len(fns_val)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    # dataset_test = dataset_test.shuffle(buffer_size=len(fns_test)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset_train, dataset_val, dataset_test, param_names


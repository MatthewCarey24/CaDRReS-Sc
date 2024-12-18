import pandas as pd
import numpy as np
import os, pickle, time

import tensorflow as tf
from tensorflow.python.framework import ops


def load_model(model_fname):
    """Load a pre-trained model
    :param model_fname: File name of the model
    :return: model_dict contains model information
    """
    with open(model_fname, 'rb') as f:
        model_dict = pickle.load(f)
    return model_dict


def predict_from_model(model_dict, test_kernel_df, model_spec_name='cadrres-wo-sample-bias'):
    """
    Make a prediction of testing samples. Only for the model without sample bias.
    """

    if model_spec_name not in ['cadrres-wo-sample-bias', 'cadrres-wo-sample-bias-weight']:
        return None

    sample_list = list(test_kernel_df.index)

    # Read drug list from model_dict
    drug_list = model_dict['drug_list']
    kernel_sample_list = model_dict['kernel_sample_list']

    # Prepare input
    X = np.array(test_kernel_df[kernel_sample_list])

    # Make a prediction
    b_q = model_dict['b_Q']
    WP = model_dict['W_P']
    WQ = model_dict['W_Q']
    n_dim = WP.shape[1]

    pred = b_q.T + np.matmul(np.matmul(X, WP), WQ.T)
    pred = pred * -1  # convert sensitivity score to IC50
    pred_df = pd.DataFrame(pred, sample_list, drug_list)

    # Projections
    P_test = np.matmul(X, WP)
    P_test_df = pd.DataFrame(P_test, index=sample_list, columns=range(1, n_dim + 1))

    return pred_df, P_test_df


def create_placeholders(n_x_features, n_y_features, sample_weight=False):
    """
    Create placeholders for model inputs
    """

    # TensorFlow 2.x uses eager execution by default. Use tf.keras.Input instead.
    X = tf.keras.Input(shape=(n_x_features,), dtype=tf.float32, name="X")
    Y = tf.keras.Input(shape=(n_y_features,), dtype=tf.float32, name="Y")

    if sample_weight:
        O = tf.keras.Input(shape=(None,), dtype=tf.float32, name="O")
        D = tf.keras.Input(shape=(None,), dtype=tf.float32, name="D")
        return X, Y, O, D
    else:
        return X, Y


def initialize_parameters(n_samples, n_drugs, n_x_features, n_y_features, n_dimensions, seed=None):
    """Initialize parameters."""
    tf.random.set_seed(seed)  # Ensure reproducibility if seed is provided

    initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.2, seed=seed)
    parameters = {
        'W_P': tf.Variable(initializer([n_x_features, n_dimensions]), name="W_P"),
        'W_Q': tf.Variable(initializer([n_y_features, n_dimensions]), name="W_Q"),
        'b_P': tf.Variable(tf.zeros([n_samples, 1]), name="b_P"),
        'b_Q': tf.Variable(tf.zeros([n_drugs, 1]), name="b_Q"),
    }
    return parameters


def inward_propagation(X, Y, parameters, n_samples, n_drugs, model_spec_name):
    """Define base objective function."""
    W_P = parameters['W_P']
    W_Q = parameters['W_Q']
    P = tf.matmul(X, W_P)
    Q = tf.matmul(Y, W_Q)

    b_P_mat = tf.matmul(parameters['b_P'], tf.ones((1, n_drugs), dtype=tf.float32))
    b_Q_mat = tf.transpose(tf.matmul(parameters['b_Q'], tf.ones((1, n_samples), dtype=tf.float32)))

    if model_spec_name == 'cadrres':
        S = b_Q_mat + b_P_mat + tf.matmul(P, tf.transpose(Q))
    elif model_spec_name in ['cadrres-wo-sample-bias', 'cadrres-wo-sample-bias-weight']:
        S = b_Q_mat + tf.matmul(P, tf.transpose(Q))
    else:
        S = None

    return S


@tf.function
def train_step(X_train, Y_train, S_train_obs_resp, parameters, train_known_idx, lda, optimizer):
    with tf.GradientTape() as tape:
        S_train_pred = inward_propagation(X_train, Y_train, parameters, *X_train.shape, 'cadrres-wo-sample-bias')
        S_train_pred_resp = tf.gather(tf.reshape(S_train_pred, [-1]), train_known_idx)
        diff_op_train = tf.subtract(S_train_pred_resp, S_train_obs_resp)
        
        base_cost = tf.reduce_sum(tf.square(diff_op_train))
        regularizer = lda * (tf.reduce_sum(tf.square(parameters['W_P'])) + tf.reduce_sum(tf.square(parameters['W_Q'])))
        cost_train = (base_cost + regularizer) / (len(train_known_idx) * 2.0)

    gradients = tape.gradient(cost_train, parameters.values())
    optimizer.apply_gradients(zip(gradients, parameters.values()))
    return cost_train


def get_latent_vectors(X, Y, parameters):

    """
    Get latent vectors of cell line (P) and drug (Q) on the pharmacogenomic space
    """

    W_P = parameters['W_P']
    W_Q = parameters['W_Q']
    P = tf.matmul(X, W_P)
    Q = tf.matmul(Y, W_Q)
    return P, Q

def train_model(train_resp_df, train_feature_df, test_resp_df, test_feature_df,
                n_dim, lda, max_iter, l_rate, model_spec_name='cadrres-wo-sample-bias',
                flip_score=True, seed=1, save_interval=1000, output_dir='output'):
    """
    Train a model. This is for the original cadrres and cadrres-wo-sample-bias models.
    """
    print('Initializing the model...')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if model_spec_name not in ['cadrres', 'cadrres-wo-sample-bias']:
        return None

    n_drugs = train_resp_df.shape[1]
    drug_list = train_resp_df.columns

    n_samples = train_resp_df.shape[0]
    sample_list_train = train_resp_df.index
    sample_list_test = test_resp_df.index

    n_x_features = train_feature_df.shape[1]
    n_y_features = n_drugs

    X_train_dat = np.array(train_feature_df, dtype=np.float32)
    Y_train_dat = np.identity(n_drugs, dtype=np.float32)

    if flip_score:
        S_train_obs = np.array(train_resp_df, dtype=np.float32) * -1
    else:
        S_train_obs = np.array(train_resp_df, dtype=np.float32)

    print('Building the TensorFlow model...')
    tf.random.set_seed(seed)

    X_train = tf.convert_to_tensor(X_train_dat, dtype=tf.float32)
    Y_train = tf.convert_to_tensor(Y_train_dat, dtype=tf.float32)
    parameters = initialize_parameters(n_samples, n_drugs, n_x_features, n_y_features, n_dim, seed)

    train_known_idx = np.where(~np.isnan(S_train_obs.reshape(-1)))[0]
    S_train_obs_resp = tf.convert_to_tensor(S_train_obs.reshape(-1)[train_known_idx], dtype=tf.float32)

    optimizer = tf.keras.optimizers.SGD(learning_rate=l_rate)

    print('Training the model...')
    cost_train_vals = []

    start = time.time()
    for i in range(max_iter):
        cost_train = train_step(X_train, Y_train, S_train_obs_resp, parameters, train_known_idx, lda, optimizer)
        if i % save_interval == 0:
            cost_train_vals.append(float(cost_train))
            time_used = time.time() - start
            print(f"Step {i}, Train Loss: {cost_train:.4f}, Time Elapsed: {time_used / 60:.2f} min")

    print('Finalizing training...')
    parameters_trained = {k: v.numpy() for k, v in parameters.items()}
    parameters_trained['mse_train_vals'] = cost_train_vals

    # Add additional metadata to the saved model
    parameters_trained['drug_list'] = drug_list
    parameters_trained['kernel_sample_list'] = list(train_feature_df.columns)
    parameters_trained['sample_list_train'] = list(sample_list_train)

    print('Saving the trained model...')
    output_dict = {
        'parameters': parameters_trained,
        'mse_train_vals': cost_train_vals,
    }

    return parameters_trained, output_dict


import pandas as pd
import numpy as np
import optuna
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # Import SimpleImputer for imputation

# Function to calculate NDCG
def ndcg_at_k(y_true, y_pred, k=10):
    """
    Compute the Normalized Discounted Cumulative Gain (NDCG) at rank k.
    
    Parameters:
    - y_true: True IC50 values (array-like)
    - y_pred: Predicted IC50 values (array-like)
    - k: Rank position for NDCG (default 10)
    
    Returns:
    - NDCG score (float)
    """
    # Sort the true values and predicted values in descending order
    order_true = np.argsort(y_true)[::-1]
    order_pred = np.argsort(y_pred)[::-1]
    
    # Compute DCG
    dcg = 0
    for i in range(k):
        if i < len(y_true):
            # Discounted Cumulative Gain (DCG)
            dcg += (2**y_true[order_pred[i]] - 1) / np.log2(i + 2)
    
    # Compute Ideal DCG (IDCG)
    idcg = 0
    for i in range(k):
        if i < len(order_true):
            idcg += (2**y_true[order_true[i]] - 1) / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0

###################################objective###################################
#
# Objective function to be optimized by the Optuna study. Suggests 
# hyperparameters for the SVD feature formation and TensorFlow neural network,
# and uses CV to maximize accuracy of the model.
#
# Inputs: 
#           df_path: path to data frame that contains all cell lines, drugs, and IC50 values
#           frac_test: fraction of data to be used as the test set during evaluation
#           trial: the trial of the optuna study
# 
# Outputs: 
#           The cross-validation R^2 score of the model with given hyperparameters.
#
###############################################################################

def objective(df_path, frac_test, trial):
    # Load the data
    df = pd.read_csv(df_path)
    
    # Parameters for SVD and model
    svd_n_components = trial.suggest_int('svd_n_components', 5, 20)
    hidden_units = trial.suggest_int('hidden_units', 32, 256, step=32)  # Number of neurons in the hidden layer
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)  # Learning rate for the optimizer
    
    cell_line_obs_df = pd.read_csv(df_path, index_col=0)
    
    # Impute missing values using the mean of each column (drug)
    imputer = SimpleImputer(strategy='mean')  # You can also try median or other strategies
    cell_line_obs_df_imputed = imputer.fit_transform(cell_line_obs_df)


    # Apply Truncated SVD for matrix factorization
    svd = TruncatedSVD(n_components=svd_n_components)
    latent_matrix = svd.fit_transform(cell_line_obs_df_imputed)
    
    # Get the number of cell lines and drugs
    num_cell_lines = latent_matrix.shape[0]  # Rows of the latent matrix (cell lines)
    num_drugs = cell_line_obs_df.shape[1]    # Columns of the original matrix (drugs)
    
    # Create empty lists to hold feature vectors and targets (IC50 values)
    features = []
    targets = []
    
    # Loop over each cell line and drug combination
    for i in range(num_cell_lines):
        for j in range(num_drugs):
            # Get the latent vector for the cell line i
            cell_line_latent = latent_matrix[i]    
            # Get the latent vector for the drug j
            drug_latent = svd.components_[:, j] 
            feature_vector = np.concatenate([cell_line_latent, drug_latent])

            ic50_value = cell_line_obs_df_imputed[i, j]
            features.append(feature_vector)
            targets.append(ic50_value)
    
    # Convert the feature list and target list to numpy arrays
    features = np.array(features)
    print(np.isfinite(features).all())  # Ensure no Inf values
    targets = np.array(targets)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=frac_test, random_state=0)
    
    # Build the neural network model in TensorFlow
    model = keras.Sequential([ 
        layers.InputLayer(input_shape=(X_train.shape[1],)), 
        layers.Dense(hidden_units, activation='relu'), 
        layers.Dense(1)  # Single output unit for predicting IC50 value 
    ])
    
    # Compile the model with Adam optimizer and mean squared error loss
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse',  # Mean squared error for regression
                  metrics=['mae'])  # Mean absolute error for evaluation
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    
    # Evaluate the model on the validation set (R^2 score)
    y_pred = model.predict(X_test).flatten()  # Flatten to match dimensions
    
    # Calculate R^2
    r2_score = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    # Calculate NDCG at rank 10
    ndcg_score = ndcg_at_k(y_test, y_pred, k=10)
    
    print(f"R^2 Score: {r2_score:.4f}, NDCG at 10: {ndcg_score:.4f}")
    
    return ndcg_score  # You can modify this to return both metrics or just R^2

def run_optuna_study():
    study = optuna.create_study(direction='maximize')  # We want to maximize R^2 score
    study.optimize(lambda trial: objective(df_path='../data/GDSC/gdsc_all_abs_ic50_bayesian_sigmoid_only9dosages.csv', 
                                           frac_test=0.2, trial=trial), n_trials=50)
    print("Best trial:")
    print(study.best_trial)

# Start the optimization process
run_optuna_study()

import os
import json
import numpy as np 
import tensorflow as tf
import scipy
from scipy.spatial import distance
import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from utils.json_utils import NumpyEncoder


FIGURE_X = 6.0
FIGURE_Y = 4.0

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/data/'

SHOW_PLOTS = False

args = {
    'nb_of_samples': 100,
    'number_of_functions': 10,
    'dirichlet_alphas': [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    'num_dirichlet_samples': 5,
    'resample_size': 10_000,
}

def get_model(n_input_features):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8, activation='tanh', input_shape=(n_input_features,)))
    model.add(tf.keras.layers.Dense(32, activation='tanh'))
    model.add(tf.keras.layers.Dense(32, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    # compile the model
    model.compile(optimizer='adam', loss='mse')
    return model

def exponentiated_quadratic(xa, xb):
    sq_norm = -0.25 * distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)


if __name__ == '__main__':

    # Setup experiment data folder.
    exp_name = 'gen_errors_'+ str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
    print('\nExperiment ID:', exp_name)
    print('gen_errors.py arguments:')
    print(args)
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    f = open(exp_path + "/args.json", "w")
    json.dump(args, f)
    f.close()

    # Setup dictionary to hold experiment data.
    data = {}

    """
        Randomly generate functions to fit (samples from G.P. prior).
    """
    X = np.linspace(-10, 10, args['nb_of_samples'])
    X_ = np.expand_dims(X, 1)
    cov_matrix = exponentiated_quadratic(X_, X_)
    Ys = np.random.multivariate_normal(
        mean=np.zeros(args['nb_of_samples']), cov=cov_matrix,
        size=args['number_of_functions']) # + 0.1*np.random.normal(size=args['nb_of_samples'])
    
    if SHOW_PLOTS:
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)
        for s in range(args['number_of_functions']):
            plt.scatter(X, Ys[s], label=f'Sample {s}')
        plt.title('Original functions: GP prior samples')
        plt.legend()
        plt.show()

    data['X'] = X
    data['Ys'] = Ys

    """
        Resample, train and predict.
    """
    # Prepare dicts to store experiment data.
    error_ranks_at_max = {}
    errors_at_max = {}
    average_errors = {}
    max_estimation_mistakes = {}
    max_estimation_mistakes_2 = {}
    for alpha in args['dirichlet_alphas']:
        error_ranks_at_max[str(alpha)] = []
        errors_at_max[str(alpha)] = []
        average_errors[str(alpha)] = []
        max_estimation_mistakes[str(alpha)] = []
        max_estimation_mistakes_2[str(alpha)] = []

    for (i, Y) in enumerate(Ys):
        print('G.P. function sample number', i)

        for alpha in args['dirichlet_alphas']:
            print('Alpha=', alpha)

            for _ in range(args['num_dirichlet_samples']):
            
                # Resample data.
                X_resampled = []
                Y_resampled = []
                dirichlet_sample = np.random.dirichlet([alpha]*args['nb_of_samples'])
                for _ in range(args['resample_size']):
                    idx = np.random.choice(range(len(X)), p=dirichlet_sample)
                    X_resampled.append(X[idx])
                    Y_resampled.append(Y[idx])
                X_resampled = np.array(X_resampled)
                Y_resampled = np.array(Y_resampled)

                # Plot resampled data.
                if SHOW_PLOTS:
                    fig, ax1 = plt.subplots()
                    fig.set_size_inches(FIGURE_X, FIGURE_Y)
                    ax1.scatter(X_resampled, Y_resampled, color='blue', label='Resampled data')
                    ax1.legend()
                    ax2 = ax1.twinx()
                    ax2.hist(X_resampled, 100, density=True, label='p(x)')
                    ax2.set_ylim([0,1.05])
                    plt.title('Resampled data')
                    plt.legend()
                    plt.show()

                # Train and predict.
                model = get_model(n_input_features=1)
                model.fit(X_resampled, Y_resampled, epochs=200, batch_size=128, verbose=False)
                Yhat = model.predict(X)
                Yhat = Yhat.reshape(Yhat.shape[0])

                # Plot predictions.
                if SHOW_PLOTS:
                    fig = plt.figure()
                    fig.set_size_inches(FIGURE_X, FIGURE_Y)
                    plt.scatter(X_resampled, Y_resampled, label='True')
                    plt.plot(X, Yhat, label='Prediction')
                    plt.title(f'Fit to function {i}')
                    plt.show()

                max_idx = np.argmax(Yhat)
                Yhat_max_idx = Yhat[max_idx]
                Y_max_idx = Y[max_idx]
                X_max_idx = X[max_idx]
                error_max_idx = np.abs(Y_max_idx - Yhat_max_idx)
                # print('Yhat_max_idx:', Yhat_max_idx)
                # print('Y_max_idx:', Y_max_idx)
                # print('X_max_idx:', X_max_idx)
                # print('error_max_idx:', error_max_idx)

                errors = np.abs(Y - Yhat)
                errors_sorted = np.argsort(errors)
                error_rank = np.where(errors_sorted == max_idx)
                error_rank = int(error_rank[0])
                error_rank = len(errors) - error_rank
                # print('average_error:', error_rank)

                average_error = np.mean(errors)
                # print('error_rank:', error_rank)

                true_X_max_idx = X[np.argmax(Y)]
                is_max_estimation_mistake = not (X_max_idx == true_X_max_idx)
                # print('is_max_estimation_mistake:', is_max_estimation_mistake)

                is_max_estimation_mistake_2 = not (np.abs(true_X_max_idx - X_max_idx) <= 3)
                # print('is_max_estimation_mistake_2:', is_max_estimation_mistake_2)

                errors_at_max[str(alpha)].append(error_max_idx)
                error_ranks_at_max[str(alpha)].append(error_rank)
                average_errors[str(alpha)].append(average_error)
                max_estimation_mistakes[str(alpha)].append(is_max_estimation_mistake)
                max_estimation_mistakes_2[str(alpha)].append(is_max_estimation_mistake_2)

                # Plot residuals.
                # fig = plt.figure()
                # fig.set_size_inches(FIGURE_X, FIGURE_Y)
                # plt.hist(errors, 50, density=True)
                # plt.title('Residuals')
                # plt.show()

    # Print and store experiment data.
    print('errors_at_max:', errors_at_max)
    print('error_ranks_at_max:', error_ranks_at_max)
    print('average_errors:', average_errors)
    print('max_estimation_mistakes:', max_estimation_mistakes)
    print('max_estimation_mistakes_2:', max_estimation_mistakes_2)

    data['errors_at_max'] = errors_at_max
    data['error_ranks_at_max'] = error_ranks_at_max
    data['average_errors'] = average_errors
    data['max_estimation_mistakes'] = max_estimation_mistakes
    data['max_estimation_mistakes_2'] = max_estimation_mistakes_2

    # Store results data.
    f = open(exp_path + "/data.json", "w")
    dumped = json.dumps(data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

import numpy as np 
import tensorflow as tf
import scipy
from scipy.spatial import distance

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

FIGURE_X = 6.0
FIGURE_Y = 4.0


def generate_data(N):
    """
        Creates Mackey-Glass time-series data.
        :param N:       number of points (starting from zero)
        :type  N:       integer
    """
    x = np.zeros((N))
    x[0] = 1.5

    for i in range(N-1):
        if i < 25:
            x[i+1] = x[i] - 0.1*x[i]
        else:
            x[i+1] = x[i] + (0.2*x[i-25])/(1 + np.power(x[i-25], 10)) - 0.1*x[i]
    
    return x

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
    """Exponentiated quadratic  with sigma=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.25 * distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)


if __name__ == '__main__':

    # Entropy plot.
    """ alphas = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0]
    num_classes = 100
    entropies = []
    for alpha in alphas:
        expected_entropy = scipy.special.digamma(num_classes*alpha + 1) - scipy.special.digamma(alpha + 1)
        entropies.append(expected_entropy)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    plt.plot(alphas, entropies)
    plt.xlabel('Alpha (Dirichelet parameter)')
    plt.ylabel('Expected entropy of categorical dist.')
    plt.show() """

    # Mackey-Glass time-series data.
    """ num_points = 100
    Y = generate_data(num_points)
    X = np.linspace(0,num_points-1,num_points)
    # Plot data.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    plt.plot(X, Y)
    plt.title('Original function: Mackey-Glass time-series')
    plt.show()
    # plt.savefig('{0}/time_series.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    # plt.close() """

    # X*cos(X) data.
    """ X = np.linspace(-10, 10, num=1000)
    Y = 0.1*X*np.cos(X) #+ 0.1*np.random.normal(size=1000)
    # Plot data.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    plt.plot(X, Y)
    plt.title('Original function: aX*cos(X)')
    plt.show()
    # plt.savefig('{0}/time_series.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    # plt.close() """

    # Gaussian process prior samples data.
    nb_of_samples = 100
    number_of_functions = 5
    X = np.linspace(-10, 10, nb_of_samples)
    X_ = np.expand_dims(X, 1)
    cov_matrix = exponentiated_quadratic(X_, X_)
    Ys = np.random.multivariate_normal(
        mean=np.zeros(nb_of_samples), cov=cov_matrix,
        size=number_of_functions) # + 0.1*np.random.normal(size=nb_of_samples)
    
    # Plot data.
    """ fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    for s in range(number_of_functions):
        plt.scatter(X, Ys[s], label=f'Sample {s}')
    plt.title('Original functions: GP prior samples')
    plt.legend()
    plt.show() """

    dirichlet_alphas = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    num_dirichlet_samples = 5
    resample_size = 10_000

    results_ranks = {}
    results_errors = {}
    for alpha in dirichlet_alphas:
        results_ranks[str(alpha)] = []
        results_errors[str(alpha)] = []

    for (i, Y) in enumerate(Ys):
        print('G.P. function sample number', i)

        for alpha in dirichlet_alphas:
            print('Alpha=', alpha)

            for _ in range(num_dirichlet_samples):
            
                # Resample data.
                X_resampled = []
                Y_resampled = []
                dirichlet_sample = np.random.dirichlet([alpha]*nb_of_samples)
                for _ in range(resample_size):
                    idx = np.random.choice(range(len(X)), p=dirichlet_sample)
                    X_resampled.append(X[idx])
                    Y_resampled.append(Y[idx])
                X_resampled = np.array(X_resampled)
                Y_resampled = np.array(Y_resampled)

                # Plot resampled data.
                """ fig, ax1 = plt.subplots()
                fig.set_size_inches(FIGURE_X, FIGURE_Y)
                ax1.scatter(X_resampled, Y_resampled, color='blue', label='Resampled data')
                ax1.legend()
                ax2 = ax1.twinx()
                ax2.hist(X_resampled, 100, density=True, label='p(x)')
                ax2.set_ylim([0,1.05])
                plt.title('Resampled data')
                plt.legend()
                plt.show() """

                # Train and predict.
                model = get_model(n_input_features=1)
                model.fit(X_resampled, Y_resampled, epochs=200, batch_size=128, verbose=False)
                Yhat = model.predict(X)
                Yhat = Yhat.reshape(Yhat.shape[0])

                # Plot.
                """ fig = plt.figure()
                fig.set_size_inches(FIGURE_X, FIGURE_Y)
                plt.scatter(X_resampled, Y_resampled, label='True')
                plt.plot(X, Yhat, label='Prediction')
                plt.title(f'Fit to function {i}')
                plt.show() """

                max_idx = np.argmax(Yhat)
                Yhat_max_idx = Yhat[max_idx]
                Y_max_idx = Y[max_idx]
                X_max_idx = X[max_idx]
                error_max_idx = np.abs(Y_max_idx - Yhat_max_idx)
                # print('Yhat_max_idx:', Yhat_max_idx)
                # print('Y_max_idx:', Y_max_idx)
                # print('X_max_idx:', X_max_idx)
                print('error_max_idx:', error_max_idx)

                errors = np.abs(Y - Yhat)
                errors_sorted = np.argsort(errors)
                error_rank = np.where(errors_sorted == max_idx)
                error_rank = int(error_rank[0])
                error_rank = len(errors) - error_rank
                print('error_rank:', error_rank)

                results_errors[str(alpha)].append(error_max_idx)
                results_ranks[str(alpha)].append(error_rank)

                """ fig = plt.figure()
                fig.set_size_inches(FIGURE_X, FIGURE_Y)
                plt.hist(errors, 50, density=True)
                plt.title('Residuals errors')
                plt.show() """

    print(results_errors)
    print(results_ranks)

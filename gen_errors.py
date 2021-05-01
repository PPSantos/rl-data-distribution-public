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

    alphas = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0]
    """ num_classes = 100

    entropies = []
    for alpha in alphas:
        expected_entropy = scipy.special.digamma(num_classes*alpha + 1) - scipy.special.digamma(alpha + 1)
        entropies.append(expected_entropy)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    plt.plot(alphas, entropies)
    plt.xlabel('Alpha (Dirichelet parameter)')
    plt.ylabel('Expected entropy')
    #plt.title('Expected entropy of categorical dist. for different dirichlet alpha param.')
    plt.show()
    # plt.savefig('{0}/time_series.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    # plt.close() """

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

    # Gaussian process prior samples.
    nb_of_samples = 100
    number_of_functions = 5
    X = np.linspace(-10, 10, nb_of_samples)
    X_ = np.expand_dims(X, 1)
    cov_matrix = exponentiated_quadratic(X_, X_)
    Ys = np.random.multivariate_normal(
        mean=np.zeros(nb_of_samples), cov=cov_matrix,
        size=number_of_functions) # + 0.1*np.random.normal(size=nb_of_samples)
    
    # Plot data.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    for s in range(number_of_functions):
        plt.scatter(X, Ys[s], label=f'Sample {s}')
    plt.title('Original functions: GP prior samples')
    plt.legend()
    plt.show()
    # plt.savefig('{0}/time_series.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    # plt.close()

    for (i, Y) in enumerate(Ys):
        print('Y=', i)

        # Resample data.
        X_data = []
        Y_data = []
        for _ in range(10000):
            idx = np.random.choice(range(len(X)))
            X_data.append(X[idx])
            Y_data.append(Y[idx])
        X_data = np.array(X_data)
        Y_data = np.array(Y_data)

        # Plot resampled data.
        fig, ax1 = plt.subplots()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        ax1.scatter(X_data, Y_data, color='blue', label='Resampled data')
        ax2 = ax1.twinx()
        ax2.hist(X_data, 100, density=True, label='p(x)')
        ax2.set_ylim([0,1.05])

        plt.title('Resampled data')
        plt.legend()
        plt.show()
        # plt.savefig('{0}/time_series.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
        # plt.close()

        # Train and predict.
        model = get_model(n_input_features=1)
        model.fit(X_data, Y_data, epochs=500, batch_size=128, verbose=True)
        yhat = model.predict(X)

        # Plot.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)
        plt.scatter(X_data, Y_data, label='True')
        plt.plot(X, yhat, label='Prediction')
        plt.title(f'Fit to function {i}')
        plt.show()
        # plt.savefig('{0}/time_series.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
        # plt.close()

        """ Yhat_max_pred = np.max(yhat)
        Y_max_pred = Y[np.argmax(yhat)]
        X_max_pred = X[np.argmax(yhat)]
        error = np.abs(Y_max_pred - Yhat_max_pred)
        print('Yhat_max_pred:', Yhat_max_pred)
        print('Y_max_pred:', Y_max_pred)
        print('X_max_pred:', X_max_pred)
        print('Error (abs(Y_max_pred-Yhat_max_pred)):', error) """



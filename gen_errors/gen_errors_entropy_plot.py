import scipy
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

FIGURE_X = 6.0
FIGURE_Y = 4.0

if __name__ == "__main__":

    # Entropy plot.
    alphas = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0] #, 20.0, 30.0]
    num_classes = 5
    entropies = []
    for alpha in alphas:
        expected_entropy = scipy.special.digamma(num_classes*alpha + 1) - scipy.special.digamma(alpha + 1)
        entropies.append(expected_entropy)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    plt.plot(alphas, entropies)
    plt.xlabel('Alpha (Dirichlet parameter)')
    plt.ylabel('Expected entropy of categorical dist.')
    plt.show()

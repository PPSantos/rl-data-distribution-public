import numpy as np 


def f(x):
    return x

def norm(vec, dist):
    return np.sqrt(np.dot(dist,np.power(vec,2)))


num_trials = 10_000
res = []
for _ in range(num_trials):

    dist_mu = np.random.dirichlet([10.0]*10)
    dist_beta = np.random.dirichlet([10.0]*10)

    x = np.random.rand(10)

    C_constant = np.sqrt(np.max(dist_beta/dist_mu))

    flag = np.dot(np.abs(dist_mu-dist_beta), np.power(f(x),2)) + norm(f(x), dist_mu) <= C_constant*norm(f(x), dist_mu)
    res.append(flag)

print(res)
print(num_trials-np.sum(res))

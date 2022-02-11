import numpy as np

from scripts.run import RUN_ARGS
from scripts.run import main as run

###########################################################################
ENVS = ['pendulum']

###########################################################################

if __name__ == "__main__":

    exp_ids = []

    """
        DQN.
    """
    """
    # Dirichlet dataset: vary alpha dirichlet parameter.
    run_args = RUN_ARGS
    run_args['dataset_args']['dataset_type'] = 'dirichlet'
    run_args['train_args']['algo'] = 'offline_dqn'

    for env in ENVS:

        # Setup train args.
        run_args['env_name'] = env
        print('env=', env)

        run_args['dataset_args']['force_full_coverage'] = False
        for a in [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]:

            print('Alpha=', a)
            run_args['dataset_args']['dirichlet_dataset_args']['dirichlet_alpha_coef'] = a

            # Run.
            exp_id = run(run_args)
            exp_ids.append(exp_id)

            print('Exp. ids:', exp_ids)


        run_args['dataset_args']['force_full_coverage'] = True
        for a in [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]:

            print('Alpha=', a)
            run_args['dataset_args']['dirichlet_dataset_args']['dirichlet_alpha_coef'] = a

            # Run.
            exp_id = run(run_args)
            exp_ids.append(exp_id)

            print('Exp. ids:', exp_ids) """

    # Epsilon-greedy dataset: vary epsilon parameter.
    run_args = RUN_ARGS
    run_args['dataset_args']['dataset_type'] = 'eps-greedy'
    run_args['train_args']['algo'] = 'offline_dqn'

    for env in ENVS:

        # Setup train args.
        run_args['env_name'] = env
        print('env=', env)

        run_args['dataset_args']['force_full_coverage'] = False
        for e in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:

            print('Epsilon=', e)
            run_args['dataset_args']['eps_greedy_dataset_args']['epsilon'] = e

            # Run.
            exp_id = run(run_args)
            exp_ids.append(exp_id)

            print('Exp. ids:', exp_ids)

        run_args['dataset_args']['force_full_coverage'] = True
        for e in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:

            print('Epsilon=', e, 'algo=', run_args['train_args']['algo'])
            run_args['dataset_args']['eps_greedy_dataset_args']['epsilon'] = e

            # Run.
            exp_id = run(run_args)
            exp_ids.append(exp_id)

            print('Exp. ids:', exp_ids)

    # Boltzmann dataset: vary temperature parameter.
    run_args = RUN_ARGS
    run_args['dataset_args']['dataset_type'] = 'boltzmann'
    run_args['train_args']['algo'] = 'offline_dqn'

    for env in ENVS:

        # Setup train args.
        run_args['env_name'] = env
        print('env=', env)

        temps = np.linspace(-10.0, 10-0, 21)
        run_args['dataset_args']['force_full_coverage'] = False
        for t in temps:

            print('Temperature=', t)
            run_args['dataset_args']['boltzmann_dataset_args']['temperature'] = t

            # Run.
            exp_id = run(run_args)
            exp_ids.append(exp_id)

            print('Exp. ids:', exp_ids)

        run_args['dataset_args']['force_full_coverage'] = True
        for t in temps:

            print('Temperature=', t, 'algo=', run_args['train_args']['algo'])
            run_args['dataset_args']['boltzmann_dataset_args']['temperature'] = t

            # Run.
            exp_id = run(run_args)
            exp_ids.append(exp_id)

            print('Exp. ids:', exp_ids)

    exit()

    """
        CQL.
    """
    # Dirichlet dataset: vary alpha dirichlet parameter.
    run_args = RUN_ARGS
    run_args['dataset_args']['dataset_type'] = 'dirichlet'
    run_args['train_args']['algo'] = 'offline_cql'

    for env in ENVS:

        # Setup train args.
        run_args['env_name'] = env
        print('env=', env)

        run_args['dataset_args']['force_full_coverage'] = False
        for a in [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]:

            print('Alpha=', a, 'algo=', run_args['train_args']['algo'])
            run_args['dataset_args']['dirichlet_dataset_args']['dirichlet_alpha_coef'] = a

            # Run.
            exp_id = run(run_args)
            exp_ids.append(exp_id)

            print('Exp. ids:', exp_ids)

        run_args['dataset_args']['force_full_coverage'] = True
        for a in [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]:

            print('Alpha=', a, 'algo=', run_args['train_args']['algo'])
            run_args['dataset_args']['dirichlet_dataset_args']['dirichlet_alpha_coef'] = a

            # Run.
            exp_id = run(run_args)
            exp_ids.append(exp_id)

            print('Exp. ids:', exp_ids)

    # Epsilon-greedy dataset: vary epsilon parameter.
    run_args = RUN_ARGS
    run_args['dataset_args']['dataset_type'] = 'eps-greedy'
    run_args['train_args']['algo'] = 'offline_cql'

    for env in ENVS:

        # Setup train args.
        run_args['env_name'] = env
        print('env=', env)

        run_args['dataset_args']['force_full_coverage'] = False
        for e in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:

            print('Epsilon=', e, 'algo=', run_args['train_args']['algo'])
            run_args['dataset_args']['eps_greedy_dataset_args']['epsilon'] = e

            # Run.
            exp_id = run(run_args)
            exp_ids.append(exp_id)

            print('Exp. ids:', exp_ids)


        run_args['dataset_args']['force_full_coverage'] = True
        for e in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:

            print('Epsilon=', e, 'algo=', run_args['train_args']['algo'])
            run_args['dataset_args']['eps_greedy_dataset_args']['epsilon'] = e

            # Run.
            exp_id = run(run_args)
            exp_ids.append(exp_id)

            print('Exp. ids:', exp_ids)

    # Boltzmann dataset: vary temperature parameter.
    run_args = RUN_ARGS
    run_args['dataset_args']['dataset_type'] = 'boltzmann'
    run_args['train_args']['algo'] = 'offline_cql'

    for env in ENVS:

        # Setup train args.
        run_args['env_name'] = env
        print('env=', env)

        temps = np.linspace(-10.0, 10-0, 21)

        run_args['dataset_args']['force_full_coverage'] = False
        for t in temps:

            print('Temperature=', t, 'algo=', run_args['train_args']['algo'])
            run_args['dataset_args']['boltzmann_dataset_args']['temperature'] = t

            # Run.
            exp_id = run(run_args)
            exp_ids.append(exp_id)

            print('Exp. ids:', exp_ids)

        run_args['dataset_args']['force_full_coverage'] = True
        for t in temps:

            print('Temperature=', t, 'algo=', run_args['train_args']['algo'])
            run_args['dataset_args']['boltzmann_dataset_args']['temperature'] = t

            # Run.
            exp_id = run(run_args)
            exp_ids.append(exp_id)

            print('Exp. ids:', exp_ids)

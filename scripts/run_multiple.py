from scripts.run import RUN_ARGS
from scripts.run import main as run

###########################################################################
ENVS = ['gridEnv1',]

###########################################################################

if __name__ == "__main__":

    # Vary alpha dirichlet parameter.
    """ run_args = RUN_ARGS

    exp_ids = []
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

    # Vary epsilon parameter.
    run_args = RUN_ARGS

    exp_ids = []
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

            print('Epsilon=', e)
            run_args['dataset_args']['eps_greedy_dataset_args']['epsilon'] = e

            # Run.
            exp_id = run(run_args)
            exp_ids.append(exp_id)

            print('Exp. ids:', exp_ids)


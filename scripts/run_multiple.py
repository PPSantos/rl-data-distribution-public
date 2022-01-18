from scripts.run import RUN_ARGS
from scripts.run import main as run

###########################################################################
ENVS = ['gridEnv1',]

###########################################################################

if __name__ == "__main__":

    # Load default args.
    run_args = RUN_ARGS

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

            print('Exp. ids:', exp_ids)


from train import train
from train import DEFAULT_TRAIN_ARGS

if __name__ == "__main__":

    args = DEFAULT_TRAIN_ARGS
    
    args['dqn_args']['max_replay_size'] = 500_000
    train(args)

    args['dqn_args']['max_replay_size'] = 400_000
    train(args)

    args['dqn_args']['max_replay_size'] = 300_000
    train(args)

    args['dqn_args']['max_replay_size'] = 200_000
    train(args)

    args['dqn_args']['max_replay_size'] = 100_000
    train(args)

    args['dqn_args']['max_replay_size'] = 50_000
    train(args)

    args['dqn_args']['max_replay_size'] = 25_000
    train(args)


    args = DEFAULT_TRAIN_ARGS
    args['env_args']['smooth_obs'] = True
    args['env_args']['one_hot_obs'] = False

    args['dqn_args']['max_replay_size'] = 500_000
    train(args)

    args['dqn_args']['max_replay_size'] = 400_000
    train(args)

    args['dqn_args']['max_replay_size'] = 300_000
    train(args)

    args['dqn_args']['max_replay_size'] = 200_000
    train(args)

    args['dqn_args']['max_replay_size'] = 100_000
    train(args)

    args['dqn_args']['max_replay_size'] = 50_000
    train(args)

    args['dqn_args']['max_replay_size'] = 25_000
    train(args)


    args = DEFAULT_TRAIN_ARGS
    args['env_args']['smooth_obs'] = False
    args['env_args']['one_hot_obs'] = False

    args['dqn_args']['max_replay_size'] = 500_000
    train(args)

    args['dqn_args']['max_replay_size'] = 400_000
    train(args)

    args['dqn_args']['max_replay_size'] = 300_000
    train(args)

    args['dqn_args']['max_replay_size'] = 200_000
    train(args)

    args['dqn_args']['max_replay_size'] = 100_000
    train(args)

    args['dqn_args']['max_replay_size'] = 50_000
    train(args)

    args['dqn_args']['max_replay_size'] = 25_000
    train(args)

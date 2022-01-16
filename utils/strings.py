from datetime import datetime

def create_exp_name(args: dict) -> str:
    return args['env_name'] + \
        '_' + args['algo'] + '_' + \
        str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))

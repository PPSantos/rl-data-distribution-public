import pathlib
import os
import json

import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

FIGURE_X = 6.0
FIGURE_Y = 4.0

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/analysis/plots/'

exp_id = 'gen_errors_2021-05-03-00-58-04'

def get_args_json_file(path):
    with open(path + "/args.json", 'r') as f:
        args = json.load(f)
    f.close()
    return args

if __name__ == "__main__":

    # Prepare plots output folder.
    output_folder = PLOTS_FOLDER_PATH + exp_id + '/'
    os.makedirs(output_folder, exist_ok=True)

    # Get args file (assumes all experiments share the same arguments).
    args = get_args_json_file(DATA_FOLDER_PATH + exp_id)
    print('Args:')
    print(args)

    # Store a copy of the args.json file inside plots folder.
    with open(output_folder + "args.json", 'w') as f:
        json.dump(args, f)
        f.close()

    # Open data.
    print(f"Opening experiment {exp_id}")
    exp_path = DATA_FOLDER_PATH + exp_id
    with open(exp_path + "/data.json", 'r') as f:
        data = json.load(f)
        data = json.loads(data)
    f.close()

    print(data.keys())

    # Plot function to fit (G.P. prior samples).
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    for s in range(args['number_of_functions']):
        plt.scatter(data['X'], data['Ys'][s], label=f'Sample {s}')
    plt.title('Functions to fit (G.P. prior samples)')
    plt.legend()
    # plt.show()
    # plt.savefig('{0}/functions_to_fit.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/functions_to_fit.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close()

    # average_errors
    average_errors = [np.mean(vals) for vals in data['average_errors'].values()]
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    plt.scatter(args['epsilons'], average_errors)
    plt.plot(args['epsilons'], average_errors)
    plt.xlabel('Epsilon')
    plt.ylabel('Average total abs. residual error')
    # plt.legend()
    # plt.show()
    plt.savefig('{0}/average_total_abs_residual_error.png'.format(output_folder), bbox_inches='tight', pad_inches=0)

    # errors_at_max
    average_errors_at_max = [np.mean(vals) for vals in data['errors_at_max'].values()]
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    plt.scatter(args['epsilons'], average_errors_at_max)
    plt.plot(args['epsilons'], average_errors_at_max)
    plt.xlabel('Epsilon')
    plt.ylabel('Average abs. residual error \n at x = argmax(Yhat)')
    # plt.legend()
    # plt.show()
    plt.savefig('{0}/average_abs_residual_error_at_max.png'.format(output_folder), bbox_inches='tight', pad_inches=0)

    # error_ranks_at_max
    average_error_rank_at_max = [np.mean(vals) for vals in data['error_ranks_at_max'].values()]
    #average_error_rank_at_max = np.array(average_error_rank_at_max) / args['nb_of_samples']
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    plt.scatter(args['epsilons'], average_error_rank_at_max)
    plt.plot(args['epsilons'], average_error_rank_at_max)
    plt.xlabel('Epsilon')
    plt.ylabel(f'Average rank (out of {args["nb_of_samples"]}) of abs. residual error \n at x = argmax(Yhat)')
    # plt.legend()
    # plt.show()
    plt.savefig('{0}/average_rank_abs_residual_error_at_max.png'.format(output_folder), bbox_inches='tight', pad_inches=0)

    # max_estimation_mistakes
    average_max_estimation_mistakes = [np.mean(vals) for vals in data['max_estimation_mistakes'].values()]
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    plt.scatter(args['epsilons'], average_max_estimation_mistakes)
    plt.plot(args['epsilons'], average_max_estimation_mistakes)
    plt.xlabel('Epsilon')
    plt.ylabel('Average argmax estimation mistakes')
    # plt.legend()
    # plt.show()
    plt.savefig('{0}/average_argmax_estimation_mistakes.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
   
    # max_estimation_mistakes_2
    average_max_estimation_mistakes_2 = [np.mean(vals) for vals in data['max_estimation_mistakes_2'].values()]
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    plt.scatter(args['epsilons'], average_max_estimation_mistakes_2)
    plt.plot(args['epsilons'], average_max_estimation_mistakes_2)
    plt.xlabel('Epsilon')
    plt.ylabel('Average argmax estimation mistakes \n (within eps. dist.)')
    # plt.legend()
    # plt.show()
    plt.savefig('{0}/average_argmax_estimation_mistakes_2.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
   
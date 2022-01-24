from analysis.plots import main as plots
from scripts.run import VAL_ITER_DATA

ENV_NAME = 'gridEnv1'
EXP_IDS = [
'gridEnv2_offline_dqn_2022-01-22-14-07-39', 'gridEnv2_offline_dqn_2022-01-22-14-14-53', 'gridEnv2_offline_dqn_2022-01-22-14-22-02', 'gridEnv2_offline_dqn_2022-01-22-14-29-02', 'gridEnv2_offline_dqn_2022-01-22-14-36-03', 'gridEnv2_offline_dqn_2022-01-22-14-42-59', 'gridEnv2_offline_dqn_2022-01-22-14-49-54', 'gridEnv2_offline_dqn_2022-01-22-14-56-55', 'gridEnv2_offline_dqn_2022-01-22-15-03-58', 'gridEnv2_offline_dqn_2022-01-22-15-10-54', 'gridEnv2_offline_dqn_2022-01-22-15-17-52', 'gridEnv2_offline_dqn_2022-01-22-15-24-45', 'gridEnv2_offline_dqn_2022-01-22-15-31-45', 'gridEnv2_offline_dqn_2022-01-22-15-38-44',
'gridEnv2_offline_dqn_2022-01-22-15-45-43', 'gridEnv2_offline_dqn_2022-01-22-15-52-33', 'gridEnv2_offline_dqn_2022-01-22-15-59-38', 'gridEnv2_offline_dqn_2022-01-22-16-06-28', 'gridEnv2_offline_dqn_2022-01-22-16-13-25', 'gridEnv2_offline_dqn_2022-01-22-16-20-14', 'gridEnv2_offline_dqn_2022-01-22-16-27-16', 'gridEnv2_offline_dqn_2022-01-22-16-34-09', 'gridEnv2_offline_dqn_2022-01-22-16-41-12', 'gridEnv2_offline_dqn_2022-01-22-16-48-13', 'gridEnv2_offline_dqn_2022-01-22-16-55-09', 'gridEnv2_offline_dqn_2022-01-22-17-02-03', 'gridEnv2_offline_dqn_2022-01-22-17-08-56', 'gridEnv2_offline_dqn_2022-01-22-17-15-51', 'gridEnv2_offline_dqn_2022-01-22-17-22-52', 'gridEnv2_offline_dqn_2022-01-22-17-29-50', 'gridEnv2_offline_dqn_2022-01-22-17-36-53', 'gridEnv2_offline_dqn_2022-01-22-17-43-47', 'gridEnv2_offline_dqn_2022-01-22-17-50-42', 'gridEnv2_offline_dqn_2022-01-22-17-57-44', 'gridEnv2_offline_dqn_2022-01-22-18-04-37', 'gridEnv2_offline_dqn_2022-01-22-18-13-09',
'gridEnv2_offline_dqn_2022-01-22-18-20-22', 'gridEnv2_offline_dqn_2022-01-22-18-27-28', 'gridEnv2_offline_dqn_2022-01-22-18-34-28', 'gridEnv2_offline_dqn_2022-01-22-18-41-24', 'gridEnv2_offline_dqn_2022-01-22-18-48-29', 'gridEnv2_offline_dqn_2022-01-22-18-55-22', 'gridEnv2_offline_dqn_2022-01-22-19-02-20', 'gridEnv2_offline_dqn_2022-01-22-19-09-19', 'gridEnv2_offline_dqn_2022-01-22-19-16-23', 'gridEnv2_offline_dqn_2022-01-22-19-23-29', 'gridEnv2_offline_dqn_2022-01-22-19-30-34', 'gridEnv2_offline_dqn_2022-01-22-19-37-33', 'gridEnv2_offline_dqn_2022-01-22-19-44-34', 'gridEnv2_offline_dqn_2022-01-22-19-51-30', 'gridEnv2_offline_dqn_2022-01-22-19-58-28', 'gridEnv2_offline_dqn_2022-01-22-20-05-26', 'gridEnv2_offline_dqn_2022-01-22-20-12-28', 'gridEnv2_offline_dqn_2022-01-22-20-19-34', 'gridEnv2_offline_dqn_2022-01-22-20-26-54', 'gridEnv2_offline_dqn_2022-01-22-20-34-06', 'gridEnv2_offline_dqn_2022-01-22-20-41-35', 'gridEnv2_offline_dqn_2022-01-22-20-48-51', 'gridEnv2_offline_dqn_2022-01-22-20-55-58', 'gridEnv2_offline_dqn_2022-01-22-21-02-58', 'gridEnv2_offline_dqn_2022-01-22-21-10-02', 'gridEnv2_offline_dqn_2022-01-22-21-17-11', 'gridEnv2_offline_dqn_2022-01-22-21-24-15', 'gridEnv2_offline_dqn_2022-01-22-21-31-21', 'gridEnv2_offline_dqn_2022-01-22-21-38-22', 'gridEnv2_offline_dqn_2022-01-22-21-45-34', 'gridEnv2_offline_dqn_2022-01-22-21-52-37', 'gridEnv2_offline_dqn_2022-01-22-21-59-43', 'gridEnv2_offline_dqn_2022-01-22-22-06-48', 'gridEnv2_offline_dqn_2022-01-22-22-13-47', 'gridEnv2_offline_dqn_2022-01-22-22-20-53', 'gridEnv2_offline_dqn_2022-01-22-22-27-57', 'gridEnv2_offline_dqn_2022-01-22-22-34-54', 'gridEnv2_offline_dqn_2022-01-22-22-41-52', 'gridEnv2_offline_dqn_2022-01-22-22-48-51', 'gridEnv2_offline_dqn_2022-01-22-22-55-57', 'gridEnv2_offline_dqn_2022-01-22-23-03-03', 'gridEnv2_offline_dqn_2022-01-22-23-09-59',
'gridEnv2_offline_cql_2022-01-23-01-26-31', 'gridEnv2_offline_cql_2022-01-23-01-36-47', 'gridEnv2_offline_cql_2022-01-23-01-46-30', 'gridEnv2_offline_cql_2022-01-23-01-56-08', 'gridEnv2_offline_cql_2022-01-23-02-05-56', 'gridEnv2_offline_cql_2022-01-23-02-15-41', 'gridEnv2_offline_cql_2022-01-23-02-25-30', 'gridEnv2_offline_cql_2022-01-23-02-35-19', 'gridEnv2_offline_cql_2022-01-23-02-45-08', 'gridEnv2_offline_cql_2022-01-23-02-54-59', 'gridEnv2_offline_cql_2022-01-23-03-04-38', 'gridEnv2_offline_cql_2022-01-23-03-14-25', 'gridEnv2_offline_cql_2022-01-23-03-24-07', 'gridEnv2_offline_cql_2022-01-23-03-33-55',
'gridEnv2_offline_cql_2022-01-23-03-43-36', 'gridEnv2_offline_cql_2022-01-23-03-53-20', 'gridEnv2_offline_cql_2022-01-23-04-03-08', 'gridEnv2_offline_cql_2022-01-23-04-12-46', 'gridEnv2_offline_cql_2022-01-23-04-22-26', 'gridEnv2_offline_cql_2022-01-23-04-32-07', 'gridEnv2_offline_cql_2022-01-23-04-41-53', 'gridEnv2_offline_cql_2022-01-23-04-51-45', 'gridEnv2_offline_cql_2022-01-23-05-01-36', 'gridEnv2_offline_cql_2022-01-23-05-11-30', 'gridEnv2_offline_cql_2022-01-23-05-21-11', 'gridEnv2_offline_cql_2022-01-23-05-30-55', 'gridEnv2_offline_cql_2022-01-23-05-40-45', 'gridEnv2_offline_cql_2022-01-23-05-50-24', 'gridEnv2_offline_cql_2022-01-23-06-00-03', 'gridEnv2_offline_cql_2022-01-23-06-09-55', 'gridEnv2_offline_cql_2022-01-23-06-19-39', 'gridEnv2_offline_cql_2022-01-23-06-29-16', 'gridEnv2_offline_cql_2022-01-23-06-38-59', 'gridEnv2_offline_cql_2022-01-23-06-48-34', 'gridEnv2_offline_cql_2022-01-23-06-58-15', 'gridEnv2_offline_cql_2022-01-23-07-07-58',
'gridEnv2_offline_cql_2022-01-23-07-17-40', 'gridEnv2_offline_cql_2022-01-23-07-27-29', 'gridEnv2_offline_cql_2022-01-23-07-37-07', 'gridEnv2_offline_cql_2022-01-23-07-46-51', 'gridEnv2_offline_cql_2022-01-23-07-56-34', 'gridEnv2_offline_cql_2022-01-23-08-06-12', 'gridEnv2_offline_cql_2022-01-23-08-16-11', 'gridEnv2_offline_cql_2022-01-23-08-25-52', 'gridEnv2_offline_cql_2022-01-23-08-35-39', 'gridEnv2_offline_cql_2022-01-23-08-45-15', 'gridEnv2_offline_cql_2022-01-23-08-55-06', 'gridEnv2_offline_cql_2022-01-23-09-04-48', 'gridEnv2_offline_cql_2022-01-23-09-14-33', 'gridEnv2_offline_cql_2022-01-23-09-24-13', 'gridEnv2_offline_cql_2022-01-23-09-34-03', 'gridEnv2_offline_cql_2022-01-23-09-43-42', 'gridEnv2_offline_cql_2022-01-23-09-53-25', 'gridEnv2_offline_cql_2022-01-23-10-03-03', 'gridEnv2_offline_cql_2022-01-23-10-12-51', 'gridEnv2_offline_cql_2022-01-23-10-22-30', 'gridEnv2_offline_cql_2022-01-23-10-32-11', 'gridEnv2_offline_cql_2022-01-23-10-41-50', 'gridEnv2_offline_cql_2022-01-23-10-51-34', 'gridEnv2_offline_cql_2022-01-23-11-01-26', 'gridEnv2_offline_cql_2022-01-23-11-11-12', 'gridEnv2_offline_cql_2022-01-23-11-21-05', 'gridEnv2_offline_cql_2022-01-23-11-30-48', 'gridEnv2_offline_cql_2022-01-23-11-40-32', 'gridEnv2_offline_cql_2022-01-23-11-50-15', 'gridEnv2_offline_cql_2022-01-23-12-00-11', 'gridEnv2_offline_cql_2022-01-23-12-10-17', 'gridEnv2_offline_cql_2022-01-23-12-20-14', 'gridEnv2_offline_cql_2022-01-23-12-30-21', 'gridEnv2_offline_cql_2022-01-23-12-40-27', 'gridEnv2_offline_cql_2022-01-23-12-50-45', 'gridEnv2_offline_cql_2022-01-23-13-00-52', 'gridEnv2_offline_cql_2022-01-23-13-10-55', 'gridEnv2_offline_cql_2022-01-23-13-20-58', 'gridEnv2_offline_cql_2022-01-23-13-30-57', 'gridEnv2_offline_cql_2022-01-23-13-41-00', 'gridEnv2_offline_cql_2022-01-23-13-51-06', 'gridEnv2_offline_cql_2022-01-23-14-01-10',
]

if __name__ == "__main__":

    for exp_id in EXP_IDS:
        plots(exp_id=exp_id+'.tar.gz', val_iter_exp=VAL_ITER_DATA[ENV_NAME])

import numpy as np
import argparse
import random
import torch
import os.path
import importlib
import os
import utils.fmodule
import ujson
import time
import collections
import logging

logger = None

def read_option():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument('--task', help='name of fedtask;', type=str, default='mnist_classification_cnum100_dist0_skew0_seed0')
    parser.add_argument('--algorithm', help='name of algorithm;', type=str, default='fedavg')
    parser.add_argument('--model', help='name of model;', type=str, default='cnn')
    parser.add_argument('--pretrain', help='the path of the pretrained model parameter created by torch.save;', type=str, default='')
    # hyper-parameters of training in server side
    parser.add_argument('--num_rounds', help='number of communication rounds', type=int, default=20)
    parser.add_argument('--proportion', help='proportion of clients sampled per round', type=float, default=0.2)
    parser.add_argument('--learning_rate_decay', help='learning rate decay for the training process;', type=float, default=0.998)
    parser.add_argument('--lr_scheduler', help='type of the global learning rate scheduler', type=int, default=-1)
    # hyper-parameters of local training
    parser.add_argument('--num_epochs', help='number of epochs when clients trainset on data;', type=int, default=5)
    parser.add_argument('--num_steps', help='the number of local steps, which dominate num_epochs when setting num_steps>0', type=int, default=-1)
    parser.add_argument('--learning_rate', help='learning rate for inner solver;', type=float, default=0.05)
    parser.add_argument('--batch_size', help='batch size when clients trainset on data;', type=float, default='64')
    parser.add_argument('--optimizer', help='select the optimizer for gd', type=str, default='SGD')
    parser.add_argument('--momentum', help='momentum of local update', type=float, default=0)
    parser.add_argument('--weight_decay', help='weight decay for the training process', type=float, default=0)
    # realistic machine config
    parser.add_argument('--seed', help='seed for random initialization;', type=int, default=0)
    parser.add_argument('--gpu', nargs='*', help='GPU IDs and empty input is equal to using CPU', type=int)
    parser.add_argument('--server_with_cpu', help='seed for random initialization;', action="store_true", default=False)
    parser.add_argument('--eval_interval', help='evaluate every __ rounds;', type=int, default=1)
    parser.add_argument('--num_threads', help="the number of threads in the clients computing session", type=int, default=1)
    parser.add_argument('--num_workers', help='the number of workers of DataLoader', type=int, default=0)
    parser.add_argument('--test_batch_size', help='the batch_size used in testing phase;', type=int, default=512)
    # the ratio of selected clients with larger local losses
    parser.add_argument('--C_ratio', help='the ratio of selected clients with larger local losses;', type=int, default=0.5)
    # algorithm-dependent hyper-parameters
    parser.add_argument('--algo_para', help='algorithm-dependent hyper-parameters', nargs='*', type=float)
    # logger setting
    parser.add_argument('--logger', help='the Logger in utils.logger.logger_name will be loaded', type=str, default='basic_logger')
    parser.add_argument('--log_level', help='the level of logger', type=str, default='INFO')
    parser.add_argument('--log_file', help='bool controls whether log to file and default value is False', action="store_true", default=False)
    parser.add_argument('--no_log_console', help='bool controls whether log to screen and default value is True', action="store_true", default=False)
    parser.add_argument('--no_overwrite', help='bool controls whether to overwrite the old result', action="store_true", default=False)

    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option

def setup_seed(seed):
    random.seed(1+seed)
    np.random.seed(21+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(12+seed)
    torch.cuda.manual_seed_all(123+seed)

def initialize(option):
    logger_order = {'{}Logger'.format(option['algorithm']):'%s.%s' % ('algorithm', option['algorithm']),option['logger']:'.'.join(['utils', 'logger', option['logger']]),'basic_logger':'.'.join(['utils', 'logger', 'basic_logger'])}
    global logger
    for log_name, log_path in logger_order.items():
        try:
            Logger = getattr(importlib.import_module(log_path), 'Logger')
            break
        except:
            continue
    logger = Logger(meta=option, name=log_name, level=option['log_level'])
    logger.info('Using Logger in `{}`'.format(log_path))
    logger.info("Initializing fedtask: {}".format(option['task']))
    # benchmark
    bmk_name = option['task'][:option['task'].find('cnum')-1]
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', option['model']])
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    TaskPipe = getattr(importlib.import_module(bmk_core_path), 'TaskPipe')
    train_datas, valid_datas, test_data, client_names = TaskPipe.load_task(os.path.join('fedtask', option['task']))
    try:
        utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
    except:
        utils.fmodule.Model = getattr(importlib.import_module('.'.join(['algorithm', option['algorithm']])), option['model'])
    try:
        utils.fmodule.SvrModel = getattr(importlib.import_module(bmk_model_path), 'SvrModel')
        logger.info('The server keeping the `SvrModel` in `{}`'.format(bmk_model_path))
    except:
        utils.fmodule.SvrModel = utils.fmodule.Model
    try:
        utils.fmodule.CltModel = getattr(importlib.import_module(bmk_model_path), 'CltModel')
        logger.info('Clients keeping the `CltModel` in `{}`'.format(bmk_model_path))
    except:
        utils.fmodule.CltModel = utils.fmodule.Model
    # init devices
    gpus = option['gpu']
    utils.fmodule.dev_list = [torch.device('cpu')] if gpus is None else [torch.device('cuda:{}'.format(gpu_id)) for gpu_id in gpus]
    utils.fmodule.dev_manager = utils.fmodule.get_device()
    utils.fmodule.TaskCalculator = getattr(importlib.import_module(bmk_core_path), 'TaskCalculator')
    logger.info('Initializing devices: '+','.join([str(dev) for dev in utils.fmodule.dev_list])+' will be used for this running.')
    if not option['server_with_cpu']:
        model = utils.fmodule.Model().to(utils.fmodule.dev_list[0])
    else:
        model = utils.fmodule.Model()
    try:
        if option['pretrain'] != '':
            model.load_state_dict(torch.load(option['pretrain'])['model'])
            logger.info('The pretrained model parameters in {} will be loaded'.format(option['pretrain']))
    except:
        logger.warn("Invalid Model Configuration.")
        exit(1)

    num_clients = len(client_names)
    client_path = '%s.%s' % ('algorithm', option['algorithm'])
    logger.info('Initializing clients: '+'{} clients of `{}` being created.'.format(num_clients, client_path+'.Client'))
    Client=getattr(importlib.import_module(client_path), 'Client')
    clients = [Client(option, name=client_names[cid], train_data=train_datas[cid], valid_data=valid_datas[cid]) for cid in range(num_clients)]

    server_path = '%s.%s' % ('algorithm', option['algorithm'])
    logger.info('Initializing server: '+'1 server of `{}` being created.'.format(server_path + '.Server'))
    server_module = importlib.import_module(server_path)
    server = getattr(server_module, 'Server')(option, model, clients, test_data = test_data)

    logger.register_variable(server=server, clients=clients, meta=option)
    logger.initialize()
    logger.info('Ready to start.')
    return server

def output_filename(option, server):
    header = "{}_".format(option["algorithm"])
    for para,pv in server.algo_para.items(): header = header + para + "{}_".format(pv)
    output_name = header + "M{}_R{}_B{}_".format(option['model'], option['num_rounds'],option['batch_size'])+ \
                  ("E{}_".format(server.clients[0].epochs)) + \
                  "LR{:.4f}_P{:.2f}_S{}_LD{:.3f}_WD{:.3f}_CMP{}_.json".format(
                      option['learning_rate'],
                      option['proportion'],
                      option['seed'],
                      option['lr_scheduler']+option['learning_rate_decay'],
                      option['weight_decay'],
                      option['computing_config']
                  )
    return output_name
import numpy as np
import utils.fmodule
from utils import fmodule
import copy
import utils.fflow as flw
import math
import collections
import torch.multiprocessing as mp

class BasicServer:
    def __init__(self, option, model, clients, test_data=None):
        self.task = option['task']
        self.name = option['algorithm']
        self.model = model
        self.device = self.model.get_device()
        self.test_data = test_data
        self.eval_interval = option['eval_interval']
        self.num_threads = option['num_threads']
        self.calculator = fmodule.TaskCalculator(self.device, optimizer_name=option['optimizer'])
        self.clients = clients
        self.num_clients = len(self.clients)
        self.local_data_vols = [c.datavol for c in self.clients]
        self.total_data_vol = sum(self.local_data_vols)
        self.selected_clients = []
        for c in self.clients:c.set_server(self)
        self.num_rounds = option['num_rounds']
        self.decay_rate = option['learning_rate_decay']
        self.clients_per_round = max(int(self.num_clients * option['proportion']), 1)
        self.lr_scheduler_type = option['lr_scheduler']
        self.lr = option['learning_rate']
        self.algo_para = {}
        self.current_round = -1
        self.option = option

    def run(self):
        flw.logger.time_start('Total Time Cost')
        for round in range(self.num_rounds+1):
            self.current_round = round
            flw.logger.info("--------------Round {}--------------".format(round))
            flw.logger.time_start('Time Cost')
            if flw.logger.check_if_log(round, self.eval_interval):
                flw.logger.time_start('Eval Time Cost')
                flw.logger.log_per_round()
                flw.logger.time_end('Eval Time Cost')
            self.iterate(round)
            self.global_lr_scheduler(round)
            flw.logger.time_end('Time Cost')
        flw.logger.info("--------------Final Evaluation--------------")
        flw.logger.time_start('Eval Time Cost')
        flw.logger.log_per_round()
        flw.logger.time_end('Eval Time Cost')
        flw.logger.info("=================End==================")
        flw.logger.time_end('Total Time Cost')
        flw.logger.save_output_as_json()
        return

    def iterate(self, t):
        self.selected_clients = self.sample()
        models = self.communicate(self.selected_clients)['model']
        self.model = self.aggregate(models)
        return

    def communicate(self, selected_clients):
        packages_received_from_clients = []
        client_package_buffer = {}
        communicate_clients = list(set(selected_clients))
        for cid in communicate_clients:client_package_buffer[cid] = None
        if self.num_threads <= 1:
            for client_id in communicate_clients:
                response_from_client_id = self.communicate_with(client_id)
                packages_received_from_clients.append(response_from_client_id)
        else:
            pool = mp.Pool(self.num_threads)
            for client_id in communicate_clients:
                self.clients[client_id].update_device(next(utils.fmodule.dev_manager))
                packages_received_from_clients.append(pool.apply_async(self.communicate_with, args=(int(client_id),)))
            pool.close()
            pool.join()
            packages_received_from_clients = list(map(lambda x: x.get(), packages_received_from_clients))
        for i,cid in enumerate(communicate_clients): client_package_buffer[cid] = packages_received_from_clients[i]
        packages_received_from_clients = [client_package_buffer[cid] for cid in selected_clients if client_package_buffer[cid]]
        return self.unpack(packages_received_from_clients)

    def communicate_with(self, client_id):
        svr_pkg = self.pack(client_id)
        return self.clients[client_id].reply(svr_pkg)

    def pack(self, client_id):
        return {
            "model" : copy.deepcopy(self.model),
        }

    def unpack(self, packages_received_from_clients):
        res = collections.defaultdict(list)
        for cpkg in packages_received_from_clients:
            for pname, pval in cpkg.items():
                res[pname].append(pval)
        return res

    def global_lr_scheduler(self, current_round):
        if self.lr_scheduler_type == -1:
            return
        elif self.lr_scheduler_type == 0:
            """eta_{round+1} = DecayRate * eta_{round}"""
            self.lr*=self.decay_rate
            for c in self.clients:
                c.set_learning_rate(self.lr)
        elif self.lr_scheduler_type == 1:
            """eta_{round+1} = eta_0/(round+1)"""
            self.lr = self.option['learning_rate']*1.0/(current_round+1)
            for c in self.clients:
                c.set_learning_rate(self.lr)

    def test_on_clients(self, dataflag='valid'):
        all_metrics = collections.defaultdict(list)
        for c in self.clients:
            client_metrics = c.test(self.model, dataflag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)
        return all_metrics

    def test(self, model=None):
        if model is None: model=self.model
        if self.test_data:
            return self.calculator.test(model, self.test_data, batch_size = self.option['test_batch_size'])
        else:
            return None

    def init_algo_para(self, algo_paras):
        if len(self.algo_para)==0:
            return
        elif algo_paras is not None:
            assert len(self.algo_para) == len(algo_paras)
            for para_name, value in zip(self.algo_para.keys(), algo_paras):
                self.algo_para[para_name] = type(self.algo_para[para_name])(value)
        for para_name, value in self.algo_para.items():
            self.__setattr__(para_name, value)
            for c in self.clients:
                c.__setattr__(para_name, value)
        return

    @property
    def available_clients(self):
        return [cid for cid in range(self.num_clients) if self.clients[cid].available]

class BasicClient():
    def __init__(self, option, name='', train_data=None, valid_data=None):
        self.name = name
        self.train_data = train_data
        self.valid_data = valid_data
        self.datavol = len(self.train_data)
        self.data_loader = None
        self.device = next(fmodule.dev_manager)
        self.calculator = fmodule.TaskCalculator(self.device, option['optimizer'])
        self.optimizer_name = option['optimizer']
        self.learning_rate = option['learning_rate']
        self.batch_size = len(self.train_data) if option['batch_size']<0 else option['batch_size']
        self.batch_size = int(self.batch_size) if self.batch_size>=1 else int(len(self.train_data)*self.batch_size)
        self.momentum = option['momentum']
        self.weight_decay = option['weight_decay']
        if option['num_steps']>0:
            self.num_steps = option['num_steps']
            self.epochs = 1.0 * self.num_steps/(math.ceil(len(self.train_data)/self.batch_size))
        else:
            self.epochs = option['num_epochs']
            self.num_steps = self.epochs * math.ceil(len(self.train_data) / self.batch_size)
        self.model = None
        self.test_batch_size = option['test_batch_size']
        self.loader_num_workers = option['num_workers']
        self.current_steps = 0
        self.time_response = 1
        self.available = True
        self.dropped = False
        self.server = None
        self._latency=0

    @fmodule.with_multi_gpus
    def train(self, model):
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            model.zero_grad()
            loss = self.calculator.train_one_step(model, batch_data)['loss']
            loss.backward()
            optimizer.step()
        return

    @ fmodule.with_multi_gpus
    def test(self, model, dataflag='valid'):
        dataset = self.train_data if dataflag=='train' else self.valid_data
        return self.calculator.test(model, dataset, self.test_batch_size)

    def unpack(self, received_pkg):
        return received_pkg['model']

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        self.train(model)
        cpkg = self.pack(model)
        return cpkg

    def pack(self, model):
        return {
            "model" : model,
        }

    def train_loss(self, model):
        return self.test(model,'train')['loss']

    def valid_loss(self, model):
        return self.test(model)['loss']

    def set_model(self, model):
        self.model = model

    def set_server(self, server=None):
        if server is not None:
            self.server = server

    def set_local_epochs(self, epochs=None):
        if epochs is None: return
        self.epochs = epochs
        self.num_steps = self.epochs * math.ceil(len(self.train_data)/self.batch_size)
        return

    def set_batch_size(self, batch_size=None):
        if batch_size is None: return
        self.batch_size = batch_size

    def set_learning_rate(self, lr = None):
        self.learning_rate = lr if lr else self.learning_rate

    def get_time_response(self):
        return np.inf if self.dropped else self.time_response

    def get_batch_data(self):
        try:
            batch_data = next(self.data_loader)
        except:
            self.data_loader = iter(self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size, num_workers=self.loader_num_workers))
            batch_data = next(self.data_loader)
        self.current_steps = (self.current_steps+1) % self.num_steps
        if self.current_steps == 0:self.data_loader = None
        return batch_data

    def update_device(self, dev):
        self.device = dev
        self.calculator = fmodule.TaskCalculator(dev, self.calculator.optimizer_name)
import numpy as np
import scipy.stats
from algorithm.fedbase import BasicServer
from algorithm.fedbase import BasicClient as Client
from utils.logger.basic_logger import Logger
import collections
import copy
import utils.fflow as flw
import gurobipy as grb
from utils import fmodule
import torch

def softmax(x, t=1):
    x = np.array(x)
    ex = np.exp(x / t)
    sum_ex = np.sum(ex)
    return ex / sum_ex

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.algo_para = {'alpha': 0.1}
        self.init_algo_para(option['algo_para'])
        self.batch_size = option['batch_size']
        self.C_ratio = option['C_ratio']
        self.ftr = 'embedding'
        self.epsilon = 0.1
        self.sigma = 0.01

    def test_on_clients(self, dataflag='valid'):
        all_metrics = collections.defaultdict(list)
        for c in self.clients:
            client_metrics = c.test(self.model, dataflag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)
        return all_metrics

    def iterate(self, t):
        if self.current_round == 0:
            self.models = self.communicate([cid for cid in range(self.num_clients)])['model']
            cossim_features = self.get_client_feature(self.ftr)
            self.graph = self.create_graph_from_features(cossim_features, fsim=lambda x, y: 1/((0.5*scipy.stats.entropy(x, (x+y)/2, base=2)+0.5*scipy.stats.entropy(y, (x+y)/2, base=2))+1))
            self.adj = self.create_adj_from_graph(self.graph)
            self.shortest_dist, _ = self.floyd(self.adj)
        else:
            self.selected_clients = self.sample()
            models = self.communicate(self.selected_clients)['model']
            self.model = self.aggregate(models)
        return

    def sample(self):
        losses = []
        if len(self.available_clients) == 1: return self.available_clients
        for cid in self.available_clients:
            losses.append(self.clients[cid].test(self.model)['loss'])
        sort_id = np.array(losses).argsort().tolist()
        sort_id.reverse()
        num_selected = int(self.num_clients * self.C_ratio)
        self.selectedlocal_clients = np.array(self.available_clients)[sort_id][:num_selected]
        N = len(self.selectedlocal_clients)
        M = min(len(self.selectedlocal_clients), self.clients_per_round)
        Ht = 1.0 / (N * (N - 1)) * self.shortest_dist[self.selectedlocal_clients, :][:, self.selectedlocal_clients]
        datavols = np.array(self.local_data_vols)[self.selectedlocal_clients]
        sump = np.sum(datavols)
        p = datavols / sump
        for i in range(N):
            Ht[i][i] = p[i]
        m = grb.Model(name="MIP Model")
        used = [m.addVar(vtype=grb.GRB.BINARY) for _ in range(N)]
        objective = grb.quicksum(Ht[i, j] * used[i] * used[j] for i in range(0, N) for j in range(i, N))
        m.addConstr(
            lhs=grb.quicksum(used),
            sense=grb.GRB.EQUAL,
            rhs=M
        )
        m.ModelSense = grb.GRB.MAXIMIZE
        m.setObjective(objective)
        m.setParam('OutputFlag', 0)
        m.Params.TimeLimit = 5
        m.optimize()
        res = [xi.X for xi in used]
        selected_clients = []
        for cid, flag in zip(self.selectedlocal_clients, res):
            for _ in range(int(flag)):
                selected_clients.append(cid)
        return selected_clients

    def aggregate(self, models):
        datavols = np.array(self.local_data_vols)[self.selected_clients]
        sump = np.sum(datavols)
        p = datavols / sump
        return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])

    def get_client_feature(self, key='embedding'):
        if key == 'embedding':
            data_loader = iter(self.calculator.get_data_loader(self.test_data, int(self.batch_size)))
            batch_data = next(data_loader)
            noise = torch.normal(mean=torch.mean(batch_data[0]), std=torch.std(batch_data[0]), size=batch_data[0].shape).to(self.device)
            client_features = []
            for model in self.models:
                random_output = model(noise)
                random_output = model.get_embedding(noise).view(len(random_output), -1)
                mean_out = random_output.mean(axis=0)
                client_features.append(mean_out.cpu().detach().numpy())
        return client_features

    def create_adj_from_graph(self, graph, epsilon=0.01, sigma=0.1):
        n = len(graph)
        adj = np.zeros((n, n))
        for ci in range(n):
            adj[ci][ci] = 0
            for cj in range(ci + 1, n):
                if graph[ci][cj] > epsilon:
                    adj[ci][cj] = adj[cj][ci] = np.exp(-graph[ci][cj] ** 2 / sigma)
                else:
                    adj[ci][cj] = adj[cj][ci] = np.inf
        return adj

    def create_graph_from_features(self, features, fsim=None):
        if fsim == None: fsim = np.dot
        n = len(features)
        oracle_graph = np.zeros((n, n))
        for ci in range(n):
            for cj in range(ci + 1, n):
                sim = fsim(np.array(features[ci]), np.array(features[cj]))
                oracle_graph[ci][cj] = oracle_graph[cj][ci] = sim
        max_sim = np.max(oracle_graph)
        min_sim = np.min(oracle_graph)
        for ci in range(n):
            oracle_graph[ci][ci] = max_sim
        oracle_graph = (oracle_graph - min_sim) / (max_sim - min_sim)
        for ci in range(n):
            oracle_graph[ci][ci] = np.inf
        return oracle_graph

    def floyd(self, adj):
        n = len(adj)
        path = np.zeros((n, n))
        dis = copy.deepcopy(adj)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dis[i][k] + dis[k][j] < dis[i][j]:
                        dis[i][j] = dis[i][k] + dis[j][k]
                        path[i][j] = k
        cp_dis = dis.copy()
        cp_dis[np.isinf(dis)] = -1
        max_available_dist = np.max(cp_dis)
        dis = dis / max_available_dist
        maxv = np.max(dis[np.where(dis != np.inf)])
        for i in range(self.num_clients):
            for j in range(i, self.num_clients):
                if np.isinf(dis[i][j]):
                    dis[i][j] = dis[j][i] = maxv
        return dis, path

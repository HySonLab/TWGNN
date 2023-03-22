import time
from torch.optim import Adam, Adagrad
import argparse
from lib.heuristics import heuristics_k_neighbors_multiple_wavelets, heuristics_k_neighbors_single_wavelet, heuristics_random
from lib.learnable_mmf_model import Learnable_MMF
from lib.utils import load_graph_data
import torch
import numpy as np

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Random MMF')
    parser.add_argument('--adj', '-adj', type = str, default = './data/sensor_graph/adj_mx.pkl', help = 'graph_pkl_filename')
    parser.add_argument('--dir', '-dir', type = str, default = './data', help = 'Directory')
    parser.add_argument('--name', '-name', type = str, default = 'learnable_mmf', help = 'Name')
    parser.add_argument('--data_folder', '-data_folder', type = str, default = './data/', help = 'Data folder')
    parser.add_argument('--L', '-L', type = int, default = 2, help = 'L')
    parser.add_argument('--K', '-K', type = int, default = 2, help = 'K')
    parser.add_argument('--drop', '-drop', type = int, default = 1, help = 'Drop rate')
    parser.add_argument('--dim', '-dim', type = int, default = 1, help = 'Dimension left at the end')
    parser.add_argument('--epochs', '-epochs', type = int, default = 200, help = 'Number of epochs')
    parser.add_argument('--learning_rate', '-learning_rate', type = float, default = 1e-4, help = 'Learning rate')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
    parser.add_argument('--visual', '-v', type = int, default = 0, help = 'Visualization or not')
    parser.add_argument('--heuristics', '-heuristics', type = str, default = 'smart', help = 'Heuristics to find indices')
    parser.add_argument('--device', '-device', type = str, default = 'cuda', help = 'cuda/cpu')
    args = parser.parse_args()
    return args

args = _parse_args()
log_name = args.dir + "/" + args.name + ".log"
model_name = args.dir + "/" + args.name + ".model"
LOG = open(log_name, "w")

# Fix CPU torch random seed
torch.manual_seed(args.seed)

# Fix GPU torch random seed
torch.cuda.manual_seed(args.seed)

# Fix the Numpy random seed
np.random.seed(args.seed)

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = args.device
print(device)

sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(args.adj)

N = adj_mx.shape[0]
K = args.K
L = args.L
drop = args.drop
dim = args.dim

adj = torch.tensor(adj_mx, dtype=torch.float)
adj = adj + torch.eye(N)
D = torch.sum(adj, dim=0)
DD = torch.diag(1.0 / torch.sqrt(D))

A = torch.matmul(torch.matmul(DD, torch.diag(D) - adj), DD)
A_sparse = A.to_sparse()

if args.heuristics == 'random':
    wavelet_indices, rest_indices = heuristics_random(A_sparse, L, K, drop, dim)
else:
    if drop == 1:
        wavelet_indices, rest_indices = heuristics_k_neighbors_single_wavelet(A_sparse, L, K, drop, dim)
    else:
        wavelet_indices, rest_indices = heuristics_k_neighbors_multiple_wavelets(A_sparse, L, K, drop, dim)

# Execute the randomized MMF
model = Learnable_MMF(A, L, K, drop, dim, wavelet_indices, rest_indices)
optimizer = Adagrad(model.parameters(), lr = args.learning_rate)

# Training
all_losses = []
all_errors = []
norm = torch.norm(A, p = 'fro')
A_prev = A
diff = 1
thres = 1e-4

best = 1e9
for epoch in range(args.epochs):
    print('Epoch', epoch, ' --------------------')
    t = time.time()
    optimizer.zero_grad()

    A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = model()

    loss = torch.norm(A - A_rec, p = 'fro')
    loss.backward()

    error = loss / norm
    all_losses.append(loss.item())
    all_errors.append(error.item())
    diff = torch.norm(A_prev - A_rec, 'fro') / torch.norm(A_rec, 'fro')

    print('Loss =', loss.item())
    print('Error =', error.item())
    print('Diff =', diff)
    print('Time =', time.time() - t)
    LOG.write('Loss = ' + str(loss.item()) + '\n')
    LOG.write('Error = ' + str(error.item()) + '\n')
    LOG.write('Diff = ' + str(diff) + '\n')
    LOG.write('Time = ' + str(time.time() - t) + '\n')

    # Early stopping
    if loss.item() > best or diff < thres:
        break
    best = loss.item()
    A_prev = A_rec

    # if epoch % 1000 == 0:
        # torch.save(model.state_dict(), args.dir + '/' + args.name + '.model')
        # print('Save model to file')

    for l in range(L):
        X = torch.Tensor(model.all_O[l].data)
        G = torch.Tensor(model.all_O[l].grad.data)
        Z = torch.matmul(G, X.transpose(0, 1)) - torch.matmul(X, G.transpose(0, 1))
        tau = args.learning_rate
        Y = torch.matmul(torch.matmul(torch.inverse(torch.eye(K) + tau / 2 * Z), torch.eye(K) - tau / 2 * Z), X)
        model.all_O[l].data = Y.data

# torch.save(model.state_dict(), args.dir + '/' + args.name + '.model')
print('Save wavelets to file')
A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = model()
torch.save(mother_wavelets.detach(), args.dir + '/' + args.name + '.mother_wavelets.pt')
torch.save(father_wavelets.detach(), args.dir + '/' + args.name + '.father_wavelets.pt')

import torch
import argparse
from lib.baseline_mmf_model import Baseline_MMF
from lib.utils import load_graph_data
import numpy as np

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Baseline MMF')
    parser.add_argument('--adj', '-adj', type = str, default = './data/sensor_graph/adj_mx.pkl', help = 'graph_pkl_filename')
    parser.add_argument('--dir', '-dir', type = str, default = './data', help = 'Directory')
    parser.add_argument('--name', '-name', type = str, default = 'learnable_mmf', help = 'Name')
    parser.add_argument('--data_folder', '-data_folder', type = str, default = './data/', help = 'Data folder')
    parser.add_argument('--L', '-L', type = int, default = 2, help = 'L')
    parser.add_argument('--dim', '-dim', type = int, default = 1, help = 'Dimension left at the end')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
    parser.add_argument('--device', '-device', type = str, default = 'cuda', help = 'cuda/cpu')
    args = parser.parse_args()
    return args

args = _parse_args()

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
L = args.L
dim = args.dim

adj = torch.tensor(adj_mx, dtype=torch.float)
adj = adj + torch.eye(N)
D = torch.sum(adj, dim=0)
DD = torch.diag(1.0 / torch.sqrt(D))

A = torch.matmul(torch.matmul(DD, torch.diag(D) - adj), DD)

model = Baseline_MMF(N, L, dim)

A_rec, right, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = model(A)

print('Error =', torch.norm(A_rec-A, 'fro') / torch.norm(A, 'fro'))
torch.save(mother_wavelets.detach(), args.dir + '/' + args.name + '.mother_wavelets.pt')
torch.save(father_wavelets.detach(), args.dir + '/' + args.name + '.father_wavelets.pt')

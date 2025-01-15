import torch

from pina import Condition, Graph, TorchOptimizer
from pina.problem import AbstractProblem
from pina.solvers import SupervisedSolver
from pina.model import GNO
from tqdm import tqdm
'''
import torch
from pina.solvers import SupervisedSolver

class GraphSolver(SupervisedSolver):
    def __init__(self,
                 problem,
                 model,
                 loss=None,
                 optimizer=None,
                 scheduler=None,
                ):
        super().__init__(problem, model, loss, optimizer, scheduler, None, False)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        t = batch.y
        return self._model(x, edge_index, edge_attr, t)
    
    def loss_data(self, input_pts, output_pts):
        if isinstance(output_pts, torch.Tensor):
            output_pts = output_pts.reshape(-1, * output_pts.shape[2:])
        return super().loss_data(input_pts, output_pts)

import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Tanh
from pina.model import FeedForward
from torch.nn import Parameter, Linear, Sequential

class GraphIntegralKernel(MessagePassing):
    def __init__(self,
                 width,
                 kernel_width,
                 n_layers=0,
                 inner_size=None,
                 layers=None
                 ):
        super(GraphIntegralKernel, self).__init__(aggr='add')
        self.dense = FeedForward(input_dimensions=kernel_width,
                                 output_dimensions=width**2,
                                 n_layers=n_layers,
                                 inner_size=inner_size,
                                 layers=layers)
        self.dense = Linear(kernel_width, width**2)
        self.width = width
        self.W = Parameter(torch.rand(width, width))

    def message(self, x_j, edge_attr):
        
        x = self.dense(edge_attr).view(-1, self.width, self.width)
        return torch.einsum('bij,bj->bi', x, x_j)
    
    def update(self, aggr_out, x):
        aggr_out = aggr_out + torch.mm(x, self.W)
        return aggr_out

    def forward(self, x, edge_index, edge_attr):    
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

class GNOTime(torch.nn.Module):
    def __init__(self,
                 lifting_operator,
                 projection_operator,
                 edge_features,
                 n_layers=1,
                 kernel_n_layers=0,
                 kernel_inner_size=None,
                 kernel_layers=None
                 ):
        super(GNOTime, self).__init__()
        self.lifting_operator = lifting_operator
        self.projection_operator = projection_operator
        self.tanh = Tanh()
        self.kernels = torch.nn.ModuleList(
            [GraphIntegralKernel(width=lifting_operator.out_features,
                                 kernel_width=edge_features,
                                 n_layers=kernel_n_layers,
                                 inner_size=kernel_inner_size,
                                 layers=kernel_layers
                                 )
             for _ in range(n_layers)])

    def forward(self, x, edge_index, edge_attr, t):
        x = torch.cat([x,t], dim=1)
        x = self.lifting_operator(x)
        for kernel in self.kernels:
            x = kernel(x, edge_index, edge_attr)
            x = self.tanh(x)
        x = self.projection_operator(x)
        return x
'''
def load_data():
    torch.manual_seed(1931)
    torch.cuda.manual_seed(1931)
    proc_numbers = ['00']
    conditions = {}
    for file in tqdm(proc_numbers):
        pos = torch.load('points_'+file+'.pt')#.to(torch.float32)
        times = torch.load('times.pt')#.to(torch.float32)
        edge_index = torch.load('edges_'+file+'.pt').T.to(torch.int64)
        values = torch.load('output_'+file+'.pt')#.to(torch.float32)
        u = torch.load('u_'+file+'.pt')#.to(torch.float32)
        output_ = values[1:,:,:]
        x = [values[i,:,:] for i in range(len(times)-1)]
        input_ = Graph(x=x, pos=pos, build_edge_attr=True, edge_index=edge_index).data
        conditions[file] = Condition(input_points=input_, output_points=output_)
    return conditions

torch.manual_seed(1931)
torch.cuda.manual_seed(1931)
class GraphProblem(AbstractProblem):
    output_variables = None
    conditions = {}

lifting_operator = torch.nn.Linear(in_features=1, out_features=8)
projection_operator = torch.nn.Linear(in_features=8, out_features=3)
model = GNO(edge_features=6, lifting_operator=lifting_operator, projection_operator=projection_operator,
            n_layers=1).double()
problem = GraphProblem()
optimizer = TorchOptimizer(torch.optim.AdamW, lr=1e-3)
solver = SupervisedSolver.load_from_checkpoint(checkpoint_path='checkpoints/n1_dim8_u/best-model.ckpt', problem=problem,
                                               model=model, use_lt=False)
model = solver._pina_models[0]
model.to('cpu')
model.eval()
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, "scripted_model_u.pt")

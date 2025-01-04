import torch

from pina import Condition, Graph, TorchOptimizer
from pina.problem import AbstractProblem
from pina.solvers import SupervisedSolver
from pina.model import GNO
from tqdm import tqdm

def load_data():
    torch.manual_seed(1931)
    torch.cuda.manual_seed(1931)
    proc_numbers = ['00']
    conditions = {}
    for file in tqdm(proc_numbers):
        pos = torch.load('points_'+file+'.pt').to(torch.float32)
        times = torch.load('times.pt').to(torch.float32)
        edge_index = torch.load('edges_'+file+'.pt').T.to(torch.int64)
        values = torch.load('output_'+file+'.pt').to(torch.float32)
        output_ = values[1:,:,:]
        x = [torch.cat((values[i,:,:], times[i].repeat(output_.shape[1]).unsqueeze(1)), dim=1) for i in range(len(times)-1)]
        input_ = Graph(x=x, pos=pos, build_edge_attr=True, edge_index=edge_index).data
        conditions[file] = Condition(input_points=input_, output_points=output_)
    return conditions

torch.manual_seed(1931)
torch.cuda.manual_seed(1931)
class GraphProblem(AbstractProblem):
    output_variables = None
    conditions = {}

lifting_operator = torch.nn.Linear(in_features=3, out_features=8)
projection_operator = torch.nn.Linear(in_features=8, out_features=3)
model = GNO(node_features=3, edge_features=7, lifting_operator=lifting_operator, projection_operator=projection_operator,
            n_layers=1)
problem = GraphProblem()
optimizer = TorchOptimizer(torch.optim.AdamW, lr=1e-3)
solver = SupervisedSolver.load_from_checkpoint(checkpoint_path='checkpoints/n1_dim8/best-model.ckpt', problem=problem,
                                               model=model, use_lt=False)
model = solver._pina_models[0]
model.to('cpu')
model.eval()
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, "scripted_model.pt")

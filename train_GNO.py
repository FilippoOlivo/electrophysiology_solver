import torch

from pina.trainer import Trainer

from pina import LabelTensor, Condition, Graph
from pina.problem import AbstractProblem
from pina.solvers import SupervisedSolver
from pina.model import GNO


pos = torch.load('points_00.pt')
pos = pos.to(torch.float32)
times = torch.load('times.pt')
times = times.to(torch.float32)
edge_index = torch.load('edged_00.pt').T

x =[torch.cat((pos, times[i].repeat(7900).unsqueeze(1)), dim=1) for i in range(len(times))]
output_ = torch.load('output_00.pt')
output_ = output_.to(torch.float32)
data = Graph(x=x, pos=pos, edge_index=edge_index, build_edge_attr=True).data
class GraphProblem(AbstractProblem):

    output_variables = None
    conditions = {
        'graph_data': Condition(input_points=data, output_points=output_)
    }

lifting_operator = torch.nn.Linear(in_features=5, out_features=32)
projection_operator = torch.nn.Linear(in_features=32, out_features=3)
model = GNO(lifting_operator=lifting_operator, projection_operator=projection_operator, n_layers=8)
problem = GraphProblem()
solver = SupervisedSolver(problem, model=model, use_lt=False)
trainer = Trainer(solver, batch_size=32, accelerator='gpu')
trainer.train()


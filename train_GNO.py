import torch

from pina.trainer import Trainer
import lightning
from pina import Condition, Graph, TorchOptimizer
from pina.problem import AbstractProblem
from pina.solvers import SupervisedSolver
from pina.model import GNO
from tqdm import tqdm
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


proc_numbers = ['00', '01', '02', '03', '04', '05']
data = []
output_ = None
conditions = {}
for file in tqdm(proc_numbers):
    pos = torch.load('points_'+file+'.pt').to(torch.float32)
    times = torch.load('times.pt').to(torch.float32)
    edge_index = torch.load('edged_'+file+'.pt').T.to(torch.int64)
    values = torch.load('output_'+file+'.pt').to(torch.float32)
    output_ = values[1:,:,:]
    x = [torch.cat((values[i,:,:], times[i].repeat(output_.shape[1]).unsqueeze(1)), dim=1) for i in range(len(times)-1)]
    input_ = Graph(x=x, pos=pos, build_edge_attr=True, edge_index=edge_index).data
    print(input_[0].edge_index.shape)
    input_ = Graph(x=x, pos=pos, build_edge_attr=True, method="knn", k=4, undirected=True).data
    print(input_[0].edge_index.shape)
    conditions[file] = Condition(input_points=input_, output_points=output_)
class GraphProblem(AbstractProblem):
    output_variables = None
    conditions = conditions

lifting_operator = torch.nn.Linear(in_features=4, out_features=16)
projection_operator = torch.nn.Linear(in_features=16, out_features=3)
model = GNO(node_features=4, edge_features=7, lifting_operator=lifting_operator, projection_operator=projection_operator, n_layers=2)
fabric = lightning.Fabric(devices=1)
model = torch.compile(model)
model = fabric.setup(model)
problem = GraphProblem()
optimizer = TorchOptimizer(torch.optim.AdamW, lr=1e-3)
solver = SupervisedSolver(problem, model=model, use_lt=False, optimizer=optimizer)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    verbose=False,
    mode="min"
)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints/",
    filename="best-model",
    save_top_k=1,
    mode="min",
    verbose=False
)

trainer = Trainer(solver, batch_size=10, accelerator='gpu', max_epochs=500, val_size=0.25, train_size=0.75,
                  test_size=0.0, callbacks=[early_stopping, checkpoint_callback])
trainer.train()


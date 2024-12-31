import torch
from pina.trainer import Trainer
from pina import Condition, Graph, TorchOptimizer
from pina.problem import AbstractProblem
from pina.solvers import GraphSupervisedSolver
from pina.model import GNO
from tqdm import tqdm
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import argparse

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', type=int, required=True,)
    parser.add_argument('--hidden_dim', type=int, required=True,)
    return parser.parse_args()

def load_data():
    torch.manual_seed(1931)
    torch.cuda.manual_seed(1931)
    proc_numbers = ['00', '01', '02', '03', '04', '05', '06', '07']#, '08', '09']
    conditions = {}
    for file in tqdm(proc_numbers):
        pos = torch.load('points_'+file+'.pt', weights_only=False).to(torch.float32)
        times = torch.load('times.pt', weights_only=False).to(torch.float32)
        edge_index = torch.load('edges_'+file+'.pt', weights_only=False).T.to(torch.int64)
        values = torch.load('output_'+file+'.pt', weights_only=False).to(torch.float32)
        attr = torch.load('attr_'+file+'.pt', weights_only=False).to(torch.float32)
        print(attr.shape)
        edge_attr = torch.cat([attr.unsqueeze(-1), pos[edge_index[0]], pos[edge_index[1]]], dim=-1)
        output_ = values[1:,:,:]
        print(edge_attr.shape)
        x = [torch.cat((values[i,:,:], times[i].repeat(output_.shape[1]).unsqueeze(1)), dim=1) for i in range(len(times)-1)]
        input_ = Graph(x=x, pos=pos, build_edge_attr=False, edge_attr=edge_attr, edge_index=edge_index).data
        conditions[file] = Condition(input_points=input_, output_points=output_)
    return conditions

def define_solver(problem, n_layers, hidden_dim):
    lifting_operator = torch.nn.Linear(in_features=4, out_features=hidden_dim)
    projection_operator = torch.nn.Linear(in_features=hidden_dim, out_features=3)
    model = GNO(node_features=4, edge_features=7, lifting_operator=lifting_operator, projection_operator=projection_operator, n_layers=n_layers)
    optimizer = TorchOptimizer(torch.optim.AdamW, lr=1e-3)
    solver = GraphSupervisedSolver(problem, model=model, optimizer=optimizer)
    return solver

def train_model(solver):
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
                      test_size=0.0, callbacks=[early_stopping, checkpoint_callback], log_every_n_steps=0)
    trainer.train()

if __name__ == '__main__':
    args = parse_command_line_arguments()
    hidden_dim = args.hidden_dim
    n_layers = args.n_layers
    conditions = load_data()
    class GraphProblem(AbstractProblem):
        output_variables = None
        conditions = conditions
    problem = GraphProblem()
    solver = define_solver(problem, n_layers, hidden_dim)
    train_model(solver)



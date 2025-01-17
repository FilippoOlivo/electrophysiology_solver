import torch
from pina.trainer import Trainer
from pina import Condition, Graph, TorchOptimizer
from pina.problem import AbstractProblem
from tqdm import tqdm
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import argparse
from lightning.pytorch.loggers import TensorBoardLogger
from pina.model import GNO
from pina.solvers import GraphSupervisedSolver


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', type=int, required=True,)
    parser.add_argument('--hidden_dim', type=int, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--nproc', type=int, required=True)
    return parser.parse_args()


def load_data(nproc):
    torch.manual_seed(1931)
    torch.cuda.manual_seed(1931)
    proc_numbers = [f"{i:02}" for i in range(nproc)]
    conditions = {}
    dt = 1e-5
    for file in tqdm(proc_numbers):
        pos = torch.load('data/points_'+file+'.pt', weights_only=False)
        times = torch.load('times.pt', weights_only=False)
        edge_index = torch.load('data/edges_'+file+'.pt',
                                weights_only=False).T.to(torch.int64)
        w = torch.load('data/w_'+file+'.pt', weights_only=False)
        u = torch.load('data/u_'+file+'.pt', weights_only=False)
        edge_attr = torch.cat([pos[edge_index[0]], pos[edge_index[1]]], dim=-1)
        output_ = w[1:, :, :]

        x = [u[i, :].unsqueeze(-1) for i in range(len(output_))]
        input_ = Graph(x=x, pos=pos, build_edge_attr=False,
                       edge_attr=edge_attr, edge_index=edge_index).data
        conditions[file] = Condition(
            input_points=input_, output_points=output_)
    return conditions


def define_solver(problem, n_layers, hidden_dim):
    lifting_operator = torch.nn.Linear(in_features=3, out_features=hidden_dim)
    projection_operator = torch.nn.Linear(
        in_features=hidden_dim, out_features=3)
    model = GNO(edge_features=6, lifting_operator=lifting_operator,
                projection_operator=projection_operator, n_layers=n_layers).double()
    optimizer = TorchOptimizer(torch.optim.AdamW, lr=1e-3)
    solver = GraphSupervisedSolver(problem, model=model, optimizer=optimizer)
    return solver


def train_model(solver, n_layers, hidden_dim):
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        verbose=False,
        mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/" + "n" +
        str(n_layers) + "_dim" + str(hidden_dim) + "_w",
        filename="best-model",
        save_top_k=1,
        mode="min",
        verbose=False
    )

    logger = TensorBoardLogger(
        save_dir='lightning_logs', name="n" + str(n_layers) + "_dim" + str(hidden_dim) + "_w")
    trainer = Trainer(solver, batch_size=5, accelerator='gpu', max_epochs=1, val_size=0.25, train_size=0.75,
                      test_size=0.0, callbacks=[early_stopping, checkpoint_callback], log_every_n_steps=0, logger=logger,
                      devices=2, strategy='ddp', num_nodes=1)
    trainer.train()


def save_compiled_model(solver, hidden_dim, n_layers, model_name):
    model = solver._pina_models[0]
    problem = solver.problem
    solver = GraphSupervisedSolver.load_from_checkpoint('checkpoints/'+'n' + str(n_layers) + '_dim' + str(
        hidden_dim) + '_w'+'/best-model.ckpt', problem=problem, model=model, use_lt=False)
    model = solver._pina_models[0]
    model.to('cpu')
    model.eval()
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, model_name)


if __name__ == '__main__':
    args = parse_command_line_arguments()
    hidden_dim = args.hidden_dim
    n_layers = args.n_layers
    model_name = args.model_name
    nproc = args.nproc
    conditions = load_data(nproc)

    class GraphProblem(AbstractProblem):
        output_variables = None
        conditions = conditions
    problem = GraphProblem()
    solver = define_solver(problem, n_layers, hidden_dim)
    train_model(solver, n_layers, hidden_dim)
    save_compiled_model(solver, hidden_dim, n_layers, model_name)

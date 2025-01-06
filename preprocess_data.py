import numpy as np
import os
from tqdm import tqdm
import torch

proc_numbers = ['00', '01', '02', '03', '04', '05', '06', '07']

for n in proc_numbers:
    directory = 'build/'
    times = []
    all_data = None
    points = torch.jit.load('build/points_' + n + '.pt')
    points = list(points.parameters())[0]
    torch.save(points, 'points.pt')

    edge_index = torch.jit.load('build/edges_' + n + '.pt')
    edge_index = list(edge_index.parameters())[0]
    torch.save(edge_index, 'edges.pt')

    edge_attr = torch.jit.load('build/attr_' + n + '.pt')
    edge_attr = list(edge_attr.parameters())[0]
    torch.save(edge_attr, 'attr.pt')
    directory = directory + 'snapshot/'
    file_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(n+'.pt')]

    for file in tqdm(file_names):
        times.append(float(file.replace('_values_'+n+'.pt', '')))
        tensor = torch.jit.load(directory + file)
        tensor = list(tensor.parameters())[0].unsqueeze(0)
        if all_data is None:
            all_data = tensor
        else:
            all_data = torch.vstack([all_data, tensor])
    torch.save(all_data, 'output_'+n+'.pt')
    times = torch.tensor(np.array(times))
    torch.save(times, 'times.pt')
    
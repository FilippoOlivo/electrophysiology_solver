import numpy as np
import os
from tqdm import tqdm
import torch

proc_numbers = ['00', '01', '02', '03', '04', '05', '06', '07']
tensors_names = ["w_", "u_"]
for n in proc_numbers:
    directory = 'build_inference/'
    times = []
    all_data = None
    points = torch.jit.load('build_inference/points_' + n + '.pt')
    points = list(points.parameters())[0]
    torch.save(points, 'data/points_'+n+'.pt')
    
    edge_index = torch.jit.load('build_inference/edges_' + n + '.pt')
    edge_index = list(edge_index.parameters())[0]
    torch.save(edge_index, 'data/edges_'+n+'.pt')
    
    edge_attr = torch.jit.load('build_inference/attr_' + n + '.pt')
    edge_attr = list(edge_attr.parameters())[0]
    torch.save(edge_attr, 'data/attr_'+n+'.pt')
    
    directory = directory + 'snapshot/'
    for name in tensors_names:
        file_names = sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(n+'.pt') and name in f])
        all_data = None
        for file in tqdm(file_names):
            tensor = torch.jit.load(directory + file)
            tensor = list(tensor.parameters())[0].unsqueeze(0)
            if all_data is None:
                all_data = tensor
            else:
                all_data = torch.vstack([all_data, tensor])
        
        torch.save(all_data, "data/"+name+n+'.pt')
    
#!/home/folivo/yes/envs/atsc/bin/python
import numpy as np
import os
from tqdm import tqdm
import torch
directory = 'build/snapshot/'
proc_numbers = ['06', '07', '00', '01', '02', '03', '04', '05']
for n in proc_numbers:
    file_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(n+'.npy')]
    times = []
    points = torch.tensor(np.load('build/points_'+n+'.npy'))
    torch.save(points, 'points_'+n+'.pt')
    edges = torch.tensor(np.load('build/edges_'+n+'.npy'), )
    edges = edges.to(torch.int32)
    torch.save(edges, 'edges_'+n+'.pt')
    attr = torch.tensor(np.load('build/attr_'+n+'.npy'), ) 
    torch.save(attr, 'attr_'+n+'.pt')
    all_data = None

    for file in tqdm(file_names):
        times.append(float(file.replace('_values_'+n+'.npy', '')))
        data = torch.tensor([np.load(directory + file)])
        if all_data is None:
            all_data = data
        else:
            all_data = torch.vstack([all_data,data])
    torch.save(all_data, 'output_'+n+'.pt')
    times = torch.tensor(np.array(times))
    torch.save(times, 'times.pt')





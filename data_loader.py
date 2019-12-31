import torch.utils.data


class MyData(torch.utils.data.Dataset):
    def __init__(self, dt, lb):
        self.dt = dt
        self.lb = lb

    def __len__(self):
        return len(self.dt)

    def __getitem__(self, index):
        return self.lb[index], np.array(self.dt[index])


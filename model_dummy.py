def __init__(self):
    super(Conv_AE, self).__init__()

    self.Encoder = nn.Sequential(
        nn.Conv1d(1, 4, 30, 2),
        nn.BatchNorm1d(4),
        nn.Tanh(),
        nn.Conv1d(4, 8, 20, 2),
        nn.BatchNorm1d(8),
        nn.Tanh(),
        nn.Conv1d(8, 16, 10, 2),
        nn.BatchNorm1d(16),
        nn.Tanh(),
        nn.Conv1d(16, 32, 10, 1),
        nn.BatchNorm1d(32),
        nn.Tanh(),
        nn.Conv1d(32, 64, 10, 1),
        nn.BatchNorm1d(64),
        nn.Tanh(),
    )

    self.Decoder = nn.Sequential(
        nn.ConvTranspose1d(64, 32, 10, 1),
        nn.BatchNorm1d(32),
        nn.Tanh(),
        nn.ConvTranspose1d(32, 16, 10, 1),
        nn.BatchNorm1d(16),
        nn.Tanh(),
        nn.ConvTranspose1d(16, 8, 10, 2),
        nn.BatchNorm1d(8),
        nn.Tanh(),
        nn.ConvTranspose1d(8, 4, 20, 2),
        nn.BatchNorm1d(4),
        nn.Tanh(),
        nn.ConvTranspose1d(4, 1, 30, 2),
        nn.Tanh()
    )

    self.linear1 = nn.Linear(720, 720)


def __init__(self):
    super(Conv_AE ,self).__init__()

    self.Encoder = nn.Sequential(
        nn.Conv1d(1 ,8 ,30 ,2),
        nn.BatchNorm1d(8),
        nn.Tanh(),
        nn.Conv1d(8 ,16 ,20 ,2),
        nn.BatchNorm1d(16),
        nn.Tanh(),
        nn.Conv1d(16 ,32 ,10 ,2),
        nn.BatchNorm1d(32),
        nn.Tanh(),
        nn.Conv1d(32 ,64 ,10 ,1),
        nn.BatchNorm1d(64),
        nn.Tanh(),
        nn.Conv1d(64 ,128 ,10 ,1),
        nn.BatchNorm1d(128),
        nn.Tanh(),
    )

    self.Decoder = nn.Sequential(
        nn.ConvTranspose1d(128 ,64 ,10 ,1),
        nn.BatchNorm1d(64),
        nn.Tanh(),
        nn.ConvTranspose1d(64 ,32 ,10 ,1),
        nn.BatchNorm1d(32),
        nn.Tanh(),
        nn.ConvTranspose1d(32 ,16 ,10 ,2),
        nn.BatchNorm1d(16),
        nn.Tanh(),
        nn.ConvTranspose1d(16 ,8 ,20 ,2),
        nn.BatchNorm1d(8),
        nn.Tanh(),
        nn.ConvTranspose1d(8 ,1 ,30,2),
        nn.Tanh()
    )

    self.linear1 = nn.Linear(720,720)

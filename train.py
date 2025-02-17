from sys import argv
import time
import shutil

# from progress.bar import Bar
from rich.progress import Progress

# hidden_size, num_layers, dropout
if len(argv) < 9:
    print(
        "Usage: python main.py <filename> <model-output-filename-base> <seq-length> <n-epochs> <batch-size> <hidden-size> <num-layers> <dropout>"
    )
    exit(1)

filename = argv[1]
output_filename_base = argv[2]
raw_text = open(filename, "r", encoding="utf-8").read()
raw_text = raw_text.lower()


n_epochs = int(argv[4])
if n_epochs < 1:
    print("Number of epochs must be at least 1.")
    exit(1)

batch_size = int(argv[5])
if batch_size < 1:
    print("Batch size must be at least 1.")
    exit(1)

seq_length = int(argv[3])
if seq_length < 1:
    print("Sequence length must be at least 1.")
    exit(1)


hidden_size = int(argv[6])
if hidden_size < 1:
    print("Hidden size must be at least 1.")
    exit(1)

num_layers = int(argv[7])
if num_layers < 1:
    print("Number of layers must be at least 1.")
    exit(1)

dropout = float(argv[8])
if dropout < 0 or dropout >= 1:
    print("Dropout must be between 0 and 1.")
    exit(1)


# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
# print("Total Characters: ", n_chars)
# print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i : i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)


import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.utils.data as data


# reshape X to be [samples, time steps, features]
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(n_vocab)
y = torch.tensor(dataY)


class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            # input_size=1, hidden_size=256, num_layers=4, batch_first=True, dropout=0.1
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(256, n_vocab)

    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x


model = CharModel()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# print("Checking the model")

optimizer = optim.Adam(model.parameters())

# print("Training the model 1")

loss_fn = nn.CrossEntropyLoss(reduction="sum")

# print("Training the model 2")

loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)


best_model = None
best_loss = np.inf

# tqdm
print("Training the model...")

start_times = []
stop_times = []
diff_times = []
avg_times = []
total_time = 0
training_interrupted = False

best_loss_str = None
best_epoch = None

with Progress() as progress:
    task1 = progress.add_task("Epochs...", total=n_epochs)
    for epoch in range(n_epochs):
        try:
            start_time = time.time()
            model.train()
            i = 0
            len_loader = len(loader)
            for X_batch, y_batch in loader:
                y_pred = model(X_batch.to(device))
                loss = loss_fn(y_pred, y_batch.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            loss = 0
            i = 0
            is_best = False
            with torch.no_grad():
                for X_batch, y_batch in loader:
                    y_pred = model(X_batch.to(device))
                    loss += loss_fn(y_pred, y_batch.to(device))
                if loss < best_loss:
                    best_loss = loss
                    best_model = model.state_dict()
                    is_best = True

            stop_time = time.time()
            diff_time = stop_time - start_time
            start_times.append(start_time)
            stop_times.append(stop_time)
            diff_times.append(diff_time)
            avg_time = np.mean(diff_times)
            avg_times.append(avg_time)
            total_time += diff_time
            if is_best:
                # format the total_time into H:M:S
                total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
                epoch_str = f"{epoch}/{n_epochs}"
                best_loss_str = f"{best_loss:.8f}"
                out_str = f"{total_time_str:12} {epoch_str:12} {best_loss_str:12}"
                best_epoch = epoch
                print(out_str)
            progress.update(task1, advance=1)
        except KeyboardInterrupt:
            print("Training interrupted.")
            print("Writing current progress to file.")
            training_interrupted = True
            real_output_filename = f"{output_filename_base}-{seq_length}-{hidden_size}-{num_layers}-{dropout}-{best_loss_str}.pth"
            torch.save([best_model, char_to_int], real_output_filename)
            answer = input("Do you wish to continue training? (y/n) ")
            if answer.lower() == "y":
                continue
            else:
                break


real_output_filename = f"{output_filename_base}-{seq_length}-{hidden_size}-{num_layers}-{dropout}-{best_loss_str}.pth"
torch.save([best_model, char_to_int], real_output_filename)

print("Done.")

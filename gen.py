from sys import argv

# load ascii text and covert to lowercase
# filename = "hello.txt"
# filename = "wonderland.txt"


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint["epoch"]


if len(argv) < 5:
    print(
        "Usage: python main.py <filename> <model-out-filename> <prompt-seq-length> <num-chars>"
    )
    exit(1)


filename = argv[1]

try:
    open(filename, "r", encoding="utf-8").close()
except FileNotFoundError:
    print("File not found")
    exit(1)


model_out_filename = argv[2]

try:
    open(model_out_filename, "r").close()
except FileNotFoundError:
    print("Model file not found")
    exit(1)

seq_length = int(argv[3])
num_chars = int(argv[4])

if num_chars < 1:
    print("Number of characters must be positive")
    exit(1)

raw_text = open(filename, "r", encoding="utf-8").read()
raw_text = raw_text.lower()


# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
# print("Total Characters: ", n_chars)
# print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
# seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i : i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
# print("Total Patterns: ", n_patterns)


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm


class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=256, num_layers=3, batch_first=True, dropout=0.2
        )
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)

    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x


# print("Creating model...")
model = CharModel()
device_input = "cuda:0" if torch.cuda.is_available() else "cpu"
# print("Using device:", device_input)
device = torch.device(device_input)
# device = torch.device("cpu")
model.to(device)

optimizer = optim.Adam(model.parameters())


# print("Loading data...")
# Generation using the trained model

best_model, char_to_int = torch.load(model_out_filename)
# best_model = torch.load(model_out_filename)

# best_model, optimizer, epoch = load_ckp(model_out_filename, model, optimizer)

n_vocab = len(char_to_int)
int_to_char = dict((i, c) for c, i in char_to_int.items())
# print(int_to_char)

model.load_state_dict(best_model)

# randomly generate a prompt
# filename = "hello.txt"
# seq_length = argv[3]
# num_seqs = 10
raw_text = open(filename, "r", encoding="utf-8").read()
raw_text = raw_text.lower()
raw_text_words = raw_text.split()
# raw_text = raw_text.replace("\n", " ")

start = np.random.randint(0, len(raw_text) - seq_length)
prompt = raw_text[start : start + seq_length]

# random_index = np.random.randint(0, len(raw_text_words) - 1)

# prompt = raw_text_words[random_index]
prompt = raw_text[start : start + seq_length]
# prompt = "<@stratum> "
# prompt = input("Enter prompt: ")
prompt = prompt.strip().lower()
pattern = [char_to_int[c] for c in prompt]

# prompt_list = prompt.split()
# prompt = prompt_list[0]


# while True:
# try:
#    prompt = input("Enter prompt: ")
# except Exception as e:
# print("Error:", e)
#    break
model.eval()
# print(f"Prompt: {prompt}")
print("Generating text...")
resulting_text = []
with torch.no_grad():
    for i in range(num_chars):
        # format input array of int into PyTorch tensor
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)
        # generate logits as output from the model
        prediction = model(x.to(device))
        # convert logits into one character
        index = int(prediction.argmax())
        result = int_to_char[index]
        resulting_text.append(result)
        # print(result, end="")
        # append the new character into the prompt for the next iteration
        pattern.append(index)
        pattern = pattern[1:]
print(f"Prompt: {prompt}\n")
print(f"{''.join(resulting_text)}\n")
# print("".join(resulting_text))
# print()

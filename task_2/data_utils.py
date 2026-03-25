import torch
from torch.nn.utils.rnn import pad_sequence


def load_names(file_path="TrainingNames.txt"):
    with open(file_path, "r") as f:
        raw_names = f.read().splitlines()

    return ["<" + name.lower() + ">" for name in raw_names]


def build_vocab(names, add_pad_token=True, pad_token="~"):
    chars = sorted(list(set("".join(names))))
    if add_pad_token and pad_token not in chars:
        chars.append(pad_token)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos


def encode(name, stoi):
    return [stoi[ch] for ch in name]


def decode(indices, itos):
    return "".join([itos[i] for i in indices])


class NameDataset(torch.utils.data.Dataset):
    def __init__(self, names, stoi):
        self.data = [encode(name, stoi) for name in names]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)


def build_collate_fn(input_pad_idx, target_pad_idx=-100):
    def collate_fn(batch):
        xs, ys = zip(*batch)
        x_padded = pad_sequence(xs, batch_first=True, padding_value=input_pad_idx)
        y_padded = pad_sequence(ys, batch_first=True, padding_value=target_pad_idx)
        return x_padded, y_padded

    return collate_fn

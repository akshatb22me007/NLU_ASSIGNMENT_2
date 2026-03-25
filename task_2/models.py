import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.Wxh = nn.Linear(hidden_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        x = self.embedding(x)
        h = x.new_zeros(batch_size, self.hidden_size)
        outputs = []

        for t in range(seq_len):
            xt = x[:, t, :]
            h = torch.tanh(self.Wxh(xt) + self.Whh(h))
            out = self.fc(h)
            outputs.append(out.unsqueeze(1))

        return torch.cat(outputs, dim=1)


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.Wf = nn.Linear(input_size, hidden_size)
        self.Uf = nn.Linear(hidden_size, hidden_size)

        self.Wi = nn.Linear(input_size, hidden_size)
        self.Ui = nn.Linear(hidden_size, hidden_size)

        self.Wc = nn.Linear(input_size, hidden_size)
        self.Uc = nn.Linear(hidden_size, hidden_size)

        self.Wo = nn.Linear(input_size, hidden_size)
        self.Uo = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h, c):
        f = torch.sigmoid(self.Wf(x) + self.Uf(h))
        i = torch.sigmoid(self.Wi(x) + self.Ui(h))
        c_tilde = torch.tanh(self.Wc(x) + self.Uc(h))

        c = f * c + i * c_tilde

        o = torch.sigmoid(self.Wo(x) + self.Uo(h))
        h = o * torch.tanh(c)

        return h, c


class BLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.forward_lstm = LSTMCell(hidden_size, hidden_size)
        self.backward_lstm = LSTMCell(hidden_size, hidden_size)

        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        x = self.embedding(x)

        hf = x.new_zeros(batch_size, self.hidden_size)
        cf = x.new_zeros(batch_size, self.hidden_size)

        forward_out = []
        for t in range(seq_len):
            hf, cf = self.forward_lstm(x[:, t, :], hf, cf)
            forward_out.append(hf.unsqueeze(1))

        forward_out = torch.cat(forward_out, dim=1)

        hb = x.new_zeros(batch_size, self.hidden_size)
        cb = x.new_zeros(batch_size, self.hidden_size)

        backward_out = []
        for t in reversed(range(seq_len)):
            hb, cb = self.backward_lstm(x[:, t, :], hb, cb)
            backward_out.insert(0, hb.unsqueeze(1))

        backward_out = torch.cat(backward_out, dim=1)

        out = torch.cat([forward_out, backward_out], dim=2)
        return self.fc(out)


class AttentionRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.Wxh = nn.Linear(hidden_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)

        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        x = self.embedding(x)
        h = x.new_zeros(batch_size, self.hidden_size)

        hidden_states = []
        outputs = []

        for t in range(seq_len):
            xt = x[:, t, :]
            h = torch.tanh(self.Wxh(xt) + self.Whh(h))

            hidden_states.append(h)
            hs = torch.stack(hidden_states, dim=1)

            scores = torch.bmm(h.unsqueeze(1), hs.transpose(1, 2))
            weights = torch.softmax(scores, dim=-1)

            context = torch.bmm(weights, hs).squeeze(1)

            combined = torch.cat([h, context], dim=1)
            out = self.fc(combined)

            outputs.append(out.unsqueeze(1))

        return torch.cat(outputs, dim=1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

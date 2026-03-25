import torch


@torch.no_grad()
def sample(model, stoi, itos, max_len=20, device=None, pad_idx=None):
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    generated_indices = [stoi["<"]]

    for _ in range(max_len):
        input_tensor = torch.tensor([generated_indices], dtype=torch.long, device=device)
        out = model(input_tensor)

        probs = torch.softmax(out[0, -1], dim=0)
        if pad_idx is not None:
            probs[pad_idx] = 0.0
            probs = probs / probs.sum()
        idx = torch.multinomial(probs, 1).item()

        if itos[idx] == ">":
            break

        generated_indices.append(idx)

    return "".join(itos[i] for i in generated_indices[1:])


@torch.no_grad()
def generate_names(model, stoi, itos, n=100, max_len=20, device=None, pad_idx=None):
    return [
        sample(model, stoi, itos, max_len=max_len, device=device, pad_idx=pad_idx)
        for _ in range(n)
    ]

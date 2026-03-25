import torch
import torch.nn as nn


def train(model, dataloader, vocab_size, epochs=10, lr=0.001, device=None, target_pad_idx=-100):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=target_pad_idx)
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out.reshape(-1, vocab_size), y.reshape(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    return epoch_losses

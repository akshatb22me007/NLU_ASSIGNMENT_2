import torch
from pathlib import Path
from data_utils import load_names, build_vocab, NameDataset, build_collate_fn
from models import VanillaRNN, BLSTM, AttentionRNN
from inference import generate_names
from eval import evaluate_generated_names
from train import train
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

train_epochs = [30,25,60]
def build_model(choice, vocab_size, hidden_size=128):
    if choice == 1:
        return VanillaRNN(vocab_size, hidden_size)
    if choice == 2:
        return BLSTM(vocab_size, hidden_size)
    return AttentionRNN(vocab_size, hidden_size)


def model_name_from_choice(choice):
    if choice == 1:
        return "vanilla_rnn"
    if choice == 2:
        return "blstm"
    return "attention_rnn"


def save_generated_names(model_name, generated_names, output_dir):
    output_path = output_dir / f"generated_names_{model_name}.txt"
    with open(output_path, "w") as f:
        for name in generated_names:
            f.write(name + "\n")
    return output_path


def save_model_checkpoint(model, model_name, models_dir):
    checkpoint_path = models_dir / f"{model_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def plot_training_losses(losses_by_model, output_dir):
    if plt is None:
        return None
    plt.figure(figsize=(10, 6))
    for model_name, losses in losses_by_model.items():
        epochs = list(range(1, len(losses) + 1))
        plt.plot(epochs, losses, marker="o", linewidth=2, label=model_name)

    plt.title("Training Loss Curve by Model")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path = output_dir / "training_loss_curves.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_metrics(metrics_by_model, output_dir):
    if plt is None:
        return None
    model_names = list(metrics_by_model.keys())
    novelty_scores = [metrics_by_model[m]["novelty"] for m in model_names]
    diversity_scores = [metrics_by_model[m]["diversity"] for m in model_names]

    x = list(range(len(model_names)))
    width = 0.35

    plt.figure(figsize=(10, 6))
    novelty_positions = [i - width / 2 for i in x]
    diversity_positions = [i + width / 2 for i in x]
    plt.bar(novelty_positions, novelty_scores, width=width, label="Novelty")
    plt.bar(diversity_positions, diversity_scores, width=width, label="Diversity")

    plt.title("Generated Name Quality Metrics")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.xticks(x, model_names, rotation=10)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path = output_dir / "metrics_comparison.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_name_length_distribution(names_by_model, output_dir):
    if plt is None:
        return []
    save_paths = []
    for model_name, generated_names in names_by_model.items():
        lengths = [len(name) for name in generated_names if name]
        if not lengths:
            continue

        plt.figure(figsize=(10, 6))
        bins = range(1, 22)
        plt.hist(lengths, bins=bins, alpha=0.8, color="tab:blue", edgecolor="black")
        plt.title(f"Name Length Distribution - {model_name}")
        plt.xlabel("Name Length")
        plt.ylabel("Count")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_path = output_dir / f"name_length_distribution_{model_name}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        save_paths.append(save_path)

    return save_paths

if __name__ == "__main__":
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    names = load_names("TrainingNames.txt")
    _, stoi, itos = build_vocab(names)
    vocab_size = len(stoi)
    pad_idx = stoi["~"]

    dataset = NameDataset(names, stoi)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=build_collate_fn(input_pad_idx=pad_idx, target_pad_idx=-100),
    )

    losses_by_model = {}
    metrics_by_model = {}
    names_by_model = {}

    for choice in [1, 2, 3]:
        model_name = model_name_from_choice(choice)
        model = build_model(choice, vocab_size=vocab_size, hidden_size=128)

        print(f"Training {model_name}...")
        losses = train(model, loader, vocab_size=vocab_size, epochs=train_epochs[choice-1], target_pad_idx=-100)
        losses_by_model[model_name] = losses

        model_path = save_model_checkpoint(model, model_name, models_dir)
        print(f"Saved trained model to: {model_path}")

        generated = generate_names(model, stoi, itos, n=200, pad_idx=pad_idx)
        names_by_model[model_name] = generated

        names_path = save_generated_names(model_name, generated, output_dir)
        print(f"Saved generated names to: {names_path}")

        metrics = evaluate_generated_names(generated, names)
        metrics_by_model[model_name] = metrics

        print(f"\nSample 50 Names ({model_name}):")
        print(generated[:50])

        print("\nMetrics:")
        print("Novelty:", metrics["novelty"])
        print("Diversity:", metrics["diversity"])

    loss_plot = plot_training_losses(losses_by_model, output_dir)
    metrics_plot = plot_metrics(metrics_by_model, output_dir)
    length_plots = plot_name_length_distribution(names_by_model, output_dir)

    if plt is None:
        print("\nmatplotlib is not installed. Skipping visualization generation.")
    else:
        print("\nSaved visualizations:")
        print(loss_plot)
        print(metrics_plot)
        for p in length_plots:
            print(p)
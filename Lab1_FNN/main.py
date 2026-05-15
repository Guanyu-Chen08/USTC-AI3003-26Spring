import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIZE = 16
EPOCHS = 500
# PATIENCE = float('inf')
PATIENCE = 50
DROPOUT = 0.0
FIGURE_DIR = f"Figures_epochs{EPOCHS}_dropout{DROPOUT}_patience{PATIENCE}_batchsize{SIZE}"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_dataloaders(batch_size=SIZE):
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=RANDOM_SEED
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


class FNNRegression(nn.Module):
    def __init__(self, input_dim=10, hidden_layers=None, activation_fn=nn.ReLU, dropout=DROPOUT):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 32]

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def build_optimizer(model, optimizer_type="adam", learning_rate=1e-3, weight_decay=0.0):
    opt_type = optimizer_type.lower()
    if opt_type == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif opt_type == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError("不支持的优化器类型！请填入 'adam' 或 'sgd'。")


def evaluate_regression_metrics(model, data_loader, device=DEVICE):
    model.eval()
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(X_batch)

            y_true_all.append(y_batch.cpu().numpy())
            y_pred_all.append(pred.cpu().numpy())

    y_true_all = np.vstack(y_true_all).reshape(-1)
    y_pred_all = np.vstack(y_pred_all).reshape(-1)

    mse = mean_squared_error(y_true_all, y_pred_all)
    mae = mean_absolute_error(y_true_all, y_pred_all)
    r2 = r2_score(y_true_all, y_pred_all)
    return {"MSE": mse, "MAE": mae, "R2": r2}


def train_model(
    model, train_loader, val_loader, learning_rate=1e-3, epochs=EPOCHS,
    optimizer_type="adam", weight_decay=0.0, early_stopping_patience=30,
    verbose=False, device=DEVICE
):
    criterion = nn.MSELoss()
    optimizer = build_optimizer(
        model=model, optimizer_type=optimizer_type,
        learning_rate=learning_rate, weight_decay=weight_decay,
    )
    model.to(device)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * X_batch.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                running_val_loss += loss.item() * X_batch.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"  Epoch [{epoch+1:03d}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stopping_patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}, best val loss = {best_val_loss:.4f}")
            break

    model.load_state_dict(best_model_weights)
    return train_losses, val_losses, best_val_loss


def save_curve_plot(curves_dict, title, xlabel, ylabel, filename, figure_dir=FIGURE_DIR, y_limit=None):
    ensure_dir(figure_dir)

    plt.figure(figsize=(10, 6))
    for name, values in curves_dict.items():
        plt.plot(values, label=name)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if y_limit is not None:
        plt.ylim(y_limit)
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(figure_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[图像已保存] {save_path}")


def experiment_depth(train_loader, val_loader, test_loader):
    print("\n[实验一] 探究网络深度的影响")
    depth_configs = {
        "Shallow (1 layer) [64]": [64],
        "Medium (2 layers) [64,32]": [64, 32],
        "Deep (3 layers) [64,32,16]": [64, 32, 16],
    }
    train_curves, val_curves, summary = {}, {}, []
    for name, hidden_layers in depth_configs.items():
        print(f"-> 训练模型: {name}")
        model = FNNRegression(hidden_layers=hidden_layers, activation_fn=nn.ReLU, dropout=DROPOUT)
        train_losses, val_losses, best_val_loss = train_model(
            model, train_loader, val_loader, learning_rate=1e-3, epochs=EPOCHS,
            optimizer_type="adam", early_stopping_patience=PATIENCE, verbose=False
        )
        test_metrics = evaluate_regression_metrics(model, test_loader)
        train_curves[name] = train_losses
        val_curves[name] = val_losses
        summary.append({"name": name, "best_val_loss": best_val_loss, **test_metrics})

    summary = sorted(summary, key=lambda x: x["best_val_loss"])
    save_curve_plot(val_curves, "Experiment 1 - Impact of Network Depth (Validation Loss)", "Epoch", "MSE Loss", "exp01_depth_validation_loss.png")
    save_curve_plot(train_curves, "Experiment 1 - Impact of Network Depth (Training Loss)", "Epoch", "MSE Loss", "exp01_depth_training_loss.png")
    return summary


def experiment_learning_rate(train_loader, val_loader, test_loader):
    print("\n[实验二] 探究学习率的影响")
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    train_curves, val_curves, summary = {}, {}, []
    for lr in learning_rates:
        name = f"LR={lr}"
        print(f"-> 训练模型: {name}")
        model = FNNRegression(hidden_layers=[64, 32], activation_fn=nn.ReLU, dropout=DROPOUT)
        train_losses, val_losses, best_val_loss = train_model(
            model, train_loader, val_loader, learning_rate=lr, epochs=EPOCHS,
            optimizer_type="adam", early_stopping_patience=PATIENCE, verbose=False
        )
        test_metrics = evaluate_regression_metrics(model, test_loader)
        train_curves[name] = train_losses
        val_curves[name] = val_losses
        summary.append({"name": name, "best_val_loss": best_val_loss, **test_metrics})

    summary = sorted(summary, key=lambda x: x["best_val_loss"])
    save_curve_plot(train_curves, "Experiment 2 - Impact of Learning Rate (Training Loss)", "Epoch", "MSE Loss", "exp02_lr_training_loss.png")
    save_curve_plot(val_curves, "Experiment 2 - Impact of Learning Rate (Validation Loss)", "Epoch", "MSE Loss", "exp02_lr_validation_loss.png")
    return summary


def experiment_activation(train_loader, val_loader, test_loader):
    print("\n[实验三] 探究激活函数的影响")
    activation_funcs = {
        "Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh, "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU, "Swish(SiLU)": nn.SiLU,
    }
    train_curves, val_curves, summary = {}, {}, []
    for name, act_fn in activation_funcs.items():
        print(f"-> 训练模型: Activation={name}")
        model = FNNRegression(hidden_layers=[64, 32], activation_fn=act_fn, dropout=DROPOUT)
        train_losses, val_losses, best_val_loss = train_model(
            model, train_loader, val_loader, learning_rate=1e-3, epochs=EPOCHS,
            optimizer_type="adam", early_stopping_patience=PATIENCE, verbose=False
        )
        test_metrics = evaluate_regression_metrics(model, test_loader)
        train_curves[name] = train_losses
        val_curves[name] = val_losses
        summary.append({"name": name, "best_val_loss": best_val_loss, **test_metrics})

    summary = sorted(summary, key=lambda x: x["best_val_loss"])
    save_curve_plot(val_curves, "Experiment 3 - Impact of Activation Function (Validation Loss)", "Epoch", "MSE Loss", "exp03_activation_validation_loss.png")
    
    save_curve_plot(train_curves, "Experiment 3 - Impact of Activation Function (Training Loss)", "Epoch", "MSE Loss", "exp03_activation_training_loss.png")
    return summary


def experiment_optimizer(train_loader, val_loader, test_loader):
    print("\n[实验四] 探究优化器的影响 (Adam vs SGD)")
    optimizers_to_test = ["adam", "sgd"]
    train_curves, val_curves, summary = {}, {}, []
    for opt_name in optimizers_to_test:
        print(f"-> 训练模型: Optimizer={opt_name.upper()}")
        model = FNNRegression(hidden_layers=[64, 32], activation_fn=nn.ReLU, dropout=DROPOUT)
        train_losses, val_losses, best_val_loss = train_model(
            model, train_loader, val_loader, learning_rate=1e-3, epochs=EPOCHS,
            optimizer_type=opt_name, early_stopping_patience=PATIENCE, verbose=False
        )
        test_metrics = evaluate_regression_metrics(model, test_loader)
        curve_label = f"Optimizer: {opt_name.upper()}"
        train_curves[curve_label] = train_losses
        val_curves[curve_label] = val_losses
        summary.append({"name": opt_name.upper(), "best_val_loss": best_val_loss, **test_metrics})

    summary = sorted(summary, key=lambda x: x["best_val_loss"])
    save_curve_plot(train_curves, "Experiment 4 - Impact of Optimizer (Training Loss)", "Epoch", "MSE Loss", "exp04_optimizer_training_loss.png")
    save_curve_plot(val_curves, "Experiment 4 - Impact of Optimizer (Validation Loss)", "Epoch", "MSE Loss", "exp04_optimizer_validation_loss.png")
    return summary

def print_summary_table(title, summary):
    print(f"\n{title}")
    print("-" * 95)
    print(f"{'Rank':<6}{'Config':<35}{'Best Val MSE':<15}{'Test MSE':<13}{'Test MAE':<13}{'Test R2':<10}")
    print("-" * 95)
    for i, row in enumerate(summary, 1):
        print(f"{i:<6}{row['name']:<35}{row['best_val_loss']:<15.4f}{row['MSE']:<13.4f}{row['MAE']:<13.4f}{row['R2']:<10.4f}")
    print("-" * 95)


if __name__ == "__main__":
    print("=" * 60)
    print(" 糖尿病数据集 FNN 回归实验（增强版 - 已标准化）")
    print("=" * 60)
    print(f"运行设备: {DEVICE}")
    print(f"随机种子: {RANDOM_SEED}")

    set_seed(RANDOM_SEED)
    ensure_dir(FIGURE_DIR)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=SIZE)
    print("数据加载与划分完成！(Train/Val/Test = 64%/16%/20%，特征已标准化)")

    depth_summary = experiment_depth(train_loader, val_loader, test_loader)
    lr_summary = experiment_learning_rate(train_loader, val_loader, test_loader)
    act_summary = experiment_activation(train_loader, val_loader, test_loader)
    opt_summary = experiment_optimizer(train_loader, val_loader, test_loader)

    print_summary_table("实验一结果汇总（网络深度）", depth_summary)
    print_summary_table("实验二结果汇总（学习率）", lr_summary)
    print_summary_table("实验三结果汇总（激活函数）", act_summary)
    print_summary_table("实验四结果汇总（优化器对比）", opt_summary)
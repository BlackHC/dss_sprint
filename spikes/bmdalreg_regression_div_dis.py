from itertools import repeat, cycle, chain
from typing import List

import torch
from bmdal_reg.data import ParallelDictDataLoader, DictDataset
from bmdal_reg.models import create_tabular_model
from bmdal_reg.task_execution import get_devices


def fit_model_pool_diversity(model, data, n_models, train_idxs, pool_idxs, valid_idxs, n_epochs=256, batch_size=256,
                             lr=3e-1, weight_decay=0.0,
                             valid_batch_size=8192, **config):
    train_dl = ParallelDictDataLoader(data, train_idxs.expand(n_models, -1), batch_size=batch_size, shuffle=True,
                                      adjust_bs=False, drop_last=True)
    pool_dl = ParallelDictDataLoader(data, pool_idxs.expand(1, -1), batch_size=batch_size, shuffle=True,
                                     adjust_bs=False, drop_last=False)
    valid_dl = ParallelDictDataLoader(data, valid_idxs.expand(n_models, -1), batch_size=valid_batch_size, shuffle=False,
                                      adjust_bs=False, drop_last=False)
    n_steps = n_epochs * len(train_dl)
    best_valid_rmses = [np.Inf] * n_models
    best_model_params = [p.detach().clone() for p in model.parameters()]
    if config.get('opt_name', 'adam') == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        if weight_decay > 0.0:
            opt = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay)
        else:
            opt = torch.optim.Adam(model.parameters())
    step = 0
    for i in range(n_epochs):
        # do one training epoch
        # grad_nonzeros = 0
        model.train()
        lr_sched = config.get('lr_sched', 'lin')
        for train_batch, pool_batch in zip(train_dl, chain.from_iterable(repeat(pool_dl))):
            X_train, y_train = train_batch['X'], train_batch['y']
            y_pred = model(X_train)  # shape: n_models x batch_size x 1

            #### NEW CODE TO ENCOURAGE DIVERSITY ####
            X_pool = pool_batch['X']
            y_pool = model(X_pool)  # shape: n_models x batch_size x 1
            # rows are the variables (batch_size) and columns are the observations (n_models).
            cov = torch.cov(y_pool[..., 0].T)  # shape: batch_size x batch_size
            # compute the entropy of the covariance matrix
            entropy = torch.linalg.slogdet(cov + torch.eye(len(cov)) * 1e-30)[1] / 2 / len(y_pool)
            # try divdiv: substract the entropy of the diagonal
            if False:
                entropy = torch.linalg.slogdet(torch.diag(torch.diag(cov)) + torch.eye(len(cov)) * 1e-30)[
                    1] / 2 / len(y_pool) - entropy
                # Mmm this also works. Why do I get better uncertainty whatever I do?
                entropy *= -1
            # if we have -inf, then we have a singular matrix, so we set the entropy to 0
            if torch.isinf(entropy):
                entropy = torch.tensor(0.0)
                # log print
                print('Singular matrix determ: ', torch.linalg.det(cov))


            # compute the loss
            loss = ((y_train - y_pred) ** 2).mean(dim=-1).mean(dim=-1).sum()  # sum over n_models
            loss = loss - entropy * config.get('entropy_weight', 0.0)
            loss.backward()
            if lr_sched == 'lin':
                current_lr = lr * (1.0 - step / n_steps)
            elif lr_sched == 'hat':
                current_lr = lr * 2 * (0.5 - np.abs(0.5 - step / n_steps))
            elif lr_sched == 'warmup':
                peak_at = 0.1
                current_lr = lr * min((step / n_steps) / peak_at, (1 - step / n_steps) / (1 - peak_at))
            else:
                raise ValueError(f'Unknown lr sched "{lr_sched}"')
            for group in opt.param_groups:
                group['lr'] = current_lr
            opt.step()
            with torch.no_grad():
                for param in model.parameters():
                    # grad_nonzeros += torch.count_nonzero(param.grad).item()
                    param.grad = None

            step += 1

        # do one valid epoch
        valid_sses = torch.zeros(n_models, device=data.device)
        model.eval()
        with torch.no_grad():
            # linear_layers = [module for module in model.modules() if isinstance(module, ParallelLinearLayer)]
            # hooks = [ll.register_forward_hook(
            #     lambda layer, inp, out:
            #     print(f'dead neurons: {(out < 0).all(dim=0).all(dim=0).count_nonzero().item()}'))
            #     for ll in linear_layers]
            for batch in valid_dl:
                X_train, y_train = batch['X'], batch['y']
                valid_sses = valid_sses + ((y_train - model(X_train)) ** 2).mean(dim=-1).sum(dim=-1)
            valid_rmses = torch.sqrt(valid_sses / len(valid_idxs)).detach().cpu().numpy()
            # for hook in hooks:
            #     hook.remove()

        # mean_param_norm = np.mean([p.norm().item() for p in model.parameters()])
        # first_param_mean_abs = list(model.parameters())[0].abs().mean().item()
        # print(f'Epoch {i+1}, Valid RMSEs: {valid_rmses}, first param mean abs: {first_param_mean_abs:g}, '
        #       f'grad nonzeros: {grad_nonzeros}')
        print('.', end='')
        for i in range(n_models):
            if valid_rmses[i] < best_valid_rmses[i]:
                best_valid_rmses[i] = valid_rmses[i]
                for p, best_p in zip(model.parameters(), best_model_params):
                    best_p[i] = p[i]

    print('', flush=True)

    with torch.no_grad():
        for p, best_p in zip(model.parameters(), best_model_params):
            p.set_(best_p)


# %%
import numpy as np


class NNDivDisRegressor:
    """
    Scikit-learn style interface for the NN regression (without active learning) used in this repository.
    """

    def __init__(self, lr: float = 0.15, hidden_sizes: List[int] = None, act: str = 'silu', n_ensemble: int = 1,
                 batch_size: int = 256, n_epochs: int = 256, weight_decay: float = 0.0,
                 weight_gain: float = 0.5, bias_gain: float = 1.0, valid_fraction: float = 0.1, device: str = None,
                 preprocess_data: bool = True, seed: int = 0, entropy_weight=0.):
        """
        Constructor with sensible default values (optimized as in the paper).
        :param lr: Learning rate.
        :param hidden_sizes: Sizes of hidden layers. If None, set to [512, 512]
        :param act: Activation function such as 'relu' or 'silu'. For more options, see layers.get_act_layer().
        :param n_ensemble: How many NNs should be used in the ensemble. Defaults to 1.
        :param batch_size: Batch size to use (will be automatically adjusted to a smaller one if needed).
        :param n_epochs: Number of epochs to train maximally.
        :param weight_decay: Weight decay parameter.
        :param weight_gain: Factor for the weight parameters of linear layers.
        :param bias_gain: Factor for the bias parameters of linear layers.
        :param valid_fraction: Which fraction of the training data set should be used for validation.
        :param device: Device to train on. Should be a string that PyTorch accepts, such as 'cpu' or 'cuda:0'.
        If None, the first GPU is used if one is found, otherwise the CPU is used.
        :param preprocess_data: Whether X and y values should be standardized for training and X should be soft-clipped.
        :param seed: Random seed for training.
        """
        self.n_models = n_ensemble
        self.hidden_sizes = hidden_sizes or [512, 512]
        self.lr = lr
        self.act = act
        self.weight_gain = weight_gain
        self.bias_gain = bias_gain
        self.model = None
        self.valid_fraction = valid_fraction
        self.device = device or get_devices()[0]
        self.seed = seed
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.preprocess_data = preprocess_data
        self.means = None
        self.stds = None
        self.y_mean = None
        self.y_std = None
        self.entropy_weight = entropy_weight

    def fit(self, X: np.ndarray, y: np.ndarray, pool_idxs):
        n_features = X.shape[1]
        if len(y.shape) == 1:
            y = y[:, None]
        n_outputs = y.shape[1]
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.model = create_tabular_model(n_models=self.n_models, n_features=n_features,
                                          hidden_sizes=self.hidden_sizes,
                                          act=self.act, n_outputs=n_outputs,
                                          weight_gain=self.weight_gain, bias_gain=self.bias_gain).to(self.device)

        X = torch.as_tensor(X, dtype=torch.float).to(self.device)
        y = torch.as_tensor(y, dtype=torch.float).to(self.device)

        # Turn indices into a boolean mask
        pool_mask = np.zeros(len(X), dtype=bool)
        pool_mask[pool_idxs] = True
        X_train, y_train = X[~pool_mask], y[~pool_mask]

        # Turn boolean mask into indices
        non_pool_idxs = np.arange(len(X))[~pool_mask]

        if self.preprocess_data:
            self.means = X_train.mean(dim=0, keepdim=True)
            self.stds = X_train.std(dim=0, keepdim=True)
            self.y_mean = y_train.mean().item()
            self.y_std = y_train.std().item()
            X = (X - self.means) / (self.stds + 1e-30)
            X = 5 * torch.tanh(0.2 * X)
            y = (y - self.y_mean) / (self.y_std + 1e-30)

        data = DictDataset({'X': X, 'y': y})
        n_valid = int(self.valid_fraction * X.shape[0])
        perm = np.random.permutation(non_pool_idxs)
        valid_idxs = torch.as_tensor(perm[:n_valid]).to(self.device)
        train_idxs = torch.as_tensor(perm[n_valid:]).to(self.device)

        fit_model_pool_diversity(self.model, data, n_models=self.n_models, train_idxs=train_idxs,
                                 pool_idxs=pool_idxs, valid_idxs=valid_idxs,
                                 n_epochs=self.n_epochs, batch_size=self.batch_size, lr=self.lr,
                                 weight_decay=self.weight_decay, valid_batch_size=8192,
                                 entropy_weight=self.entropy_weight)

    def predict(self, X, reduce: bool = True):
        X = torch.as_tensor(X, dtype=torch.float).to(self.device)

        if self.preprocess_data:
            X = (X - self.means) / (self.stds + 1e-30)
            X = 5 * torch.tanh(0.2 * X)

        data = DictDataset({'X': X})
        idxs = torch.arange(X.shape[0], device=self.device)

        dl = ParallelDictDataLoader(data, idxs.expand(self.n_models, -1), batch_size=8192, shuffle=False,
                                    adjust_bs=False, drop_last=False)
        with torch.no_grad():
            self.model.eval()
            y_pred = torch.cat([self.model(batch['X']) for batch in dl], dim=1)

            if reduce:
                y_pred = y_pred.mean(dim=0)

        if self.preprocess_data:
            y_pred = y_pred * self.y_std + self.y_mean

        return y_pred.detach().cpu().numpy()


# %%


# This follows the example in the documentation
import numpy as np


def f(x):
    return np.exp(0.5 * x - 0.5) + np.sin(1.5 * x)


n_train = 512
np.random.seed(0)

X_train = 2 * np.random.randn(n_train)[:, None]
# Sort X_train
X_train = np.sort(X_train, axis=0)
y_train = f(X_train) + 0.5 * np.random.randn(n_train, 1)

X_test = np.linspace(-6.0, 6.0, 500)[:, None]
y_test = f(X_test)

# %%

pool_idxs = torch.arange(len(X_train))[:128]

prior_strength = 1
entropy_weight = prior_strength / (len(X_train) - len(pool_idxs))

from bmdal_reg.nn_interface import NNRegressor

div_model = NNDivDisRegressor(n_ensemble=3, entropy_weight=entropy_weight, seed=30)
normal_model = NNDivDisRegressor(n_ensemble=3, entropy_weight=0., seed=50)

div_model.fit(X_train, y_train, pool_idxs=pool_idxs)
normal_model.fit(X_train, y_train, pool_idxs=pool_idxs)

div_y_test_pred = div_model.predict(X_test, reduce=False)[..., 0]
normal_y_test_pred = normal_model.predict(X_test, reduce=False)[..., 0]

# %%
import matplotlib.pyplot as plt

# Plot the results. For train we use black dots, for test we use a line.
plt.plot(X_train, y_train, 'k.')
plt.plot(X_train[pool_idxs], y_train[pool_idxs], 'r.')

plt.plot(X_test, y_test, 'g--', label='Ground Truth')

for normal_y_test_pred_member in normal_y_test_pred:
    plt.plot(X_test[:, 0], normal_y_test_pred_member, 'y', linewidth=1, alpha=0.7)

# Compute the mean and stddev of the predictions
normal_y_test_pred_mean = normal_y_test_pred.mean(axis=0)
normal_y_test_pred_stddev = normal_y_test_pred.std(axis=0)
plt.plot(X_test, normal_y_test_pred.mean(axis=0), 'y', linewidth=2)
# Plot stddev as a transparent region
plt.fill_between(X_test[:, 0],
                 normal_y_test_pred_mean - normal_y_test_pred_stddev,
                 normal_y_test_pred_mean + normal_y_test_pred_stddev,
                 alpha=0.7, facecolor='yellow')

for div_y_test_pred_member in div_y_test_pred:
    plt.plot(X_test[:, 0], div_y_test_pred_member, 'b', linewidth=1, alpha=0.7)

# Compute the mean and stddev of the predictions
div_y_test_pred_mean = div_y_test_pred.mean(axis=0)
div_y_test_pred_stddev = div_y_test_pred.std(axis=0)
plt.plot(X_test, div_y_test_pred.mean(axis=0), 'b', linewidth=2)
# Plot stddev as a transparent region
plt.fill_between(X_test[:, 0],
                 div_y_test_pred_mean - div_y_test_pred_stddev,
                 div_y_test_pred_mean + div_y_test_pred_stddev,
                 alpha=0.7, facecolor='blue')

# Create a custom legend
import matplotlib.lines as mlines

black_dot = mlines.Line2D([], [], color='black', marker='.', linestyle='None', markersize=10, label='Train')
red_dot = mlines.Line2D([], [], color='red', marker='.', markersize=15, label='Pool')
blue_line = mlines.Line2D([], [], color='blue', label='Diversity')
yellow_line = mlines.Line2D([], [], color='yellow', label='No Regularization')
green_line = mlines.Line2D([], [], color='green', label='Ground Truth')
plt.legend(handles=[black_dot, red_dot, yellow_line, blue_line, green_line])

plt.show()

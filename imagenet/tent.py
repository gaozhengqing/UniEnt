from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
from sklearn.mixture import GaussianMixture


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, alpha=[0.5], criterion="ent"):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.alpha = alpha
        self.criterion = criterion

        self.model0 = deepcopy(self.model)
        for param in self.model0.parameters():
            param.detach()

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model0, self.model, self.optimizer, self.alpha, self.criterion)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_mean_entropy(x: torch.Tensor) -> torch.Tensor:
    """Mean entropy of softmax distribution from logits."""
    x = x.softmax(1).mean(0)
    return -(x * torch.log(x)).sum()


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model0, model, optimizer, alpha, criterion):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    if criterion != "ent":
        model1 = deepcopy(model0)
        model1.model.head = nn.Identity()
        cos_sim = F.cosine_similarity(model1(x).unsqueeze(1), model.model.head.weight, dim=2)
        max_cos_sim, _ = cos_sim.max(1)
        min_value = max_cos_sim.min()
        max_value = max_cos_sim.max()
        max_cos_sim = (max_cos_sim - min_value) / (max_value - min_value)
        os = 1 - max_cos_sim
        gm = GaussianMixture(n_components=2).fit(os.detach().cpu().numpy().reshape(-1, 1))
        if criterion != "ent_unf":
            filter_ids = gm.predict(os.detach().cpu().numpy().reshape(-1, 1))
            filter_ids = filter_ids if gm.means_[0, 0] < gm.means_[1, 0] else 1 - filter_ids
        else:
            weight = gm.predict_proba(os.detach().cpu().numpy().reshape(-1, 1))
            weight = weight if gm.means_[0, 0] < gm.means_[1, 0] else 1 - weight
    # adapt
    entropys = softmax_entropy(outputs)
    if criterion != "ent":
        if criterion != "ent_unf":
            entropys_ind = entropys[filter_ids == 0]
        else:
            entropys_ind = entropys.mul(torch.from_numpy(weight[:, 0]).to(entropys.device))
        loss = entropys_ind.mean(0)
        if criterion != "ent_ind":
            if criterion != "ent_unf":
                entropys_ood = entropys[filter_ids == 1]
            else:
                entropys_ood = entropys.mul(torch.from_numpy(weight[:, 1]).to(entropys.device))
            loss -= alpha[1] * entropys_ood.mean(0)
    else:
        loss = entropys.mean(0)
    loss -= alpha[0] * softmax_mean_entropy(outputs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        elif isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

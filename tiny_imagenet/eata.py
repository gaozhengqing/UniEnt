"""
Copyright to EATA ICML 2022 Authors, 2022.03.20
Based on Tent ICLR 2021 Spotlight. 
"""

from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from sklearn.mixture import GaussianMixture

import math
import torch.nn.functional as F


class EATA(nn.Module):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, fishers=None, fisher_alpha=2000.0, steps=1, episodic=False, e_margin=math.log(1000)/2-1, d_margin=0.05, alpha=[0.5], criterion="ent"):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "EATA requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.alpha = alpha
        self.criterion = criterion

        self.model0 = deepcopy(self.model)
        for param in self.model0.parameters():
            param.detach()

        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = e_margin # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = d_margin # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5)

        self.current_model_probs = None # the moving average of probability vector (Eqn. 4)

        self.fishers = fishers # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
        self.fisher_alpha = fisher_alpha # trade-off \beta for two losses (Eqn. 8) 

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        if self.steps > 0:
            for _ in range(self.steps):
                outputs, num_counts_2, num_counts_1, updated_probs = forward_and_adapt_eata(x, self.model0, self.model, self.optimizer, self.fishers, self.e_margin, self.current_model_probs, fisher_alpha=self.fisher_alpha, num_samples_update=self.num_samples_update_2, d_margin=self.d_margin, alpha=self.alpha, criterion=self.criterion)
                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                self.reset_model_probs(updated_probs)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


@torch.jit.script
def softmax_mean_entropy(x: torch.Tensor) -> torch.Tensor:
    """Mean entropy of softmax distribution from logits."""
    x = x.softmax(1).mean(0)
    return -(x * torch.log(x)).sum()


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_eata(x, model0, model, optimizer, fishers, e_margin, current_model_probs, fisher_alpha=50.0, d_margin=0.05, scale_factor=2, num_samples_update=0, alpha=[0.5], criterion="ent"):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    Return: 
    1. model outputs; 
    2. the number of reliable and non-redundant samples; 
    3. the number of reliable samples;
    4. the moving average  probability vector over all previous samples
    """
    # forward
    outputs = model(x)
    if criterion != "ent":
        model1 = deepcopy(model0)
        model1.model.fc = nn.Identity()
        cos_sim = F.cosine_similarity(model1(x).unsqueeze(1), model.model.fc.weight, dim=2)
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
            # filter unreliable samples
            filter_ids_1 = torch.where(entropys_ind < e_margin)
            ids1 = filter_ids_1
            ids2 = torch.where(ids1[0]>-0.1)
            entropys_ind = entropys_ind[filter_ids_1] 
            # filter redundant samples
            if current_model_probs is not None: 
                cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), outputs[filter_ids == 0][filter_ids_1].softmax(1), dim=1)
                filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
                entropys_ind = entropys_ind[filter_ids_2]
                ids2 = filter_ids_2
                updated_probs = update_model_probs(current_model_probs, outputs[filter_ids == 0][filter_ids_1][filter_ids_2].softmax(1))
            else:
                updated_probs = update_model_probs(current_model_probs, outputs[filter_ids == 0][filter_ids_1].softmax(1))
            coeff = 1 / (torch.exp(entropys_ind.clone().detach() - e_margin))
            # implementation version 1, compute loss, all samples backward (some unselected are masked)
            entropys_ind = entropys_ind.mul(coeff) # reweight entropy losses for diff. samples
        else:
            entropys_ind = entropys.mul(torch.from_numpy(weight[:, 0]).to(entropys.device))
            # filter unreliable samples
            filter_ids_1 = torch.where(entropys_ind < e_margin)
            ids1 = filter_ids_1
            ids2 = torch.where(ids1[0]>-0.1)
            entropys_ind = entropys_ind[filter_ids_1] 
            # filter redundant samples
            if current_model_probs is not None: 
                cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
                filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
                entropys_ind = entropys_ind[filter_ids_2]
                ids2 = filter_ids_2
                updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
            else:
                updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
            coeff = 1 / (torch.exp(entropys_ind.clone().detach() - e_margin))
            # implementation version 1, compute loss, all samples backward (some unselected are masked)
            entropys_ind = entropys_ind.mul(coeff) # reweight entropy losses for diff. samples
        loss = entropys_ind.mean(0)
        if criterion != "ent_ind":
            if criterion != "ent_unf":
                entropys_ood = entropys[filter_ids == 1]
            else:
                entropys_ood = entropys.mul(torch.from_numpy(weight[:, 1]).to(entropys.device))
            loss -= alpha[1] * entropys_ood.mean(0)
    else:
        # filter unreliable samples
        filter_ids_1 = torch.where(entropys < e_margin)
        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0]>-0.1)
        entropys = entropys[filter_ids_1] 
        # filter redundant samples
        if current_model_probs is not None: 
            cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
            entropys = entropys[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
        coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
        # implementation version 1, compute loss, all samples backward (some unselected are masked)
        entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
        loss = entropys.mean(0)
    """
    # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
    # if x[ids1][ids2].size(0) != 0:
    #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
    """
    if fishers is not None:
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in fishers:
                ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()
        loss += ewc_loss
    loss -= alpha[0] * softmax_mean_entropy(outputs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
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
    """Configure model for use with eata."""
    # train mode, because eata optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what eata updates
    model.requires_grad_(False)
    # configure norm for eata updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with eata."""
    is_training = model.training
    assert is_training, "eata needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "eata needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "eata should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "eata needs normalization for its optimization"
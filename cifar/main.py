import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from iopath.common.file_io import g_pathmgr
from prettytable import PrettyTable
from scipy import interpolate
from sklearn import metrics
from torch.utils.data import DataLoader, SubsetRandomSampler

from robustbench.data import load_cifar10c, load_cifar100c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

import cotta
import eata
import ostta
import norm
import tent
from data import load_svhn, load_svhn_c
from utils import AverageMeter, get_logger, set_random_seed


parser = argparse.ArgumentParser()

# Model options
parser.add_argument("--arch", default="Hendrycks2020AugMix_WRN",
                    choices=["Standard", "Hendrycks2020AugMix_WRN", "Hendrycks2020AugMix_ResNeXt"])
parser.add_argument("--adaptation", default="tent",
                    choices=["source", "norm", "cotta", "tent", "eata", "ostta"])
parser.add_argument("--episodic", action="store_true")
# Corruption options
parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
parser.add_argument("--type", default="gaussian_noise")
parser.add_argument("--severity", default=5, type=int)
parser.add_argument("--num_ex", default=10000, type=int)
# Optimizer options
parser.add_argument("--steps", default=1, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--method", default="Adam", choices=["Adam", "SGD"])
parser.add_argument("--momentum", default=0.9, type=float)
# Testing options
parser.add_argument("--batch_size", default=100, type=int)
# Misc options
parser.add_argument("--rng_seed", default=1, type=int)
parser.add_argument("--save_dir", default="./output")
parser.add_argument("--data_dir", default="./data")
parser.add_argument("--ckpt_dir", default="./ckpt")
parser.add_argument("--log_dest", default="log.txt")
# CoTTA options
parser.add_argument("--mt", default=0.999, type=float)
parser.add_argument("--rst", default=0.01, type=float)
parser.add_argument("--ap", default=0.92, type=float)
# Tent options
parser.add_argument("--alpha", nargs="+", default=[0.5], type=float)
parser.add_argument("--criterion", default="ent", choices=["ent", "ent_ind", "ent_ind_ood", "ent_unf"])
parser.add_argument("--rounds", default=1, type=int)
# EATA options
parser.add_argument("--fisher_size", default=2000, type=int)
parser.add_argument("--fisher_alpha", default=1., type=float)
parser.add_argument("--e_margin", default=math.log(10)*0.40, type=float)
parser.add_argument("--d_margin", default=0.05, type=float)

args = parser.parse_args()

args.type = ["gaussian_noise", "shot_noise", "impulse_noise",
             "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
             "snow", "frost", "fog", "brightness", "contrast",
             "elastic_transform", "pixelate", "jpeg_compression"]
args.severity = [5]
args.log_dest = "{}_{}_lr_{}_alpha_{}_{}.txt".format(
    args.adaptation, args.dataset, args.lr, "_".join(str(alpha) for alpha in args.alpha), args.criterion)
args.ap = 0.92 if args.dataset == "cifar10" else 0.72
args.e_margin = math.log(10)*0.40 if args.dataset == "cifar10" else math.log(100)*0.40
args.d_margin = 0.4 if args.dataset == "cifar10" else 0.1

g_pathmgr.mkdirs(args.save_dir)

set_random_seed(args.rng_seed)

logger = get_logger(__name__, args.save_dir, args.log_dest)
logger.info(f"args:\n{args}")


def evaluate():
    # configure model
    base_model = load_model(args.arch, args.ckpt_dir,
                            args.dataset, ThreatModel.corruptions).cuda()
    if args.adaptation == "source":
        base_model.eval()
        model = base_model
    elif args.adaptation == "norm":
        model = norm.Norm(base_model)
    elif args.adaptation == "cotta":
        base_model = cotta.configure_model(base_model)
        params, param_names = cotta.collect_params(base_model)
        optimizer = setup_optimizer(params)
        model = cotta.CoTTA(base_model, optimizer,
                            steps=args.steps,
                            episodic=args.episodic,
                            mt_alpha=args.mt,
                            rst_m=args.rst,
                            ap=args.ap)
    elif args.adaptation == "tent":
        base_model = tent.configure_model(base_model)
        params, param_names = tent.collect_params(base_model)
        optimizer = setup_optimizer(params)
        model = tent.Tent(base_model, optimizer,
                          steps=args.steps,
                          episodic=args.episodic,
                          alpha=args.alpha,
                          criterion=args.criterion)
    elif args.adaptation == "eata":
        fisher_dataset = eval("datasets." + f"{args.dataset}".upper())(args.data_dir, transform=transforms.ToTensor())
        sampled_indices = torch.randperm(len(fisher_dataset))[:args.fisher_size]
        sampler = SubsetRandomSampler(sampled_indices)
        fisher_loader = DataLoader(fisher_dataset, batch_size=args.batch_size * 2, sampler=sampler)
        base_model = eata.configure_model(base_model)
        params, param_names = eata.collect_params(base_model)
        ewc_optimizer = optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        for iter_, (images, targets) in enumerate(fisher_loader, start=1):
            images, targets = images.cuda(), targets.cuda()
            outputs = base_model(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in base_model.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        del ewc_optimizer
        optimizer = setup_optimizer(params)
        model = eata.EATA(base_model, optimizer, fishers, args.fisher_alpha, e_margin=args.e_margin, d_margin=args.d_margin, alpha=args.alpha, criterion=args.criterion)
    elif args.adaptation == "ostta":
        base_model = ostta.configure_model(base_model)
        params, param_names = ostta.collect_params(base_model)
        optimizer = setup_optimizer(params)
        model = ostta.OSTTA(base_model, optimizer,
                            steps=args.steps,
                            episodic=args.episodic,
                            alpha=args.alpha,
                            criterion=args.criterion)
    # evaluate on each severity and type of corruption in turn
    for i in range(args.rounds):
        t = PrettyTable(["corruption", "acc", "auroc", "fpr95tpr", "oscr"])
        top1 = AverageMeter()
        auroc, fpr95tpr, oscr = AverageMeter(), AverageMeter(), AverageMeter()
        for severity in args.severity:
            for corruption_type in args.type:
                # continual adaptation for all corruption
                logger.info("not resetting model")
                x_ind, y_ind = eval(f"load_{args.dataset}c")(args.num_ex,
                                                             severity, args.data_dir, False,
                                                             [corruption_type])
                x_ood, _ = load_svhn_c(args.num_ex, severity, args.data_dir, False, [corruption_type])
                x_ind, y_ind, x_ood = x_ind.cuda(), y_ind.cuda(), x_ood.cuda()
                acc, (auc, fpr), oscr_ = get_results(model, x_ind, y_ind, x_ood, args.batch_size)
                err = 1. - acc
                logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")
                t.add_row([f"{severity}/{corruption_type}", f"{acc:.2%}", f"{auc:.2%}", f"{fpr:.2%}", f"{oscr_:.2%}"])
                top1.update(acc)
                auroc.update(auc)
                fpr95tpr.update(fpr)
                oscr.update(oscr_)
        t.add_row(["mean", f"{top1.avg:.2%}", f"{auroc.avg:.2%}", f"{fpr95tpr.avg:.2%}", f"{oscr.avg:.2%}"])
        logger.info(f"results of round {i}:\n{t}")


def setup_optimizer(params):
    if args.method == "Adam":
        return optim.Adam(params, lr=args.lr)
    elif args.method == "SGD":
        return optim.SGD(params, args.lr, momentum=args.momentum)
    else:
        raise NotImplementedError


def get_results(model: nn.Module,
                x_ind: torch.Tensor,
                y_ind: torch.Tensor,
                x_ood: torch.Tensor,
                batch_size: int = 100,
                device: torch.device = None):
    if device is None:
        device = x_ind.device
    acc = 0.
    y_true, y_score = torch.zeros((0)), torch.zeros((0))
    score_ind, score_ood, pred = torch.zeros((0)), torch.zeros((0)), torch.zeros((0))
    n_batches = math.ceil(x_ind.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_ind_curr = x_ind[counter * batch_size:(counter + 1) *
                               batch_size].to(device)
            y_ind_curr = y_ind[counter * batch_size:(counter + 1) *
                               batch_size].to(device)
            x_ood_curr = x_ood[counter * batch_size:(counter + 1) *
                               batch_size].to(device)
            x_curr = torch.cat((x_ind_curr, x_ood_curr), dim=0)

            output = model(x_curr)
            max_logit, pred_ = output.max(1)
            energy = output.logsumexp(1)
            prob = output.softmax(1)
            max_prob, pred_ = prob.max(1)

            acc += (pred_[:x_ind_curr.shape[0]] == y_ind_curr).float().sum()

            y_true = torch.cat((y_true, torch.cat((torch.ones(x_ind_curr.shape[0]), torch.zeros(x_ood_curr.shape[0])), dim=0)), dim=0)
            y_score = torch.cat((y_score, energy.cpu()), dim=0)
            score_ind = torch.cat((score_ind, energy[:x_ind_curr.shape[0]].cpu()), dim=0)
            score_ood = torch.cat((score_ood, energy[x_ood_curr.shape[0]:].cpu()), dim=0)
            pred = torch.cat((pred, pred_[:x_ind_curr.shape[0]].cpu()), dim=0)

    return acc.item() / x_ind.shape[0], get_ood_metrics(y_true.numpy(), y_score.numpy()), \
           get_oscr(score_ind.numpy(), score_ood.numpy(), pred.numpy(), y_ind.cpu().numpy())


def get_ood_metrics(y_true, y_score):
    auroc = metrics.roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    return auroc, float(interpolate.interp1d(tpr, fpr)(0.95))


def get_oscr(score_ind, score_ood, pred, y_ind):
    score = np.concatenate((score_ind, score_ood), axis=0)
    def get_fpr(t):
        return (score_ood >= t).sum() / len(score_ood)
    def get_ccr(t):
        return ((score_ind > t) & (pred == y_ind)).sum() / len(score_ind)
    fpr = [0.0]
    ccr = [0.0]
    for s in -np.sort(-score):
        fpr.append(get_fpr(s))
        ccr.append(get_ccr(s))
    fpr.append(1.0)
    ccr.append(1.0)
    roc = sorted(zip(fpr, ccr), reverse=True)
    oscr = 0.0
    for i in range(len(score)):
        oscr += (roc[i][0] - roc[i + 1][0]) * (roc[i][1] + roc[i + 1][1]) / 2.0
    return oscr


if __name__ == "__main__":
    evaluate()

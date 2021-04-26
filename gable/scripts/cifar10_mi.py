import h5py
import time
import random
import datetime
import copy
import numpy as np
import os
import csv
import json
import subprocess
import sys
import PIL.Image as Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from matplotlib import pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from cords.cords.selectionstrategies.supervisedlearning import DataSelectionStrategy
# from cords.cords.utils.models import ResNet18
from distil.distil.utils.models.resnet import ResNet18
from gable.gable.utils.custom_dataset import load_dataset_custom
from torch.utils.data import Subset
from torch.autograd import Variable
import tqdm
from math import floor
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from distil.distil.active_learning_strategies import BADGE, EntropySampling, GLISTER, GradMatchActive, CoreSet, LeastConfidence, MarginSampling 
from distil.distil.utils.DataHandler import DataHandler_CIFAR10, DataHandler_MNIST
seed=42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
from gable.gable.utils.custom_utils import *
# for cuda
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False

def model_eval_loss(data_loader, model, criterion):
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss

def init_weights(m):
#     torch.manual_seed(35)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
                
def create_model(name, num_cls, device):
    if name == 'ResNet18':
        model = ResNet18(num_cls)
    elif name == 'MnistNet':
        model = MnistNet()
    elif name == 'ResNet164':
        model = ResNet164(num_cls)
    model.apply(init_weights)
    model = model.to(device)
    return model

def loss_function():
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    return criterion, criterion_nored

def optimizer_with_scheduler(model, num_epochs, learning_rate, m=0.9, wd=5e-4):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=m, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    return optimizer, scheduler

def optimizer_without_scheduler(model, learning_rate, m=0.9, wd=5e-4):
#     optimizer = optim.Adam(model.parameters(),weight_decay=wd)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=m, weight_decay=wd)
    return optimizer

def generate_cumulative_timing(mod_timing):
    tmp = 0
    mod_cum_timing = np.zeros(len(mod_timing))
    for i in range(len(mod_timing)):
        tmp += mod_timing[i]
        mod_cum_timing[i] = tmp
    return mod_cum_timing/3600

def kernel(x, y, measure="cosine", exp=2):
    if(measure=="eu_sim"):
        dist = pairwise_distances(x.cpu().numpy(), y.cpu().numpy())
        sim = max(dist.ravel()) - dist
#         n = x.size(0)
#         m = y.size(0)
#         d = x.size(1)
#         x = x.unsqueeze(1).expand(n, m, d)
#         y = y.unsqueeze(0).expand(n, m, d)
#         dist = torch.pow(x - y, exp).sum(2)
#         const = torch.max(dist).item()
#         sim = (const - dist)
    
        #dist = torch.exp(-1 * torch.pow(x - y, 2).sum(2))
    if(measure=="cosine"):
        sim = cosine_similarity(x.cpu().numpy(), y.cpu().numpy())
    return sim


def save_kernel_hdf5(lake_kernel, lake_target_kernel, target_kernel=[], lake_private_kernel=[], pp_kernel=[], qp_kernel=[], numpy=True):
    if(not(numpy)):
        lake_kernel = lake_kernel.cpu().numpy()
    with h5py.File("smi_lake_kernel_" + device.split(":")[1] +".hdf5", 'w') as hf:
        hf.create_dataset("kernel",  data=lake_kernel)
    if(not(numpy)):
        lake_target_kernel = lake_target_kernel.cpu().numpy()
    with h5py.File("smi_lake_target_kernel_" + device.split(":")[1] + ".hdf5", 'w') as hf:
        hf.create_dataset("kernel",  data=lake_target_kernel)
    if(not(numpy)):
        target_kernel = target_kernel.cpu().numpy()
    with h5py.File("smi_target_kernel_" + device.split(":")[1] + ".hdf5", 'w') as hf:
        hf.create_dataset("kernel",  data=target_kernel)
    if(not(numpy)):
        lake_private_kernel = lake_private_kernel.cpu().numpy()
    with h5py.File("smi_lake_private_kernel_" + device.split(":")[1] + ".hdf5", 'w') as hf:
        hf.create_dataset("kernel",  data=lake_private_kernel)
    if(not(numpy)):
        pp_kernel = pp_kernel.cpu().numpy()
    with h5py.File("smi_pp_kernel_" + device.split(":")[1] + ".hdf5", 'w') as hf:
        hf.create_dataset("kernel",  data=pp_kernel)
    if(not(numpy)):
        qp_kernel = qp_kernel.cpu().numpy()
    with h5py.File("smi_qp_kernel_" + device.split(":")[1] + ".hdf5", 'w') as hf:
        hf.create_dataset("kernel",  data=qp_kernel)
            
def find_err_per_class(test_set, val_set, final_val_classifications, final_val_predictions, final_tst_classifications, 
                       final_tst_predictions, saveDir, prefix):
    #find queries from the validation set that are erroneous
#     saveDir = os.path.join(saveDir, prefix)
#     if(not(os.path.exists(saveDir))):
#         os.mkdir(saveDir)
    val_err_idx = list(np.where(np.array(final_val_classifications) == False)[0])
    tst_err_idx = list(np.where(np.array(final_tst_classifications) == False)[0])
    val_class_err_idxs = []
    tst_err_log = []
    val_err_log = []
    for i in range(num_cls):
        if(feature=="ood"): tst_class_idxs = list(torch.where(torch.Tensor(test_set.targets.float()) == i)[0].cpu().numpy())
        if(feature=="classimb"): tst_class_idxs = list(torch.where(torch.Tensor(test_set.targets) == i)[0].cpu().numpy())
        val_class_idxs = list(torch.where(torch.Tensor(val_set.targets.float()) == i)[0].cpu().numpy())
        #err classifications per class
        val_err_class_idx = set(val_err_idx).intersection(set(val_class_idxs))
        tst_err_class_idx = set(tst_err_idx).intersection(set(tst_class_idxs))
        if(len(val_class_idxs)>0):
            val_error_perc = round((len(val_err_class_idx)/len(val_class_idxs))*100,2)
        else:
            val_error_perc = 0
        tst_error_perc = round((len(tst_err_class_idx)/len(tst_class_idxs))*100,2)
        print("val, test error% for class ", i, " : ", val_error_perc, tst_error_perc)
        val_class_err_idxs.append(val_err_class_idx)
        tst_err_log.append(tst_error_perc)
        val_err_log.append(val_error_perc)
    tst_err_log.append(sum(tst_err_log)/len(tst_err_log))
    val_err_log.append(sum(val_err_log)/len(val_err_log))
    return tst_err_log, val_err_log, val_class_err_idxs


def aug_train_subset(train_set, lake_set, true_lake_set, subset, lake_subset_idxs, budget, augrandom=False):
    all_lake_idx = list(range(len(lake_set)))
    if(not(len(subset)==budget) and augrandom):
        print("Budget not filled, adding ", str(int(budget) - len(subset)), " randomly.")
        remain_budget = int(budget) - len(subset)
        remain_lake_idx = list(set(all_lake_idx) - set(subset))
        random_subset_idx = list(np.random.choice(np.array(remain_lake_idx), size=int(remain_budget), replace=False))
        subset += random_subset_idx
    lake_ss = custom_subset(true_lake_set, subset, torch.Tensor(true_lake_set.targets.float())[subset])
    if(feature=="ood"): 
        ood_lake_idx = list(set(lake_subset_idxs)-set(subset))
        private_set =  custom_subset(true_lake_set, ood_lake_idx, torch.Tensor(np.array([split_cfg['num_cls_idc']]*len(ood_lake_idx))).float())
    remain_lake_idx = list(set(all_lake_idx) - set(lake_subset_idxs))
    remain_lake_set = custom_subset(lake_set, remain_lake_idx, torch.Tensor(lake_set.targets.float())[remain_lake_idx])
    remain_true_lake_set = custom_subset(true_lake_set, remain_lake_idx, torch.Tensor(true_lake_set.targets.float())[remain_lake_idx])
    print(len(lake_ss),len(remain_lake_set),len(lake_set))
    if(feature!="ood"): assert((len(lake_ss)+len(remain_lake_set))==len(lake_set))
    aug_train_set = torch.utils.data.ConcatDataset([train_set, lake_ss])
    if(feature=="ood"): 
        return aug_train_set, remain_lake_set, remain_true_lake_set, private_set, lake_ss
    else:
        return aug_train_set, remain_lake_set, remain_true_lake_set
                        
def getMisclsSet(val_set, val_class_err_idxs, imb_cls_idx):
    miscls_idx = []
    for i in range(len(val_class_err_idxs)):
        if i in imb_cls_idx:
            miscls_idx += val_class_err_idxs[i]
    print("total misclassified ex from imb classes: ", len(miscls_idx))
    return Subset(val_set, miscls_idx)

def getMisclsSetNumpy(X_val, y_val, val_class_err_idxs, imb_cls_idx):
    miscls_idx = []
    for i in range(len(val_class_err_idxs)):
        if i in imb_cls_idx:
            miscls_idx += val_class_err_idxs[i]
    print("total misclassified ex from imb classes: ", len(miscls_idx))
    return X_val[miscls_idx], y_val[miscls_idx]

def getPrivateSet(lake_set, subset, private_set):
    #augment prev private set and current subset
    new_private_set = custom_subset(lake_set, subset, torch.Tensor(lake_set.targets.float())[subset])
#     new_private_set =  Subset(lake_set, subset)
    total_private_set = torch.utils.data.ConcatDataset([private_set, new_private_set])
    return total_private_set

def getSMI_ss(datkbuildPath, exePath, hdf5Path, budget, numQueries, sf):
    if(sf=="fl1mi"):
        command = os.path.join(datkbuildPath, exePath) + " -mode query -naiveOrRandom naive -magnificationLambda 1 -numSummaries 1 -budget " + str(budget) + " -queryPrivacyOptimizer " + sf + " -numQueries " + numQueries + " -dontComputeKernel true -imageKernelFile " + os.path.join(hdf5Path, "smi_lake_kernel_"+str(device).split(":")[1]+".hdf5") +  " -queryKernelFile " + os.path.join(hdf5Path, "smi_lake_target_kernel_"+str(device).split(":")[1]+".hdf5")
    elif(sf == "logdetmi"):
        command = os.path.join(datkbuildPath, "cifarSubsetSelector_ng") + " -mode query -naiveOrRandom naive -logDetLambda 1 -magnificationLambda 1 -numSummaries 1 -budget " + str(budget) + " -queryPrivacyOptimizer " + sf + " -numQueries  " + numQueries + "  -dontComputeKernel true -imageKernelFile " + os.path.join(hdf5Path, "smi_lake_kernel_"+str(device).split(":")[1]+".hdf5") + " -queryKernelFile " + os.path.join(hdf5Path, "smi_lake_target_kernel_"+str(device).split(":")[1]+".hdf5") + " -queryqueryKernelFile " + os.path.join(hdf5Path, "smi_target_kernel_"+str(device).split(":")[1]+".hdf5")
    elif(sf=="fl2mi"):
        command = os.path.join(datkbuildPath, exePath) + " -mode query -naiveOrRandom naive -queryDiversityLambda 1 -magnificationLambda 1 -numSummaries 1 -budget " + str(budget) + " -queryPrivacyOptimizer " + sf + " -numQueries  " + numQueries + " -dontComputeKernel true -imageKernelFile " + os.path.join(hdf5Path, "smi_lake_kernel_"+str(device).split(":")[1]+".hdf5") + " -queryKernelFile " + os.path.join(hdf5Path, "smi_lake_target_kernel_"+str(device).split(":")[1]+".hdf5")
    elif(sf=="gcmi" or sf=="div-gcmi"):
        command = os.path.join(datkbuildPath, exePath) + " -mode query -naiveOrRandom naive -magnificationLambda 1 -numSummaries 1 -budget " + str(budget) + " -queryPrivacyOptimizer " + sf + " -numQueries " + numQueries + " -dontComputeKernel true -imageKernelFile " + os.path.join(hdf5Path,"smi_lake_kernel_"+str(device).split(":")[1]+".hdf5") + " -queryKernelFile " + os.path.join(hdf5Path,"smi_lake_target_kernel_"+str(device).split(":")[1]+".hdf5")
    elif(sf=="gccg"):
        command = os.path.join(datkbuildPath, exePath) + " -mode private -naiveOrRandom naive -gcLambda 1 -magnificationLambda 1 -numSummaries 1 -budget " + str(budget) + " -queryPrivacyOptimizer " + sf + " -numPrivates " + numQueries + " -dontComputeKernel true -imageKernelFile " + os.path.join(hdf5Path,"smi_lake_kernel_"+str(device).split(":")[1]+".hdf5") + " -privateKernelFile " + os.path.join(hdf5Path,"smi_lake_target_kernel_"+str(device).split(":")[1]+".hdf5")
    elif(sf=="fl1cg"):
        command = os.path.join(datkbuildPath, exePath) + " -mode private -naiveOrRandom naive -magnificationLambda 1 -numSummaries 1 -budget " + str(budget) + " -queryPrivacyOptimizer " + sf + " -numPrivates " + numQueries + " -dontComputeKernel true -imageKernelFile " + os.path.join(hdf5Path,"smi_lake_kernel_"+str(device).split(":")[1]+".hdf5") + " -privateKernelFile " + os.path.join(hdf5Path,"smi_lake_target_kernel_"+str(device).split(":")[1]+".hdf5")
    elif(sf=="logdetcg"):
        command = os.path.join(datkbuildPath, exePath) + " -mode private -naiveOrRandom naive -magnificationLambda 1 -numSummaries 1 -budget " + str(budget) + " -queryPrivacyOptimizer " + sf + " -numPrivates " + numQueries + " -dontComputeKernel true -imageKernelFile " + os.path.join(hdf5Path,"smi_lake_kernel_"+str(device).split(":")[1]+".hdf5") + " -privateKernelFile " + os.path.join(hdf5Path,"smi_lake_target_kernel_"+str(device).split(":")[1]+".hdf5") + " -privateprivateKernelFile " + os.path.join(hdf5Path, "smi_target_kernel_"+str(device).split(":")[1]+".hdf5")
    elif(sf=="fl" or sf=="logdet"):
        command = os.path.join(datkbuildPath, "cifarSubsetSelector_ng") + " -mode generic -naiveOrRandom naive -logDetLambda 1 -numSummaries 1 -budget " + str(budget) + " -genericOptimizer " + sf + " -dontComputeKernel true -imageKernelFile " + os.path.join(hdf5Path,"smi_lake_kernel_"+str(device).split(":")[1]+".hdf5")
    elif(sf =="gc"):
        command = os.path.join(datkbuildPath, exePath) + " -mode generic -naiveOrRandom naive -gcLambda 1 -numSummaries 1 -budget " + str(budget) + " -genericOptimizer " + sf + " -dontComputeKernel true -imageKernelFile " + os.path.join(hdf5Path,"smi_lake_kernel_"+str(device).split(":")[1]+".hdf5")
    elif(sf =="dsum"):
        command = os.path.join(datkbuildPath, exePath) + " -mode generic -naiveOrRandom naive -gcLambda 1 -numSummaries 1 -budget " + str(budget) + " -genericOptimizer " + sf + " -dontComputeKernel true -imageKernelFile " + os.path.join(hdf5Path,"smi_lake_kernel_"+str(device).split(":")[1]+".hdf5")
    print("Executing SIM command: ", command)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=True, shell=True)
    subset = process.communicate()[0]
    subset = subset.decode("utf-8")
    subset = subset.strip().split(" ")
    subset = list(map(int, subset))
    return subset

def getCMI_ss(datkbuildPath, exePath, hdf5Path, budget, numQueries, numPrivates, sf):
    if(sf=="flmic"):
        command = os.path.join(datkbuildPath, exePath) + " -mode joint -naiveOrRandom naive -magnificationLambda 1 -numSummaries 1 -budget " + str(budget) + " -queryPrivacyOptimizer " + sf + " -numQueries " + numQueries + " -numPrivates " + numPrivates + " -dontComputeKernel true -imageKernelFile " + os.path.join(hdf5Path, "smi_lake_kernel_"+str(device).split(":")[1]+".hdf5") +  " -queryKernelFile " + os.path.join(hdf5Path, "smi_lake_target_kernel_"+str(device).split(":")[1]+".hdf5") + " -privateKernelFile " + os.path.join(hdf5Path,"smi_lake_private_kernel_"+str(device).split(":")[1]+".hdf5")
    if(sf == "logdetmic"):
        command = os.path.join(datkbuildPath, exePath) + " -mode joint -naiveOrRandom naive -magnificationLambda 1 -numSummaries 1 -budget " + str(budget) + " -queryPrivacyOptimizer " + sf + " -numQueries " + numQueries + " -numPrivates " + numPrivates + " -dontComputeKernel true -imageKernelFile " + os.path.join(hdf5Path, "smi_lake_kernel_"+str(device).split(":")[1]+".hdf5") +  " -queryKernelFile " + os.path.join(hdf5Path, "smi_lake_target_kernel_"+str(device).split(":")[1]+".hdf5") + " -privateKernelFile " + os.path.join(hdf5Path,"smi_lake_private_kernel_"+str(device).split(":")[1]+".hdf5") + " -queryqueryKernelFile " + os.path.join(hdf5Path, "smi_target_kernel_"+str(device).split(":")[1]+".hdf5") + " -privateprivateKernelFile " + os.path.join(hdf5Path, "smi_pp_kernel_"+str(device).split(":")[1]+".hdf5")  + " -queryprivateKernelFile " +  os.path.join(hdf5Path, "smi_qp_kernel_"+str(device).split(":")[1]+".hdf5")
    print("Executing CMI command: ", command)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=True, shell=True)
    subset = process.communicate()[0]
    subset = subset.decode("utf-8")
    subset = subset.strip().split(" ")
    subset = list(map(int, subset))
    return subset

def remove_ood_points(lake_set, subset, idc_idx):
    idx_subset = []
    subset_cls = torch.Tensor(lake_set.targets.float())[subset]
    for i in idc_idx:
        idc_subset_idx = list(torch.where(subset_cls == i)[0].cpu().numpy())
        idx_subset += list(np.array(subset)[idc_subset_idx])
    print(len(idx_subset),"/",len(subset), " idc points.")
    return idx_subset

def getPerClassSel(lake_set, subset, num_cls):
    perClsSel = []
    subset_cls = torch.Tensor(lake_set.targets.float())[subset]
    for i in range(num_cls):
        cls_subset_idx = list(torch.where(subset_cls == i)[0].cpu().numpy())
        perClsSel.append(len(cls_subset_idx))
    return perClsSel

#check overlap with prev selections
def check_overlap(prev_idx, prev_idx_hist, idx):
    prev_idx = [int(x/num_rep) for x in prev_idx]
    prev_idx_hist = [int(x/num_rep) for x in prev_idx_hist]
    idx = [int(x/num_rep) for x in idx]
    # overlap = set(prev_idx).intersection(set(idx))
    overlap = [value for value in idx if value in prev_idx] 
    # overlap_hist = set(prev_idx_hist).intersection(set(idx))
    overlap_hist = [value for value in idx if value in prev_idx_hist]
    new_points = set(idx) - set(prev_idx_hist)
    total_unique_points = set(idx+prev_idx_hist)
    print("New unique points: ", len(new_points))
    print("Total unique points: ", len(total_unique_points))
    print("overlap % of sel with prev idx: ", len(overlap)/len(idx))
    print("overlap % of sel with all prev idx: ", len(overlap_hist)/len(idx))
    return len(overlap)/len(idx), len(overlap_hist)/len(idx)

def getFeatures(model, dataloader):
    pass

def train_model_al(datkbuildPath, exePath, num_epochs, dataset_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run,
                device, computeErrorLog, strategy="SIM", sf=""):
#     torch.manual_seed(42)
#     np.random.seed(42)
    print(strategy, sf)
    #load the dataset based on type of feature
    if(feature=="classimb" or feature=="ood"):
        if(strategy == "SIM" or strategy == "SF" or strategy=="random"):
            if(strategy == "SF" or strategy=="random"):
                train_set, val_set, test_set, lake_set, sel_cls_idx, num_cls = load_dataset_custom(datadir, dataset_name, feature, split_cfg, False, True)
            else:
                train_set, val_set, test_set, lake_set, sel_cls_idx, num_cls = load_dataset_custom(datadir, dataset_name, feature, split_cfg, False, False)
        elif(strategy=="AL"):
            if(sf=="badge" or sf=="us"):
                X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, sel_cls_idx, num_cls = load_dataset_custom(datadir, dataset_name, feature, split_cfg, True, True)
            else: #dont augment train with valid
                X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, sel_cls_idx, num_cls = load_dataset_custom(datadir, dataset_name, feature, split_cfg, True, False)
        print("selected classes are: ", sel_cls_idx)
    if(feature=="duplicate" or feature=="vanilla"):
        sel_cls_idx = None
        if(strategy == "SIM" or strategy=="random"):
            train_set, val_set, test_set, lake_set, num_cls = load_dataset_custom(datadir, dataset_name, feature, split_cfg)
        elif(strategy=="AL"):
            X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, num_cls = load_dataset_custom(datadir, dataset_name, feature, split_cfg, True)
    if(feature=="ood"): num_cls+=1 #Add one class for OOD class
    N = len(train_set)
    trn_batch_size = 20
    val_batch_size = 10
    tst_batch_size = 100

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=trn_batch_size,
                                              shuffle=True, pin_memory=True)

    valloader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size, 
                                            shuffle=False, pin_memory=True)

    tstloader = torch.utils.data.DataLoader(test_set, batch_size=tst_batch_size,
                                             shuffle=False, pin_memory=True)
    
    lakeloader = torch.utils.data.DataLoader(lake_set, batch_size=tst_batch_size,
                                         shuffle=False, pin_memory=True)
    true_lake_set = copy.deepcopy(lake_set)
    # Budget for subset selection
    bud = budget
   
    # Variables to store accuracies
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    tst_losses = np.zeros(num_epochs)
    timing = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    full_trn_acc = np.zeros(num_epochs)
    tst_acc = np.zeros(num_epochs)
    final_tst_predictions = []
    final_tst_classifications = []
    best_val_acc = -1
    csvlog = []
    val_csvlog = []
    # Results logging file
    print_every = 3
    all_logs_dir = 'SMI_active_learning_results/' + dataset_name  + '/' + feature + '/'+  sf + '/' + str(bud) + '/' + str(run)
    print("Saving results to: ", all_logs_dir)
    subprocess.run(["mkdir", "-p", all_logs_dir])
    exp_name = dataset_name + "_" + feature +  "_" + strategy + "_" + str(len(sel_cls_idx))  +"_" + sf +  '_budget:' + str(bud) + '_epochs:' + str(num_epochs) + '_linear:'  + str(linearLayer) + '_runs' + str(run)
    print(exp_name)
    res_dict = {"dataset":data_name, "feature":feature, "sel_func":sf, "sel_budget":budget, "num_selections":num_epochs, "model":model_name, "learning_rate":learning_rate, "setting":split_cfg, "all_class_acc":None, "test_acc":[],"sel_per_cls":[], "sel_cls_idx":sel_cls_idx.tolist()}
    # Model Creation
    model = create_model(model_name, num_cls, device)
    model1 = create_model(model_name, num_cls, device)
    if(strategy == "AL"):
        strategy_args = {'batch_size' : budget, 'lr':float(0.001), 'device':device}
        if(sf=="badge"):
            strategy_sel = BADGE(X_tr, y_tr, X_unlabeled, model, handler, num_cls, strategy_args)
        elif(sf=="us"):
            strategy_sel = EntropySampling(X_tr, y_tr, X_unlabeled, model, handler, num_cls, strategy_args)
        elif(sf=="glister" or sf=="glister-tss"):
            strategy_sel = GLISTER(X_tr, y_tr, X_unlabeled, model, handler, num_cls, strategy_args, valid=True, X_val=X_val, Y_val=y_val, typeOf='rand', lam=0.1)
        elif(sf=="gradmatch-tss"):
            strategy_args = {'batch_size' : 1, 'lr':float(0.01)}
            strategy_sel = GradMatchActive(X_tr, y_tr, X_unlabeled, model, F.cross_entropy, handler, num_cls, strategy_args["lr"], "PerBatch", False, strategy_args, valid=True, X_val=X_val, Y_val=y_val)
        elif(sf=="coreset"):
            strategy_sel = CoreSet(X_tr, y_tr, X_unlabeled, model, handler, num_cls, strategy_args)
        elif(sf=="leastconf"):
            strategy_sel = LeastConfidence(X_tr, y_tr, X_unlabeled, model, handler, num_cls, strategy_args)
        elif(sf=="margin"):
            strategy_sel = MarginSampling(X_tr, y_tr, X_unlabeled, model, handler, num_cls, strategy_args)
    # Loss Functions
    criterion, criterion_nored = loss_function()

    # Getting the optimizer and scheduler
#     optimizer, scheduler = optimizer_with_scheduler(model, num_epochs, learning_rate)
    optimizer = optimizer_without_scheduler(model, learning_rate)
    private_set = []
    #overlap vars
    prev_idx = None
    prev_idx_hist = []
    sel_hist = []
    per_ep_overlap = []
    overall_overlap = []
    idx_tracker = np.array(list(range(len(lake_set))))
    #kernels
    train_private_kernel = []
    train_val_kernel = []
    pp_kernel = []
    qp_kernel = []
    val_kernel = []
    
    for i in range(num_epochs):
        print("AL epoch: ", i)
        tst_loss = 0
        tst_correct = 0
        tst_total = 0
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        if(i==0):
            print("initial training epoch")
            if(os.path.exists(initModelPath)):
                model.load_state_dict(torch.load(initModelPath, map_location=device))
                print("Init model loaded from disk, skipping init training: ", initModelPath)
                with torch.no_grad():
                    final_val_predictions = []
                    final_val_classifications = []
                    for batch_idx, (inputs, targets) in enumerate(valloader):
                        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        if(feature=="ood"): 
                            _, predicted = outputs[...,:-1].max(1)
                        else:
                            _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
                        final_val_predictions += list(predicted.cpu().numpy())
                        final_val_classifications += list(predicted.eq(targets).cpu().numpy())
  
                    if((val_correct/val_total) > best_val_acc):
                        final_tst_predictions = []
                        final_tst_classifications = []
                    for batch_idx, (inputs, targets) in enumerate(tstloader):
                        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        tst_loss += loss.item()
                        if(feature=="ood"): 
                            _, predicted = outputs[...,:-1].max(1)
                        else:
                            _, predicted = outputs.max(1)
                        tst_total += targets.size(0)
                        tst_correct += predicted.eq(targets).sum().item()
                        if((val_correct/val_total) > best_val_acc):
                            final_tst_predictions += list(predicted.cpu().numpy())
                            final_tst_classifications += list(predicted.eq(targets).cpu().numpy())                
                    if((val_correct/val_total) > best_val_acc):
                        best_val_acc = (val_correct/val_total)
                    val_acc[i] = val_correct / val_total
                    tst_acc[i] = tst_correct / tst_total
                    val_losses[i] = val_loss
                    tst_losses[i] = tst_loss
                    res_dict["test_acc"].append(tst_acc[i])
                continue
        else:
#             if(full_trn_acc[i-1] >= 0.99): #The model has already trained on the seed dataset
            #use misclassifications on validation set as queries
            #compute hypothesized labels
            hyp_lake_labels = []
            model.eval()
            with torch.no_grad():
                for batch_idx, (inputs, _) in enumerate(lakeloader):
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    hyp_lake_labels += list(predicted)
            if(feature=="ood"): print(len(hyp_lake_labels), hyp_lake_labels.count(split_cfg['num_cls_idc']))
            lake_set = custom_subset(lake_set, list(range(len(hyp_lake_labels))), torch.Tensor(hyp_lake_labels))
            lakeloader = torch.utils.data.DataLoader(lake_set, batch_size=tst_batch_size, shuffle=False, pin_memory=True)
#             sys.exit()
            #compute the error log before every selection
            if(computeErrorLog):
                tst_err_log, val_err_log, val_class_err_idxs = find_err_per_class(test_set, val_set, final_val_classifications, final_val_predictions, final_tst_classifications, final_tst_predictions, all_logs_dir, sf+"_"+str(bud))
                csvlog.append(tst_err_log)
                val_csvlog.append(val_err_log)
            ####SIM####
            if(strategy=="SIM" or strategy=="SF"):
                if(sf.endswith("mi")):
                    if(feature=="classimb"):
                        #make a dataloader for the misclassifications - only for experiments with targets
                        miscls_set = getMisclsSet(val_set, val_class_err_idxs, sel_cls_idx)
                        misclsloader = torch.utils.data.DataLoader(miscls_set, batch_size=1, shuffle=False, pin_memory=True)
                        setf_model = DataSelectionStrategy(lakeloader, misclsloader, model1, num_cls, linearLayer, device) #set last arg to true for linear layer
                    else:
                        setf_model = DataSelectionStrategy(lakeloader, valloader, model1, num_cls, linearLayer, device)
                elif(sf.endswith("cg")): #atleast one selection must be done for private set in cond gain functions
                    if(len(private_set)!=0):
                        privateSetloader = torch.utils.data.DataLoader(private_set, batch_size=1, shuffle=False, pin_memory=True)
                        setf_model = DataSelectionStrategy(lakeloader, privateSetloader, model1, num_cls, linearLayer, device) #set last arg to true for linear layer
                    else:
                        #compute subset with private set a NULL
                        setf_model = DataSelectionStrategy(lakeloader, valloader, model1, num_cls, linearLayer, device)
                elif(sf.endswith("mic")): #configured for the OOD setting
                    print("val set targets: ", val_set.targets)
                    setf_model = DataSelectionStrategy(lakeloader, valloader, model1, num_cls, linearLayer, device) #In-dist samples are in Val
                    if(len(private_set)!=0):
                        print("private set targets: ", private_set.targets)
                        privateSetloader = torch.utils.data.DataLoader(private_set, batch_size=1, shuffle=False, pin_memory=True)
                        setf_model_private = DataSelectionStrategy(privateSetloader, privateSetloader, model1, num_cls, linearLayer, device)
                else:
                    setf_model = DataSelectionStrategy(lakeloader, valloader, model1, num_cls, linearLayer, device)
                start_time = time.time()
                cached_state_dict = copy.deepcopy(model.state_dict())
                clone_dict = copy.deepcopy(model.state_dict())
                #update the selection strategy model with new params for gradient computation
                setf_model.update_model(clone_dict)
                if(sf.endswith("mic") and len(private_set)!=0): setf_model_private.update_model(clone_dict)
                if(sf.endswith("mi")): #SMI functions need the target set gradients
                    setf_model.compute_gradients(valid=True, batch=False, perClass=False)
                    print("train minibatch gradients shape ", setf_model.grads_per_elem.shape)
#                     print(setf_model.grads_per_elem)
                    print("val minibatch gradients shape ", setf_model.val_grads_per_elem.shape)
#                     print(setf_model.val_grads_per_elem)
                    if(doublePrecision):
                        train_val_kernel = kernel(setf_model.grads_per_elem.double(), setf_model.val_grads_per_elem.double())#img_query_kernel
                    else:
                        train_val_kernel = kernel(setf_model.grads_per_elem, setf_model.val_grads_per_elem)#img_query_kernel
                    numQueryPrivate = train_val_kernel.shape[1]
                elif(sf.endswith("cg")):
                    if(len(private_set)!=0):
                        setf_model.compute_gradients(valid=True, batch=False, perClass=False)
                        print("train minibatch gradients shape ", setf_model.grads_per_elem.shape)
                        print("val minibatch gradients shape ", setf_model.val_grads_per_elem.shape)
                        if(doublePrecision):
                            train_val_kernel = kernel(setf_model.grads_per_elem.double(), setf_model.val_grads_per_elem.double())#img_query_kernel
                        else:
                            train_val_kernel = kernel(setf_model.grads_per_elem, setf_model.val_grads_per_elem)#img_query_kernel
                        numQueryPrivate = train_val_kernel.shape[1]
                    else:
#                         assert(((i + 1)/select_every)==1)
                        setf_model.compute_gradients(valid=False, batch=False, perClass=False)
                        train_val_kernel = []
                        numQueryPrivate = 0
                elif(sf.endswith("mic")):
                    setf_model.compute_gradients(valid=True, batch=False, perClass=False)
                    print("train minibatch gradients shape ", setf_model.grads_per_elem.shape)
                    print("val minibatch gradients shape ", setf_model.val_grads_per_elem.shape)
                    if(doublePrecision):
                        train_val_kernel = kernel(setf_model.grads_per_elem.double(), setf_model.val_grads_per_elem.double())#img_query_kernel
                    else:
                        train_val_kernel = kernel(setf_model.grads_per_elem, setf_model.val_grads_per_elem)#img_query_kernel
                    numQuery = train_val_kernel.shape[1]
                    if(len(private_set)!=0): 
                        setf_model_private.compute_gradients(valid=False, batch=False, perClass=False)
                        print("private gradients shape: ", setf_model_private.grads_per_elem.shape)
                        if(doublePrecision):
                            train_private_kernel = kernel(setf_model.grads_per_elem.double(), setf_model_private.grads_per_elem.double()) #img_private_kernel
                        else:
                            train_private_kernel = kernel(setf_model.grads_per_elem, setf_model_private.grads_per_elem) #img_private_kernel
                        numPrivate = train_private_kernel.shape[1]
                    else:
                        train_private_kernel = []
                        numPrivate = 0
                else: # For other submodular functions needing only image kernel
                    setf_model.compute_gradients(valid=False, batch=False, perClass=False)
                    numQueryPrivate = 0

                kernel_time = time.time()
                if(doublePrecision):
                    train_kernel = kernel(setf_model.grads_per_elem.double(), setf_model.grads_per_elem.double()) #img_img_kernel
                else:
                    train_kernel = kernel(setf_model.grads_per_elem, setf_model.grads_per_elem) #img_img_kernel
                if(sf=="logdetmi" or sf=="logdetcg" or sf=="logdetmic"):
                    if(sf=="logdetcg"):
                        if(len(private_set)!=0):
                            val_kernel = kernel(setf_model.val_grads_per_elem, setf_model.val_grads_per_elem)#private_private_kernel
                        else:
                            val_kernel = []
                    if(sf=="logdetmi"):
                        val_kernel = kernel(setf_model.val_grads_per_elem, setf_model.val_grads_per_elem)#query_query_kernel
                    if(sf=="logdetmic"):
                        val_kernel = kernel(setf_model.val_grads_per_elem, setf_model.val_grads_per_elem)#query_query_kernel
                        if(len(private_set)!=0):
                            pp_kernel = kernel(setf_model_private.grads_per_elem, setf_model_private.grads_per_elem)#private_private_kernel
                            qp_kernel = kernel(setf_model.val_grads_per_elem, setf_model_private.grads_per_elem)#query_private_kernel
                        else:
                            pp_kernel = []
                            qp_kernel = []
                save_kernel_hdf5(train_kernel, train_val_kernel, val_kernel, train_private_kernel, pp_kernel, qp_kernel)
#                 else:
#                     save_kernel_hdf5(train_kernel, train_val_kernel)
                print("kernel compute time: ", time.time()-kernel_time)
                #call the c++ exec to read kernel and compute subset of selected minibatches
                if(sf.endswith("mic")):
                    subset = getCMI_ss(datkbuildPath, exePath, os.getcwd(), budget, str(numQuery), str(numPrivate), sf)
                else:
                    subset = getSMI_ss(datkbuildPath, exePath, os.getcwd(), budget, str(numQueryPrivate), sf)
                print("True targets of subset: ", torch.Tensor(true_lake_set.targets.float())[subset])
                print("Hypothesized targets of subset: ", torch.Tensor(lake_set.targets.float())[subset])
                model.load_state_dict(cached_state_dict)
                if(sf.endswith("cg")): #for first selection
                    if(len(private_set)==0):
                        private_set = custom_subset(lake_set, subset, torch.Tensor(lake_set.targets.float())[subset])
                    else:
                        private_set = getPrivateSet(lake_set, subset, private_set)
                    print("size of private set: ", len(private_set))

    #           temp = np.array(list(trainloader.batch_sampler))[subset] #if per batch
            ###AL###
            elif(strategy=="AL"):
                if(sf=="glister-tss" or sf=="gradmatch-tss"):
                    miscls_X_val, miscls_y_val = getMisclsSetNumpy(X_val, y_val, val_class_err_idxs, sel_cls_idx)
                    if(sf=="glister-tss"): strategy_sel = GLISTER(X_tr, y_tr, X_unlabeled, model, handler, num_cls, device, strategy_args, valid=True, X_val=miscls_X_val, Y_val=miscls_y_val, typeOf='rand', lam=0.1)
                    if(sf=="gradmatch-tss"): strategy_sel = GradMatchActive(X_tr, y_tr, X_unlabeled, model, F.cross_entropy, handler, num_cls, strategy_args["lr"], "PerBatch", False, strategy_args, valid=True, X_val=miscls_X_val, Y_val=miscls_y_val)
                    print("reinit AL with targeted miscls samples")
                strategy_sel.update_model(model)
                if(sf=="badge" or sf=="glister" or sf=="glister-tss" or sf=="coreset" or sf=="margin"):
                    subset = strategy_sel.select(budget)
                if(sf=="us" or sf=="leastconf"):
                    subset = list(strategy_sel.select(budget).cpu().numpy())
                if(sf=="gradmatch-tss"):
                    subset = strategy_sel.select(budget, False) #Fixed weight gradmatch
                print(len(subset), " samples selected")
                
            elif(strategy=="random"):
                subset = np.random.choice(np.array(list(range(len(lake_set)))), size=budget, replace=False)
            if(i>1 and sf.endswith("cg")):
                per_ep, overall = check_overlap(prev_idx, prev_idx_hist, list(idx_tracker[subset]))
                per_ep_overlap.append(per_ep)
                overall_overlap.append(overall)
            lake_subset_idxs = subset #indices wrt to lake that need to be removed from the lake
            if(feature=="ood"): #remove ood points from the subset
                subset = remove_ood_points(true_lake_set, subset, sel_cls_idx)
            
            if(strategy=="AL"):#augment train and remove from lake for AL startegies
                X_tr = np.concatenate((X_tr, X_unlabeled[subset]), axis=0)
                X_unlabeled = np.delete(X_unlabeled, lake_subset_idxs, axis = 0)
                y_tr = np.concatenate((y_tr, y_unlabeled[subset]), axis = 0)
                y_unlabeled = np.delete(y_unlabeled, lake_subset_idxs, axis = 0)
                strategy_sel.update_data(X_tr, y_tr, X_unlabeled)
                print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
            perClsSel = getPerClassSel(true_lake_set, lake_subset_idxs, num_cls)
            res_dict['sel_per_cls'].append(perClsSel)
            prev_idx = list(idx_tracker[lake_subset_idxs])
            prev_idx_hist += list(idx_tracker[lake_subset_idxs])
            sel_hist.append(list(idx_tracker[lake_subset_idxs]))
            idx_tracker = np.delete(idx_tracker, lake_subset_idxs, axis=0)
            #augment the train_set with selected indices from the lake
            if(feature=="classimb"):
                train_set, lake_set, true_lake_set = aug_train_subset(train_set, lake_set, true_lake_set, subset, lake_subset_idxs, budget, True) #aug train with random if budget is not filled
            elif(feature=="ood"):
                train_set, lake_set, true_lake_set, new_private_set, add_val_set = aug_train_subset(train_set, lake_set, true_lake_set, subset, lake_subset_idxs, budget)
                train_set = torch.utils.data.ConcatDataset([train_set, new_private_set]) #Add the OOD samples with a common OOD class
                val_set = custom_concat(val_set, add_val_set)
                if(len(private_set)!=0):
                    private_set = custom_concat(private_set, new_private_set)
                else:
                    private_set = new_private_set
            else:
                train_set, lake_set, true_lake_set = aug_train_subset(train_set, lake_set, true_lake_set, subset, lake_subset_idxs, budget)
            print("After augmentation, size of train_set: ", len(train_set), " lake set: ", len(lake_set))
#           Reinit train and lake loaders with new splits and reinit the model
            trainloader = torch.utils.data.DataLoader(train_set, batch_size=trn_batch_size, shuffle=True, pin_memory=True)
            lakeloader = torch.utils.data.DataLoader(lake_set, batch_size=tst_batch_size, shuffle=False, pin_memory=True)
            valloader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size, shuffle=False, pin_memory=True)
            assert(len(idx_tracker)==len(lake_set))
#             model =  model.apply(weight_reset).cuda()
            model = create_model(model_name, num_cls, device)
            optimizer = optimizer_without_scheduler(model, learning_rate)
                
        #Start training
        start_time = time.time()
        num_ep=1
        while(full_trn_acc[i]<0.99 and num_ep<150):
            model.train()
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                # Variables in Pytorch are differentiable.
                inputs, target = Variable(inputs), Variable(inputs)
                # This will zero out the gradients for this batch.
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
#             scheduler.step()
          
            full_trn_loss = 0
            full_trn_correct = 0
            full_trn_total = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    full_trn_loss += loss.item()
                    _, predicted = outputs.max(1)
                    full_trn_total += targets.size(0)
                    full_trn_correct += predicted.eq(targets).sum().item()
                full_trn_acc[i] = full_trn_correct / full_trn_total
                print("Selection Epoch ", i, " Training epoch [" , num_ep, "]" , " Training Acc: ", full_trn_acc[i], end="\r")
                num_ep+=1
            timing[i] = time.time() - start_time
        with torch.no_grad():
            final_val_predictions = []
            final_val_classifications = []
            for batch_idx, (inputs, targets) in enumerate(valloader):
                # print(batch_idx)
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                if(feature=="ood"): 
                    _, predicted = outputs[...,:-1].max(1)
                else:
                    _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
    #                 if(i == (num_epochs-1)):
                final_val_predictions += list(predicted.cpu().numpy())
                final_val_classifications += list(predicted.eq(targets).cpu().numpy())
                # sys.exit()

#             if((val_correct/val_total) > best_val_acc):
            final_tst_predictions = []
            final_tst_classifications = []
            for batch_idx, (inputs, targets) in enumerate(tstloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                tst_loss += loss.item()
                if(feature=="ood"): 
                    _, predicted = outputs[...,:-1].max(1)
                else:
                    _, predicted = outputs.max(1)
                tst_total += targets.size(0)
                tst_correct += predicted.eq(targets).sum().item()
#                 if((val_correct/val_total) > best_val_acc):
    #                 if(i == (num_epochs-1)):
                final_tst_predictions += list(predicted.cpu().numpy())
                final_tst_classifications += list(predicted.eq(targets).cpu().numpy())                
#             if((val_correct/val_total) > best_val_acc):
#                 best_val_acc = (val_correct/val_total)
            val_acc[i] = val_correct / val_total
            tst_acc[i] = tst_correct / tst_total
            val_losses[i] = val_loss
            fulltrn_losses[i] = full_trn_loss
            tst_losses[i] = tst_loss
            full_val_acc = list(np.array(val_acc))
            full_timing = list(np.array(timing))
            res_dict["test_acc"].append(tst_acc[i])
            print('Epoch:', i + 1, 'FullTrn,TrainAcc,ValLoss,ValAcc,TstLoss,TstAcc,Time:', full_trn_loss, full_trn_acc[i], val_loss, val_acc[i], tst_loss, tst_acc[i], timing[i])
        if(i==0): 
            print("saving initial model") 
            torch.save(model.state_dict(), initModelPath) #save initial train model if not present
    if(computeErrorLog):
        tst_err_log, val_err_log, val_class_err_idxs = find_err_per_class(test_set, val_set, final_val_classifications, final_val_predictions, final_tst_classifications, final_tst_predictions, all_logs_dir, sf+"_"+str(bud))
        csvlog.append(tst_err_log)
        val_csvlog.append(val_err_log)
        print(csvlog)
        res_dict["all_class_acc"] = csvlog
        res_dict["all_val_class_acc"] = val_csvlog
#         with open(os.path.join(all_logs_dir, exp_name+".csv"), "w") as f:
#             writer = csv.writer(f)
#             writer.writerows(csvlog)
    #save results dir with test acc and per class selections
    with open(os.path.join(all_logs_dir, exp_name+".json"), 'w') as fp:
        json.dump(res_dict, fp)

feature = sys.argv[1]
device_id = sys.argv[2]
run=sys.argv[3]
datadir = 'data/'
data_name = 'cifar10'
model_name = 'ResNet18'
num_rep = 10
learning_rate = 0.01
num_runs = 1  # number of random runs
computeClassErrorLog = True

magnification = 1
device = "cuda:"+str(device_id) if torch.cuda.is_available() else "cpu"
datkbuildPath = "/home/snk170001/bioml/dss/notebooks/datk/build"
exePath = "cifarSubsetSelector"
print("Using Device:", device)
doublePrecision = True
linearLayer = True
handler = DataHandler_CIFAR10

if(feature=="classimb"):
    num_cls = 10
    budget = 30
    num_epochs = int(20)
    split_cfg = {"num_cls_imbalance":2, "per_imbclass_train":10, "per_imbclass_val":5, "per_imbclass_lake":150, "per_class_train":200, "per_class_val":5, "per_class_lake":3000} #cifar10
    # split_cfg = {"num_cls_imbalance":2, "per_imbclass_train":10, "per_imbclass_val":5, "per_imbclass_lake":75, "per_class_train":200, "per_class_val":5, "per_class_lake":295} #cifar100
    initModelPath = data_name + "_" + model_name + "_" + str(learning_rate) + "_" + str(split_cfg["per_imbclass_train"]) + "_" + str(split_cfg["per_class_train"]) + "_" + str(split_cfg["num_cls_imbalance"])

if(feature=="ood"):
    num_cls=8
    budget=250
    num_epochs = int(10)
    split_cfg = {'num_cls_idc':8, 'per_idc_train':200, 'per_idc_val':10, 'per_idc_lake':500, 'per_ood_train':0, 'per_ood_val':0, 'per_ood_lake':5000}#cifar10
    # split_cfg = {'num_cls_idc':50, 'per_idc_train':100, 'per_idc_val':2, 'per_idc_lake':100, 'per_ood_train':0, 'per_ood_val':0, 'per_ood_lake':500}#cifar100
    initModelPath = "weights/" + data_name + "_" + feature + "_" + model_name + "_" + str(learning_rate) + "_" + str(split_cfg["per_idc_train"]) + "_" + str(split_cfg["per_idc_val"]) + "_" + str(split_cfg["num_cls_idc"])

# feature='vanilla'
# split_cfg = {"train_size":500, "val_size":1000, "lake_size":5000, "num_rep":num_rep, "lake_subset_repeat_size":1000}
# feature = 'duplicate'

if(feature=="ood"):
    #FLCMI
    train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "SIM",'flmic')

    #LOGDETCMI
    train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "SIM",'logdetmic')

#FL2MI
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "SIM",'fl2mi')

#FL1MI
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "SIM",'fl1mi')

#GCMI-DIV
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "SIM",'div-gcmi')

#GCMI
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "SIM",'gcmi')

#LOGDETMI
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "SIM",'logdetmi')

#BADGE
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "AL","badge")

#US
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "AL","us")

#GLISTER
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "AL","glister-tss")

#GradMatch-Active
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "AL","gradmatch-tss")

#CORESET
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "AL","coreset")

#LEASTCONF
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "AL","leastconf")

#MARGIN SAMPLING
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "AL","margin")

#FL
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "SF",'fl')

#LOGDET
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "SF",'logdet')

#RANDOM
train_model_al(datkbuildPath, exePath, num_epochs, data_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run, device, computeClassErrorLog, "random",'random')



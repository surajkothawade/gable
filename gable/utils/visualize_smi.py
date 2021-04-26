import matplotlib.pyplot as plt
import json
import os
import sys
import numpy as np
import itertools

expDir = sys.argv[1]
feature = sys.argv[2]
run = sys.argv[3]
budget = sys.argv[4]
epochs = sys.argv[5]
num_sel_cls = sys.argv[6]

overall_test_acc_dict = {}
target_test_acc_dict = {}
num_target_samples_dict = {}
plot_md = {}
exp_name = ""

markers = ['o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D']
colors = ['c-', 'p-', 'm-', 'd-', 'b-', 'h-', 'x-', 'v-', 'o-', 's-', 'y-', 'g-', 'r-']
mc = list(itertools.product(markers, colors))

def smoothen_target_acc(test_acc):
    best_ep_target_acc = 0
    for i in range(len(test_acc)):
        if(best_ep_target_acc < test_acc[i]):
            best_ep_target_acc = test_acc[i]
        else:
            test_acc[i] = best_ep_target_acc
    return test_acc
def overall_test_acc_plot(overall_test_acc_dict, plot_md, exp_name):
    x_axis = np.array([plot_md["train_size"]+plot_md["sel_budget"]*i for i in range(plot_md["num_selections"])])
    plt.figure()
    badge_test_acc_p = [x for x in overall_test_acc_dict["badge"]]
    # fl_test_acc_p = [x for x in overall_test_acc_dict["fl"]]
    # fl1mi_test_acc_p = [x for x in overall_test_acc_dict["fl1mi"]]
    fl2mi_test_acc_p = [x for x in overall_test_acc_dict["fl2mi"]]
    # gc_test_acc_p = [x for x in overall_test_acc_dict["gc"]]
    gcmi_test_acc_p = [x for x in overall_test_acc_dict["gcmi"]]
    div_gcmi_test_acc_p = [x for x in overall_test_acc_dict["div-gcmi"]]
    # glister_test_acc_p = [x for x in overall_test_acc_dict["gradmatch-tss"]]
    # glister_tss_test_acc_p = [x for x in overall_test_acc_dict["glister-tss"]]
    # logdet_test_acc_p = [x for x in overall_test_acc_dict["logdet"]]
    logdetmi_test_acc_p = [x for x in overall_test_acc_dict["logdetmi"]]
    us_test_acc_p = [x for x in overall_test_acc_dict["us"]]
    # coreset_test_acc_p = [x for x in overall_test_acc_dict["coreset"]]
    # lc_test_acc_p = [x for x in overall_test_acc_dict["leastconf"]]
    # margin_test_acc_p = [x for x in overall_test_acc_dict["margin"]]
    random_test_acc_p = [x for x in overall_test_acc_dict["random"]]
    flcmi_test_acc_p = [x for x in overall_test_acc_dict["flmic"]]
    logdetcmi_test_acc_p = [x for x in overall_test_acc_dict["logdetmic"]]

    plt.plot(x_axis, badge_test_acc_p, mc[0][1], label='BADGE',marker=mc[0][0])
    # plt.plot(x_axis, fl_test_acc_p, mc[1][1], label='FL',marker=mc[1][0])
    # plt.plot(x_axis, fl1mi_test_acc_p, mc[2][1], label='FL1MI',marker=mc[2][0])
    plt.plot(x_axis, fl2mi_test_acc_p, mc[3][1], label='FL2MI',marker=mc[3][0])
    # plt.plot(x_axis, gc_test_acc_p, mc[4][1], label='GC',marker=mc[4][0])
    plt.plot(x_axis, gcmi_test_acc_p, mc[5][1], label='GCMI',marker=mc[5][0])
    plt.plot(x_axis, div_gcmi_test_acc_p, mc[6][1], label='DIV-GCMI',marker=mc[6][0])
    # plt.plot(x_axis, glister_test_acc_p, mc[7][1], label='GRADMATCH-TSS',marker=mc[7][0])
    # plt.plot(x_axis, glister_tss_test_acc_p, mc[8][1], label='GLISTER-TSS',marker=mc[8][0])
    # plt.plot(x_axis, logdet_test_acc_p, mc[9][1], label='LOGDET',marker=mc[9][0])
    plt.plot(x_axis, logdetmi_test_acc_p, mc[10][1], label='LOGDETMI',marker=mc[10][0])
    plt.plot(x_axis, us_test_acc_p, mc[11][1], label='UNCERTAINITY',marker=mc[11][0])
    # plt.plot(x_axis, coreset_test_acc_p, mc[12][1], label='CORESET',marker=mc[12][0])
    # plt.plot(x_axis, lc_test_acc_p, mc[13][1], label='LEAST_CONF',marker=mc[13][0])
    # plt.plot(x_axis, margin_test_acc_p, mc[14][1], label='MARGIN',marker=mc[14][0])
    plt.plot(x_axis, random_test_acc_p, mc[15][1], label='RANDOM',marker=mc[15][0])
    plt.plot(x_axis, flcmi_test_acc_p, mc[16][1], label='FLCMI',marker=mc[16][0])
    plt.plot(x_axis, logdetcmi_test_acc_p, mc[17][1], label='LOGDETCMI',marker=mc[17][0])

    plt.legend()
    plt.xlabel('No of Images')
    plt.ylabel('Test Accuracy')
    # plt.title(exp_name)
    plt.savefig(os.path.join(expDir,"Performance_Comparison:" + exp_name))
    plt.clf()

def targeted_test_acc_plot(target_test_acc_dict, plot_md, exp_name):
    x_axis = np.array([plot_md["train_size"]+plot_md["sel_budget"]*i for i in range(plot_md["num_selections"])])
    plt.figure()
    badge_test_acc_p = [x for x in target_test_acc_dict["badge"]]
    # fl_test_acc_p = [x for x in target_test_acc_dict["fl"]]
    # fl1mi_test_acc_p = [x for x in target_test_acc_dict["fl1mi"]]
    fl2mi_test_acc_p = [x for x in target_test_acc_dict["fl2mi"]]
    # gc_test_acc_p = [x for x in target_test_acc_dict["gc"]]
    gcmi_test_acc_p = [x for x in target_test_acc_dict["gcmi"]]
    div_gcmi_test_acc_p = [x for x in target_test_acc_dict["div-gcmi"]]
    # glister_test_acc_p = [x for x in target_test_acc_dict["gradmatch-tss"]]
    # glister_tss_test_acc_p = [x for x in target_test_acc_dict["glister-tss"]]
    # logdet_test_acc_p = [x for x in target_test_acc_dict["logdet"]]
    logdetmi_test_acc_p = [x for x in target_test_acc_dict["logdetmi"]]
    us_test_acc_p = [x for x in target_test_acc_dict["us"]]
    # coreset_test_acc_p = [x for x in target_test_acc_dict["coreset"]]
    # lc_test_acc_p = [x for x in target_test_acc_dict["leastconf"]]
    # margin_test_acc_p = [x for x in target_test_acc_dict["margin"]]
    random_test_acc_p = [x for x in target_test_acc_dict["random"]]
    flcmi_test_acc_p = [x for x in target_test_acc_dict["flmic"]]
    logdetcmi_test_acc_p = [x for x in target_test_acc_dict["logdetmic"]]
    
    plt.plot(x_axis, badge_test_acc_p, mc[0][1], label='BADGE',marker=mc[0][0])
    # plt.plot(x_axis, fl_test_acc_p, mc[1][1], label='FL',marker=mc[1][0])
    # plt.plot(x_axis, fl1mi_test_acc_p, mc[2][1], label='FL1MI',marker=mc[2][0])
    plt.plot(x_axis, fl2mi_test_acc_p, mc[3][1], label='FL2MI',marker=mc[3][0])
    # plt.plot(x_axis, gc_test_acc_p, mc[4][1], label='GC',marker=mc[4][0])
    plt.plot(x_axis, gcmi_test_acc_p, mc[5][1], label='GCMI',marker=mc[5][0])
    plt.plot(x_axis, div_gcmi_test_acc_p, mc[6][1], label='DIV-GCMI',marker=mc[6][0])
    # plt.plot(x_axis, glister_test_acc_p, mc[7][1], label='GRADMATCH-TSS',marker=mc[7][0])
    # plt.plot(x_axis, glister_tss_test_acc_p, mc[8][1], label='GLISTER-TSS',marker=mc[8][0])
    # plt.plot(x_axis, logdet_test_acc_p, mc[9][1], label='LOGDET',marker=mc[9][0])
    plt.plot(x_axis, logdetmi_test_acc_p, mc[10][1], label='LOGDETMI',marker=mc[10][0])
    plt.plot(x_axis, us_test_acc_p, mc[11][1], label='UNCERTAINITY',marker=mc[11][0])
    # plt.plot(x_axis, coreset_test_acc_p, mc[12][1], label='CORESET',marker=mc[12][0])
    # plt.plot(x_axis, lc_test_acc_p, mc[13][1], label='LEAST_CONF',marker=mc[13][0])
    # plt.plot(x_axis, margin_test_acc_p, mc[14][1], label='MARGIN',marker=mc[14][0])
    plt.plot(x_axis, random_test_acc_p, mc[15][1], label='RANDOM',marker=mc[15][0])
    plt.plot(x_axis, flcmi_test_acc_p, mc[16][1], label='FLCMI',marker=mc[16][0])
    plt.plot(x_axis, logdetcmi_test_acc_p, mc[17][1], label='LOGDETCMI',marker=mc[17][0])

    plt.legend()
    plt.xlabel('No of Images')
    plt.ylabel('Targeted Test Accuracy')
    # plt.title(exp_name)
    plt.savefig(os.path.join(expDir,"TargetedPerformance_Comparison:" + exp_name))
    plt.clf()

def target_samples_plot(num_target_samples_dict, plot_md, exp_name):
    x_axis = np.array(list(range(2,plot_md["num_selections"]+1)))
    plt.figure()
    badge_num_targeted_samples = num_target_samples_dict["badge"]
    # fl_num_targeted_samples = num_target_samples_dict["fl"]
    # fl1mi_num_targeted_samples = num_target_samples_dict["fl1mi"]
    fl2mi_num_targeted_samples = num_target_samples_dict["fl2mi"]
    # gc_num_targeted_samples = num_target_samples_dict["gc"]
    gcmi_num_targeted_samples = num_target_samples_dict["gcmi"]
    div_gcmi_num_targeted_samples = num_target_samples_dict["div-gcmi"]
    # glister_num_targeted_samples = num_target_samples_dict["gradmatch-tss"]
    # glister_tss_num_targeted_samples = num_target_samples_dict["glister-tss"]
    # logdet_num_targeted_samples = num_target_samples_dict["logdet"]
    logdetmi_num_targeted_samples = num_target_samples_dict["logdetmi"]
    us_num_targeted_samples = num_target_samples_dict["us"]
    # coreset_num_targeted_samples = num_target_samples_dict["coreset"]
    # lc_num_targeted_samples = num_target_samples_dict["leastconf"]
    # margin_num_targeted_samples = num_target_samples_dict["margin"]
    random_num_targeted_samples = num_target_samples_dict["random"]
    flcmi_num_targeted_samples = [x for x in num_target_samples_dict["flmic"]]
    logdetcmi_num_targeted_samples = [x for x in num_target_samples_dict["logdetmic"]]

    plt.plot(x_axis, badge_num_targeted_samples, mc[0][1], label='BADGE',marker=mc[0][0])
    # plt.plot(x_axis, fl_num_targeted_samples, mc[1][1], label='FL',marker=mc[1][0])
    # plt.plot(x_axis, fl1mi_num_targeted_samples, mc[2][1], label='FL1MI',marker=mc[2][0])
    plt.plot(x_axis, fl2mi_num_targeted_samples, mc[3][1], label='FL2MI',marker=mc[3][0])
    # plt.plot(x_axis, gc_num_targeted_samples, mc[4][1], label='GC',marker=mc[4][0])
    plt.plot(x_axis, gcmi_num_targeted_samples, mc[5][1], label='GCMI',marker=mc[5][0])
    plt.plot(x_axis, div_gcmi_num_targeted_samples, mc[6][1], label='DIV-GCMI',marker=mc[6][0])
    # plt.plot(x_axis, glister_num_targeted_samples, mc[7][1], label='GRADMATCH-TSS',marker=mc[7][0])
    # plt.plot(x_axis, glister_tss_num_targeted_samples, mc[8][1], label='GLISTER-TSS',marker=mc[8][0])
    # plt.plot(x_axis, logdet_num_targeted_samples, mc[9][1], label='LOGDET',marker=mc[9][0])
    plt.plot(x_axis, logdetmi_num_targeted_samples, mc[10][1], label='LOGDETMI',marker=mc[10][0])
    plt.plot(x_axis, us_num_targeted_samples, mc[11][1], label='UNCERTAINITY',marker=mc[11][0])
    # plt.plot(x_axis, coreset_num_targeted_samples, mc[12][1], label='CORESET',marker=mc[12][0])
    # plt.plot(x_axis, lc_num_targeted_samples, mc[13][1], label='LEAST_CONF',marker=mc[13][0])
    # plt.plot(x_axis, margin_num_targeted_samples, mc[14][1], label='MARGIN',marker=mc[14][0])
    plt.plot(x_axis, random_num_targeted_samples, mc[15][1], label='RANDOM',marker=mc[15][0])
    plt.plot(x_axis, flcmi_num_targeted_samples, mc[16][1], label='FLCMI',marker=mc[16][0])
    plt.plot(x_axis, logdetcmi_num_targeted_samples, mc[17][1], label='LOGDETCMI',marker=mc[17][0])

    plt.legend()
    plt.xlabel('Selection epoch')
    plt.ylabel('#Targeted Samples')
    # plt.title(exp_name)
    plt.savefig(os.path.join(expDir,"Targeted_Samples_comparison:" + exp_name))
    plt.clf()
    
if __name__=="__main__":
    for (root,_,files) in os.walk(expDir, topdown=True):
        for file in files:
            # if(file.endswith("runs"+str(run)+".json")): 
            if(file.endswith("runs"+str(run)+".json") and ("budget:"+str(budget) in file) and ("epochs:"+str(epochs) in file) and ("_"+str(num_sel_cls)+"_" in file)):
                print(file)
                with open(os.path.join(root, file)) as f:
                    print(os.path.join(root, file))
                    data = json.load(f)
                    num_cls = len(data["sel_per_cls"][0]) #last val is mean acc
                    if(feature=="classimb"):
                        if(exp_name==""): exp_name = data["dataset"] + "_" + data["feature"] + "_bud:" + str(data["sel_budget"]) + "_imb-train:" + str(data["setting"]["per_imbclass_train"]) + "_imb-lake:" + str(data["setting"]["per_imbclass_lake"]) + "_pc-train:" + str(data["setting"]["per_class_train"]) + "_pc-lake:" + str(data["setting"]["per_class_lake"])
                        if "train_size" not in plot_md: plot_md["train_size"] = (data["setting"]["per_imbclass_train"]*data["setting"]["num_cls_imbalance"]) + (data["setting"]["per_class_train"]*(num_cls-data["setting"]["num_cls_imbalance"]))
                        if "sel_budget" not in plot_md: plot_md["sel_budget"] = data["sel_budget"]
                        if "num_selections" not in plot_md: plot_md["num_selections"] = data["num_selections"]
                        overall_test_acc_dict[data["sel_func"]] = data["test_acc"]
                        target_test_acc = []
                        all_ep_num_targets_sel = []
                        for ep_num in range(len(data["all_class_acc"])):
                            target_ep_acc = 0
                            num_target_sel = 0
                            for idx in data["sel_cls_idx"]:
                                target_ep_acc += (100-data["all_class_acc"][ep_num][idx])
                                if(ep_num!=len(data["all_class_acc"])-1): num_target_sel += data["sel_per_cls"][ep_num][idx]
                            target_test_acc.append(target_ep_acc/len(data["sel_cls_idx"]))
                            if(ep_num!=len(data["all_class_acc"])-1): all_ep_num_targets_sel.append(num_target_sel)
                        target_test_acc_dict[data["sel_func"]] = target_test_acc
                        num_target_samples_dict[data["sel_func"]] = all_ep_num_targets_sel

                    if(feature=="ood"):
                        if(exp_name==""): exp_name = data["dataset"] + "_" + data["feature"] + "_bud:" + str(data["sel_budget"]) + "_idc-train:" + str(data["setting"]["per_idc_train"]) + "_idc-lake:" + str(data["setting"]["per_idc_lake"]) + "_pc-train:" + str(data["setting"]["per_ood_train"]) + "_pc-lake:" + str(data["setting"]["per_ood_lake"])
                        if "train_size" not in plot_md: plot_md["train_size"] = (data["setting"]["per_idc_train"]*data["setting"]["num_cls_idc"]) + (data["setting"]["per_ood_train"]*(num_cls-data["setting"]["num_cls_idc"]))
                        if "sel_budget" not in plot_md: plot_md["sel_budget"] = data["sel_budget"]
                        if "num_selections" not in plot_md: plot_md["num_selections"] = data["num_selections"]
                        overall_test_acc_dict[data["sel_func"]] = data["test_acc"]
                        print(data['test_acc'])
                        all_ep_num_targets_sel = []
                        for ep_num in range(len(data["sel_per_cls"])):
                            num_target_sel = 0
                            for idx in data["sel_cls_idx"]:
                                num_target_sel += data["sel_per_cls"][ep_num][idx]
                            all_ep_num_targets_sel.append(num_target_sel)
                        num_target_samples_dict[data["sel_func"]] = all_ep_num_targets_sel

    #make plots
    overall_test_acc_plot(overall_test_acc_dict, plot_md, exp_name)
    target_samples_plot(num_target_samples_dict, plot_md, exp_name)
    if(feature=="classimb"): targeted_test_acc_plot(target_test_acc_dict, plot_md, exp_name)
    print("plots saved at: ", expDir)
#python visualize_cg.py /home/snk170001/bioml/dss/notebooks/CG_active_learning_results/cifar10/duplicate
import matplotlib.pyplot as plt
import json
import os
import sys
import numpy as np

expDir = sys.argv[1]


test_acc_dict = {}
num_unique_samples_dict = {}
plot_md = {}
exp_name = ""

def test_acc_plot(test_acc_dict, plot_md, exp_name):
    x_axis = np.array([plot_md["train_size"]+plot_md["sel_budget"]*i for i in range(plot_md["num_selections"])])
    plt.figure()
    badge_test_acc_p = [round(float(x)*100, 2) for x in test_acc_dict["badge"]]
    # gccg_test_acc_p = [round(float(x)*100, 2) for x in test_acc_dict["badge"]]
    us_test_acc_p = [round(float(x)*100, 2) for x in test_acc_dict["us"]]
    # glister_test_acc_p = [round(float(x)*100, 2) for x in test_acc_dict["glister"]]
    fl1cg_test_acc_p = [round(float(x)*100, 2) for x in test_acc_dict["fl1cg"]]
    logdetcg_test_acc_p = [round(float(x)*100, 2) for x in test_acc_dict["logdetcg"]]
    random_test_acc_p = [round(float(x)*100, 2) for x in test_acc_dict["random"]]
    # fl_test_acc_p = [round(float(x)*100, 2) for x in test_acc_dict["fl"]]
    # gc_test_acc_p = [round(float(x)*100, 2) for x in test_acc_dict["gc"]]
    # logdet_test_acc_p = [round(float(x)*100, 2) for x in test_acc_dict["logdet"]]

    # plt.plot(x_axis, gccg_test_acc_p, 'b-', label='GCCG',marker='o')
    plt.plot(x_axis, fl1cg_test_acc_p, 'm-', label='FL1CG',marker='o')
    plt.plot(x_axis, logdetcg_test_acc_p, 'y-', label='LOGDETCG',marker='o')
    # plt.plot(x_axis, fl_test_acc_p, 'p-', label='FL',marker='o')
    # plt.plot(x_axis, gc_test_acc_p, 'b-', label='GC',marker='o')
    # plt.plot(x_axis, logdet_test_acc_p, 's-', label='LOGDET',marker='o')
    plt.plot(x_axis, us_test_acc_p, 'g-', label='UNCERTAINITY',marker='o')
    # plt.plot(x_axis, glister_test_acc_p, 'v-', label='GLISTER',marker='o')
    plt.plot(x_axis, badge_test_acc_p, 'c', label='BADGE',marker='o')
    plt.plot(x_axis, random_test_acc_p, 'r', label='RANDOM',marker='o')

    plt.legend()
    plt.xlabel('No of Images')
    plt.ylabel('Test Accuracy')
    # plt.title(exp_name)
    plt.savefig(os.path.join(expDir,"Performance_Comparison:" + exp_name))
    plt.clf()

def unique_samples_plot(num_unique_samples_dict, plot_md, exp_name):
    x_axis = np.array(list(range(3,plot_md["num_selections"]+1)))
    plt.figure()
    badge_num_unique_samples = num_unique_samples_dict["badge"][:]
    # gccg_num_unique_samples = num_unique_samples_dict["badge"][1:]
    us_num_unique_samples = num_unique_samples_dict["us"][:]
    # glister_num_unique_samples = num_unique_samples_dict["glister"][1:]
    fl1cg_num_unique_samples = num_unique_samples_dict["fl1cg"][1:]
    logdetcg_num_unique_samples = num_unique_samples_dict["logdetcg"][1:]
    random_num_unique_samples = num_unique_samples_dict["random"][:]
    # fl_num_unique_samples = num_unique_samples_dict["fl"][1:]
    # gc_num_unique_samples = num_unique_samples_dict["gc"][1:]
    # logdet_num_unique_samples = num_unique_samples_dict["logdet"][1:]

    # plt.plot(x_axis, gccg_test_acc_p, 'b-', label='GCCG',marker='o')
    plt.plot(x_axis, fl1cg_num_unique_samples, 'm-', label='FL1CG',marker='o')
    plt.plot(x_axis, logdetcg_num_unique_samples, 'y-', label='LOGDETCG',marker='o')
    # plt.plot(x_axis, fl_num_unique_samples, 'p-', label='FL',marker='o')
    # plt.plot(x_axis, gc_num_unique_samples, 'b-', label='GC',marker='o')
    # plt.plot(x_axis, logdet_num_unique_samples, 's-', label='LOGDET',marker='o')
    plt.plot(x_axis, us_num_unique_samples, 'g-', label='UNCERTAINITY',marker='o')
    # plt.plot(x_axis, glister_num_unique_samples, 'v-', label='GLISTER',marker='o')
    plt.plot(x_axis, badge_num_unique_samples, 'c', label='BADGE',marker='o')
    plt.plot(x_axis, random_num_unique_samples, 'r', label='RANDOM',marker='o')

    plt.legend()
    plt.xlabel('Selection epoch')
    plt.ylabel('#Unique Samples')
    # plt.title(exp_name)
    plt.savefig(os.path.join(expDir,"Unique_Samples_comparison:" + exp_name))
    plt.clf()

if __name__=="__main__":
    for (root,_,files) in os.walk(expDir, topdown=True):
        for file in files:
            if(file.endswith("json")):
                with open(os.path.join(root, file)) as f:
                    print(os.path.join(root, file))
                    data = json.load(f)
                    if(exp_name==""): exp_name = data["dataset"] + "_" + data["feature"] + "_bud:" + str(data["sel_budget"]) + "_train:" + str(data["setting"]["train_size"]) + "_lake:" + str(data["setting"]["lake_size"]) + "_num_rep:" + str(data["setting"]["num_rep"])
                    if "train_size" not in plot_md: plot_md["train_size"] = data["setting"]["train_size"]
                    if "sel_budget" not in plot_md: plot_md["sel_budget"] = data["sel_budget"]
                    if "num_selections" not in plot_md: plot_md["num_selections"] = data["num_selections"]
                    test_acc_dict[data["sel_func"]] = data["test_acc"]
                    num_unique_samples_dict[data["sel_func"]] = [i + plot_md["train_size"] for i in data["num_unique_samples"]]
    #make plots
    test_acc_plot(test_acc_dict, plot_md, exp_name)
    unique_samples_plot(num_unique_samples_dict, plot_md, exp_name)
    print("plots saved at: ", expDir)
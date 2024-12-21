import os
import sys
from typing import List
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))


def remove_padding(x: List[int]) -> List[int]:
    while -1 in x:
        x.remove(-1)
    return x


def process_one_gate(gates_files: List[str], model_name: str, catalog: str):

    gates_tensors = [torch.load(catalog + file) for file in gates_files]

    data_all_1 = remove_padding([int(i) for i in gates_tensors[0].flatten().cpu()])
    data_all_2 = remove_padding([int(i) for i in gates_tensors[1].flatten().cpu()])

    # ALL LAYERS

    y_labels = list(set(data_all_1))
    df = pd.DataFrame(
        {
            "Programming": [data_all_1.count(i) for i in y_labels],
            "Gaming": [data_all_2.count(i) for i in y_labels],
        },
        index=y_labels,
    )
    plt.figure(figsize=(16, 12))
    ax = df.plot.bar(color=["#a020f0", "#f364a1"])
    for r in ax.patches:
        height = r.get_height()
        ax.annotate(
            "{}".format(height),
            xy=(r.get_x() + r.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    ax.set_xticklabels(y_labels, rotation=0)
    ax.set_title("Распределение экспертов по всем слоям")
    ax.set_xlabel("Номер эксперта")
    ax.set_ylabel("Количество обращений к эксперту")
    plt.savefig("process_exp_results/images/" + model_name + "_all.png")

    # TO SUBREDDITS

    data_1_1 = remove_padding([int(i) for i in gates_tensors[0][::3].flatten().cpu()])
    data_1_2 = remove_padding([int(i) for i in gates_tensors[0][1::3].flatten().cpu()])
    data_1_3 = remove_padding([int(i) for i in gates_tensors[0][2::3].flatten().cpu()])

    df = pd.DataFrame(
        {
            "First block": [data_1_1.count(i) for i in y_labels],
            "Second block": [data_1_2.count(i) for i in y_labels],
            "Third block": [data_1_3.count(i) for i in y_labels],
        },
        index=y_labels,
    )
    plt.figure(figsize=(16, 12))
    ax = df.plot.bar(color=["#fe019a", "#04d9ff", "#fd4825"])
    for r in ax.patches:
        height = r.get_height()
        ax.annotate(
            "{}".format(height),
            xy=(r.get_x() + r.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
        )
    ax.set_xticklabels(y_labels, rotation=0)
    ax.set_xlabel("Номер эксперта")
    ax.set_ylabel("Количество обращений к эксперту")
    ax.set_title("Распределение экспертов по слоям Programming")
    plt.savefig("process_exp_results/images/" + model_name + "_programming.png")

    data_2_1 = remove_padding([int(i) for i in gates_tensors[1][::3].flatten().cpu()])
    data_2_2 = remove_padding([int(i) for i in gates_tensors[1][1::3].flatten().cpu()])
    data_2_3 = remove_padding([int(i) for i in gates_tensors[1][2::3].flatten().cpu()])

    df = pd.DataFrame(
        {
            "First block": [data_2_1.count(i) for i in y_labels],
            "Second block": [data_2_2.count(i) for i in y_labels],
            "Third block": [data_2_3.count(i) for i in y_labels],
        },
        index=y_labels,
    )
    plt.figure(figsize=(16, 12))
    ax = df.plot.bar(color=["#8ae777", "#7977e7", "#efC55e"])
    for r in ax.patches:
        height = r.get_height()
        ax.annotate(
            "{}".format(height),
            xy=(r.get_x() + r.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
        )
    ax.set_xticklabels(y_labels, rotation=0)
    ax.set_title("Распределение экспертов по слоям Gaming")
    ax.set_xlabel("Номер эксперта")
    ax.set_ylabel("Количество обращений к эксперту")
    plt.savefig("process_exp_results/images/" + model_name + "_gaming.png")


def main():

    catalog = "process_exp_results/results/"
    model_name = "moe_model"

    files = []
    for file in os.listdir(catalog):
        if file.endswith(".pt") and "_".join(file.split("_")[:-2]) == model_name:
            files.append(file)

    process_one_gate(files, model_name, catalog)


if __name__ == "__main__":
    main()

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import math

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def graph_loss(loss_train, loss_valid, title=None, save=None, print_min = False) :
    plt.clf()

    x = range(1, len(loss_train) + 1)

    y_min = math.floor(min(loss_train) * 1000) / 1000 - 0.001
    y_max = math.ceil(max(loss_valid) * 1000) / 1000
    diff = 0.2 * (y_max - y_min)
    y_tick = [round(y_min, 3), round(y_min + diff, 3), round(y_min + 2 * diff, 3), round(y_min + 3 * diff, 3), round(y_min + 4 * diff, 3), round(y_max, 3)]

    plt.plot(x, loss_train)
    plt.plot(x, loss_valid)
    plt.legend(["train", "valid"])
    plt.ylim(y_min, y_max)
    plt.yticks(y_tick)
    plt.xlabel("epoch")
    plt.ylabel("loss")

    if title is None :
        graph_title = f"Graph of Loss"
    else :
        graph_title = title

    if print_min :
        graph_title = graph_title + f"\nMin : {round(min(loss_train), 4)}/{round(min(loss_valid), 4)} at epoch {loss_train.index(min(loss_train))+1}/{loss_valid.index(min(loss_valid))+1}"

    plt.title(graph_title)

    if save is None :
        plt.show()
    else :
        plt.savefig(save)

def graph_single(item, y_label, title=None, save=None, print_min = False, print_max = False) :
    plt.clf()

    x = range(1, len(item) + 1)

    plt.plot(x, item)
    plt.xlabel("epoch")
    plt.ylabel(y_label)

    if title is None:
        graph_title = f"Graph of {y_label}"
    else:
        graph_title = title

    if print_max:
        graph_title = graph_title + f"\nMax : {round(max(item), 4)} at epoch {item.index(max(item)) + 1}"
    if print_min:
        graph_title = graph_title + f"\nMin : {round(min(item), 4)} at epoch {item.index(min(item)) + 1}"

    plt.title(graph_title)

    if save is None :
        plt.show()
    else :
        plt.savefig(save)

def graph_double(item_train, item_valid, y_label, title=None, save=None, print_min = False, print_max = False) :
    plt.clf()

    x = range(1, len(item_train) + 1)

    plt.plot(x, item_train)
    plt.plot(x, item_valid)
    plt.legend(["train", "valid"])

    plt.xlabel("epoch")
    plt.ylabel(y_label)

    if title is None :
        graph_title = f"Graph of {y_label}"
    else :
        graph_title = title

    if print_max :
        graph_title = graph_title + f"\nMax : {round(max(item_train), 4)}/{round(max(item_valid), 4)} at epoch {item_train.index(max(item_train))+1}/{item_valid.index(max(item_valid))+1}"
    if print_min :
        graph_title = graph_title + f"\nMin : {round(min(item_train), 4)}/{round(min(item_valid), 4)} at epoch {item_train.index(min(item_train))+1}/{item_valid.index(min(item_valid))+1}"

    plt.title(graph_title)

    if save is None :
        plt.show()
    else :
        plt.savefig(save)
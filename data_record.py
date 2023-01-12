import os
import sys
import copy

import matplotlib.pyplot as plt

class RecordBox():
    # -------------------------
    # 1. init
    #   box_loss = RecordBox(name = "loss_name")
    #
    # 2. add item in batch
    #   box_loss.add_item(some_float_value)
    #
    # 3. at the end of batch
    #   box_loss.update_batch()     #-> str 형으로 log return 기능 추가히기 (is_return_str)
    #
    # 4. at the end of epoch
    #   box_loss.update_epoch() #this draws log graph
    # -------------------------

    def __init__(self, name='name', is_print=True, print_interval=1, will_update_graph=True, **kargs):
        # Parameter name
        self.name = name
        self.print_head = "(RecordBox) " + self.name + " -> "

        # print(self.print_head, "init")

        self.is_print = is_print

        if print_interval < 1:
            self.print_interval = 1
        else:
            self.print_interval = int(print_interval)

        # print(self.print_head, "Generated RecordBox name:", self.name)
        if '/' in self.name:
            print(self.print_head, "(exc) name should not include /")
            sys.exit(9)

        if self.is_print:
            print(self.print_head, "Update Batch log will be printed once per every", str(self.print_interval),
                  "updates")

        self.count_single_item = 0  # (float)   single item number in 1 batch
        self.single_item_sum = 0  # (float)   single item sum per batch

        self.count_batch = 0  # (int)     batch number in 1 epoch
        self.batch_item = 0  # (float)   avg per batch (single_item_sum / count_single_item)
        self.batch_item_sum = 0  # (float)   batch_item sum per epoch
        self.record_batch_item = []  # (list)    all batch_item in 1 epoch
        self.record_batch_item_prev = []  # (list)    prev epoch's record_batch_item

        self.count_epoch = 0  # (int)     epoch number in total RUN
        self.epoch_item = 0  # (float)   avg per epoch (batch_item_sum / count_batch)
        self.record_epoch_item = []  # (list)    all epoch_items in total RUN

        self.total_min = (0, 0)  # (tuple)   (epoch number, min epoch_item)
        self.total_max = (0, 0)  # (tuple)   (epoch number, MAX epoch_item)

        self.count_fig_save = 0  # (int)     log fig save count

        self.is_best_max = False  # (bool)    is lastly updated value is best (max)
        self.is_best_min = False  # (bool)    is lastly updated value is best (min)

        self.will_update_graph = will_update_graph  # (bool)    will update(save) log graph

        if self.will_update_graph:
            print(self.print_head, "This will update graph every epoch.")
        else:
            print(self.print_head, "This will NOT update graph every epoch!")


    # --- functions used outside

    # add new item (in batch)
    def add_item(self, item):
        self.count_single_item += 1  # (int)     update to current number of items

        self.single_item_sum += item  # (float)   Sum items in batch

    # update when batch end
    def update_batch(self, is_return=False):
        self.count_batch += 1  # (int)     update to current number of batches

        try:
            self.batch_item = self.single_item_sum / self.count_single_item
            self.single_item_sum = 0
            self.count_single_item = 0
        except:
            print(self.print_head, "(exc) self.count_single_item is Zero")
            sys.exit(9)

        self.batch_item_sum += self.batch_item

        if self.is_print and (self.count_batch - 1) % self.print_interval == 0:
            print(self.print_head, "update batch <", str(self.count_epoch + 1), "-", str(self.count_batch), ">",
                  str(round(self.batch_item, 4)))

        self.record_batch_item.append(self.batch_item)

        if is_return:
            # returns avg value of items in batch
            return self.batch_item

    # update when epoch end
    def update_epoch(self, is_return=False
                     , is_show=False, is_save=True, path="./"
                     , is_print_sub=False
                     , is_update_graph=None
                     ):
        self.is_best_max = False  # reset flag
        self.is_best_min = False  # reset flag
        self.count_epoch += 1  # update to current number of epoches

        if is_update_graph is None:
            _update_graph = self.will_update_graph
        else:
            _update_graph = is_update_graph
            if _update_graph:
                print(self.print_head, "graph updated.")
            else:
                print(self.print_head, "graph update SKIPPED!")

        try:
            self.epoch_item = self.batch_item_sum / self.count_batch
            self.batch_item_sum = 0
            self.count_batch = 0
        except:
            print(self.print_head, "(exc) self.count_batch is Zero")
            sys.exit(9)

        if self.is_print or is_print_sub:
            if _update_graph:
                print(self.print_head, "update epoch <", str(self.count_epoch), ">", str(round(self.epoch_item, 4)))

        self.record_epoch_item.append(self.epoch_item)

        if self.count_epoch == 1:
            self.total_min = (self.count_epoch, self.epoch_item)
            self.total_max = (self.count_epoch, self.epoch_item)
            self.is_best_max = True
            self.is_best_min = True
        else:
            if self.total_min[-1] >= self.epoch_item:
                self.total_min = (self.count_epoch, self.epoch_item)
                self.is_best_min = True

            if self.total_max[-1] <= self.epoch_item:
                self.total_max = (self.count_epoch, self.epoch_item)
                self.is_best_max = True

        self.record_batch_item_prev = copy.deepcopy(self.record_batch_item)
        self.record_batch_item = []

        # --- for update_graph()
        self.is_show = is_show
        self.is_save = is_save
        self.path = path

        if _update_graph:
            self.update_graph(is_show=self.is_show, is_save=self.is_save, path=self.path)

        if is_return:
            # returns avg value of items in epoch
            return self.epoch_item

    # return record list
    def get_record(self, select):
        if select == 'batch':
            # get last updated epoch's batch item list (per epoch)
            return self.record_batch_item_prev
        elif select == 'epoch':
            # get total's epoch item list
            return self.record_epoch_item

    # --- functions used inside class

    # save graph
    def update_graph(self, is_show=False, is_save=True, path="./"):
        self.count_fig_save += 1

        list_x_labels = []
        for i in range(len(self.record_epoch_item)):
            list_x_labels.append(i + 1)

        plt.figure(figsize=(10, 8))
        # main graph
        plt.plot(list_x_labels, self.record_epoch_item)
        # MAX point
        plt.scatter([int(self.total_max[0])], [self.total_max[-1]], c='red', s=100)
        # min point
        plt.scatter([int(self.total_min[0])], [self.total_min[-1]], c='lawngreen', s=100)

        plt.xlabel("\nEpoch")

        tmp_title = ("[log] " + self.name + " in epoch " + str(int(self.count_epoch))
                     + "\nMAX in epoch " + str(int(self.total_max[0])) + ", " + str(round(self.total_max[-1], 4))
                     + "\nmin in epoch " + str(int(self.total_min[0])) + ", " + str(round(self.total_min[-1], 4))
                     + "\n"
                     )
        plt.title(tmp_title)
        if is_show:
            plt.show()

        if is_save:
            if path[-1] != '/':
                path += '/'
            tmp_path = path + "log_graph/" + self.name + "/"
            tmp_path_2 = path + "log_graph/"
            try:
                if not os.path.exists(tmp_path):
                    os.makedirs(tmp_path)
                plt.savefig(tmp_path + self.name + "_" + str(int(1 + ((self.count_fig_save - 1) % 5))) + ".png",
                            dpi=200)
                plt.savefig(tmp_path_2 + self.name + ".png", dpi=200)
            except:
                print(self.print_head, "(exc) log graph save FAIL")

        plt.close()
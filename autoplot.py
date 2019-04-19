from sklearn.metrics import roc_curve, auc, roc_auc_score
import keras.callbacks as kc
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import os

class AutoPlot(kc.Callback):
    def __init__(self, fig_path=None, fold=None):
        self.fig = plt.figure(figsize=(10, 8))
        self.dice = [0]
        self.loss = [0]
        self.val_loss = [0]
        self.val_dice = [0]
        self.lr = []
        self.fold = fold
        self.fig_path = fig_path

    def plot_keras(self, loss, dice, val_loss, val_dice, fold, lr):

        def __plot(ax, title, xlabel, ylabel, label1='train', label2='valid', loc='upper left'):
            def _plot(data1, data2):
                data1 = data1[1:]
                data2 = data2[1:]
                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                l1, = ax.plot(data1)
                if len(data2) != 0:
                    l2, = ax.plot(data2)
                    plt.legend([l1, l2], [label1 + ' = %.4f' % data1[-1], label2 + ' = %.4f' % data2[-1]], loc=loc)
            return _plot


        def __plot_lr(ax, title, xlabel, ylabel, label='lr', loc='upper left'):
            def _plot_lr(lr):
                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                l1, = ax.plot(lr)
                plt.legend([l1], [label + ' = %.4f' % lr[-1]], loc=loc)
            return _plot_lr


        # def __plot_lr_and_loss(ax, title, xlabel, ylabel, label1='lr', label2='loss', loc='lower right'):
        #     def _plot_lr_and_loss(lr, loss):
        #         plt.title(title)
        #         plt.xlabel(xlabel)
        #         plt.ylabel(ylabel)
        #         l1, = ax.plot(lr, loss)
        #         plt.legend([l1], [label1 + ' = %.4f' % lr], loc=loc)
        #     return _plot_lr_and_loss
        plt.ion()
        plt.clf()

        ax = plt.subplot(2, 2, 1)
        # __plot_lr_and_loss(ax, 'Runtime lr and loss', 'lr', 'loss')(lr, loss)
        ax = plt.subplot(2, 2, 2)
        __plot_lr(ax, 'Runtime lr', 'epochs', 'lr')(lr)
        ax = plt.subplot(2, 2, 3)
        __plot(ax, 'Runtime dice', 'epochs', 'dice')(dice, val_dice)
        ax = plt.subplot(2, 2, 4)
        __plot(ax, 'Runtime Loss', 'epochs', 'loss')(loss, val_loss)
        # plt.show()

        save_dir = self.fig_path
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        plt.savefig(save_dir + str(fold) + '.png')

    def on_epoch_end(self, epoch, logs={}):
        self.dice.append(logs['dice_coef'])
        self.loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])
        self.val_dice.append(logs['val_dice_coef'])
        self.lr.append(K.get_value(self.model.optimizer.lr) * 1e3)

        self.plot_keras(
            self.loss, self.dice,
            self.val_loss, self.val_dice,
            self.fold, self.lr)
        self.fig.canvas.draw()

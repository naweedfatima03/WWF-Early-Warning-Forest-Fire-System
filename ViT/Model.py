import torch
from torch import nn
from d2l import torch as d2l
import collections
from IPython import display

def validation_step(model, batch):
        Y_hat = model(*batch[:-1])
        plot(model, 'loss', model.loss(Y_hat, batch[-1]), train=False)
        plot(model, 'acc', model.accuracy(Y_hat, batch[-1]), train=False)
        
def plot(model, key, value, train):
        """Plot a point in animation."""
        assert hasattr(model, 'trainer'), 'Trainer is not inited'
        model.board.xlabel = 'epoch'
        if train:
            x = model.trainer.train_batch_idx / \
                model.trainer.num_train_batches
            n = model.trainer.num_train_batches / \
                model.plot_train_per_epoch
        else:
            x = model.trainer.epoch + 1
            n = model.trainer.num_val_batches / \
                model.plot_valid_per_epoch
        draw(model.board, x, d2l.numpy(d2l.to(value, d2l.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))
        
def draw(board, x, y, label, every_n=1):
        """Defined in :numref:`sec_utils`"""
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(board, 'raw_points'):
            board.raw_points = collections.OrderedDict()
            board.data = collections.OrderedDict()
        if label not in board.raw_points:
            board.raw_points[label] = []
            board.data[label] = []
        points = board.raw_points[label]
        line = board.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))
        points.clear()
        if not board.display:
            return
        d2l.use_svg_display()
        if board.fig is None:
            board.fig = d2l.plt.figure(figsize=board.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(board.data.items(), board.ls, board.colors):
            plt_lines.append(d2l.plt.plot([p.x for p in v], [p.y for p in v],
                                          linestyle=ls, color=color)[0])
            labels.append(k)
        axes = board.axes if board.axes else d2l.plt.gca()
        if board.xlim: axes.set_xlim(board.xlim)
        if board.ylim: axes.set_ylim(board.ylim)
        if not board.xlabel: board.xlabel = board.x
        axes.set_xlabel(board.xlabel)
        axes.set_ylabel(board.ylabel)
        axes.set_xscale(board.xscale)
        axes.set_yscale(board.yscale)
        axes.legend(plt_lines, labels)
        display.display(board.fig)
        print('Saving display...')
        board.fig.savefig('test.png')
        display.clear_output(wait=True)
        

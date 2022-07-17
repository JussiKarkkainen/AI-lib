import matplotlib.pyplot as plt
import collections


class printLoss:
    '''plot datapoints'''
    def __init__(self, x_label=None, y_label=None, X_lim=None
                y_lim=None, x_scale='linear', y_scale='linear',
                ls=['-', '--', '-,', ':'], colors=['c0', 'c1', 'c2', 'c3',],
                fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        
        self.x_label = x_label
        self.y_label = y_label
        self.X_lim = X_lim
        self.y_lim = y_lim
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.ls = ls
        self.colors = colors
        self.fig = fig
        self.axes = axes
        self.figsize = figsize
        self.display = display

    def draw(self, x, y, label, refresh=1):
        


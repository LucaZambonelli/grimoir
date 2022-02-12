#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""EaZy Plot Module

class:
    Plottable: a class to easily plot up to 4 items

function:
    check_law: a function to easily check laws of motion w/ derivatives
"""

__version__ = '0.1.00'
__author__ = 'Luca Zambonelli'
__copyright__ = '2022, Luca Zambonelli'
__license__ = 'GPL'
__maintainer__ = 'Luca Zambonelli'
__email__ = 'luca.zambonelli@gmail.com'
__status__ = 'Development'

import numpy as np
import matplotlib.pyplot as plt


# class definition
class Plottable:
    """a class to easily plot up to 4 items

---------   ---------   ---------   ---------
|       |   |   0   |   | 0 | 1 |   | 0 | 1 |
|   0   |   |-------|   |-------|   |-------|
|       |   |   1   |   | 2 |   |   | 2 | 3 |
---------   ---------   ---------   ---------

input:
    master: ascissae axis

methods:
    add_abscissa(master): adds an abscissae axis, returns the number
    add_ordinate(wh_master, wh_plot, wh_funct): adds an ordinates axis
    plot([norm]): shows the plot [default: normalized]
"""

    # class initialization
    def __init__(self):
        """private function: class initialization"""
        self._master = []
        self._s0 = []
        self._s1 = []
        self._s2 = []
        self._s3 = []
        self._slave = [self._s0, self._s1, self._s2, self._s3]

    # public methods
    def add_abscissa(self, master):
        """adds one abscissa"""
        self._master.append(master)
        return len(self._master) -1

    def add_ordinate(self, wh_master, wh_plot, wh_funct):
        """adds one ordinate"""
        self._slave[wh_plot].append((wh_master, wh_funct))

    def plot(self, norm=True):
        """shows the plot"""
        if len(self._slave[0]) == 0:
            return
        if len(self._slave[1]) == 0:
            fig, ax = plt.subplots(1, 1)
            for s in self._slave[0]:
                if norm:
                    y = s[1] / max(max(s[1]), -min(s[1]))
                else:
                    y = s[1]
                ax.plot(self._master[s[0]], y)
        elif len(self._slave[2]) == 0:
            fig, ax = plt.subplots(2, 1)
            for s in self._slave[0]:
                if norm:
                    y = s[1] / max(max(s[1]), -min(s[1]))
                else:
                    y = s[1]
                ax[0].plot(self._master[s[0]], y)
            for s in self._slave[1]:
                if norm:
                    y = s[1] / max(max(s[1]), -min(s[1]))
                else:
                    y = s[1]
                ax[1].plot(self._master[s[0]], y)
        elif len(self._slave[3]) == 0:
            fig, ax = plt.subplots(2, 2)
            for s in self._slave[0]:
                if norm:
                    y = s[1] / max(max(s[1]), -min(s[1]))
                else:
                    y = s[1]
                ax[0][0].plot(self._master[s[0]], y)
            for s in self._slave[1]:
                if norm:
                    y = s[1] / max(max(s[1]), -min(s[1]))
                else:
                    y = s[1]
                ax[0][1].plot(self._master[s[0]], y)
            for s in self._slave[2]:
                if norm:
                    y = s[1] / max(max(s[1]), -min(s[1]))
                else:
                    y = s[1]
                ax[1][0].plot(self._master[s[0]], y)
        else:
            fig, ax = plt.subplots(2, 2)
            for s in self._slave[0]:
                if norm:
                    y = s[1] / max(max(s[1]), -min(s[1]))
                else:
                    y = s[1]
                ax[0][0].plot(self._master[s[0]], y)
            for s in self._slave[1]:
                if norm:
                    y = s[1] / max(max(s[1]), -min(s[1]))
                else:
                    y = s[1]
                ax[0][1].plot(self._master[s[0]], y)
            for s in self._slave[2]:
                if norm:
                    y = s[1] / max(max(s[1]), -min(s[1]))
                else:
                    y = s[1]
                ax[1][0].plot(self._master[s[0]], y)
            for s in self._slave[3]:
                if norm:
                    y = s[1] / max(max(s[1]), -min(s[1]))
                else:
                    y = s[1]
                ax[1][1].plot(self._master[s[0]], y)
        plt.tight_layout()
        plt.show()

#function definition
def check_law(*laws, norm=False, gradients=False, points=False):
    """a function to easily check laws of motion w/ derivatives

input:
    laws to be plotted
    [norm]: plots normalized to 1
    [gradient]: whether numeric gradients are shown or not
    [points]: whether point-wise calculation is shown or not
"""
    z = Plottable()
    for l in laws:
        t = l.master()
        wh = z.add_abscissa(t)
        z.add_ordinate(wh, 0, l.position())
        if gradients:
            b = [l.position()]
            z.add_ordinate(wh, 0, b[0])
        if points:
            c = [np.array(list(l.position(x) for x in t))]
            z.add_ordinate(wh, 0, c[0])
        if hasattr(l, 'velocity'):
            z.add_ordinate(wh, 1, l.velocity())
            if gradients:
                b.append(np.gradient(l.position(), t, edge_order=2))
                z.add_ordinate(wh, 1, b[1])
            if points:
                c.append(np.array(list(l.velocity(x) for x in t)))
                z.add_ordinate(wh, 1, c[1])
        if hasattr(l, 'acceleration'):
            z.add_ordinate(wh, 2, l.acceleration())
            if gradients:
                b.append(np.gradient(l.velocity(), t, edge_order=2))
                z.add_ordinate(wh, 2, b[2])
            if points:
                c.append(np.array(list(l.acceleration(x) for x in t)))
                z.add_ordinate(wh, 2, c[2])
        if hasattr(l, 'jerk'):
            z.add_ordinate(wh, 3, l.jerk())
            if gradients:
                b.append(np.gradient(l.acceleration(), t, edge_order=2))
                z.add_ordinate(wh, 3, b[3])
            if points:
                c.append(np.array(list(l.jerk(x) for x in t)))
                z.add_ordinate(wh, 3, c[3])
    z.plot(norm)

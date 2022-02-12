#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mechatronics Grimoir Package

modules:
    kinematics: .. a collection of kinematic chains and related tools
    motion: ...... a collection of laws of motion and related tools
"""

__all__ = ['motion', 'kinematics']
__version__ = '0.1.00'
__author__ = 'Luca Zambonelli'
__copyright__ = '2022, Luca Zambonelli'
__license__ = 'GPL'
__maintainer__ = 'Luca Zambonelli'
__email__ = 'luca.zambonelli@gmail.com'
__status__ = 'Prototype'

from .motion import Motion
from .kinematics import Kinematics

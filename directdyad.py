#!/usr/bin/env python
# -*- coding: utf-8 -*-

# RUN CHECK FOR PACKAGE grimoir.kinematics (open dyad, direct)

import numpy as np

import z_plot
import grimoir.kinematics as gk
import grimoir.motion as gm

f21 = gm.ConstVelStroke(
    t=np.linspace(0/6, 1/6, 32),
    xgo=0.0,
    xng=2*np.pi 
    )
f22 = gm.ConstVelStroke(
    t=np.linspace(1/6, 2/6, 32),
    xgo=0.0,
    xng=2*np.pi 
    )
f23 = gm.ConstVelStroke(
    t=np.linspace(2/6, 3/6, 32),
    xgo=0.0,
    xng=2*np.pi 
    )
f24 = gm.ConstVelStroke(
    t=np.linspace(3/6, 4/6, 32),
    xgo=0.0,
    xng=2*np.pi 
    )
f25 = gm.ConstVelStroke(
    t=np.linspace(4/6, 5/6, 32),
    xgo=0.0,
    xng=2*np.pi 
    )
f26 = gm.ConstVelStroke(
    t=np.linspace(5/6, 6/6, 32),
    xgo=0.0,
    xng=2*np.pi 
    )
f27 = gm.Stitch(f21, f22, f23, f24, f25, f26)
f28 = gm.ConstAccRamp(t=np.linspace(0.0, 1.0, 513), xgo=0.0, vgo=1.0, vng=1.0)
m2 = gm.Compose(f28, f27)
del f21, f22, f23, f24, f25, f26, f27, f28
m1 = gm.ConstVelStroke(
    t=m2.master(),
    xgo=0.0,
    xng=2*np.pi
    )

dyad = gk.OpenDyad(
    mover_1=m1,
    mover_2=m2,
    crank_1=65.0,
    crank_2=45.0
    )


z_plot.check_law(dyad.follower()[0], dyad.follower[1])

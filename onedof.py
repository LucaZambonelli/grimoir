#!/usr/bin/env python
# -*- coding: utf-8 -*-

# RUN CHECK FOR PACKAGE grimoir.kinematics (one degree of freedom)

import numpy as np

import z_plot
import grimoir.kinematics as gk
import grimoir.motion as gm

f1 = gm.PolySpline(t=np.linspace(0.0, 3/16, 2))
f1.set_startpoint(x=0.0, v=0.0)
f1.set_endpoint(v=0.36)
f1.set_leg(0, deg=2)
f1.solve()
f2 = gm.SineImpulse(
    t=np.linspace(3 / 16, 5 / 16, 2),
    xgo=f1.position()[-1],
    vgo=f1.velocity()[-1],
    ago=f1.acceleration()[-1],
    ang=-f1.acceleration()[-1]
    )
f3 = gm.ConstImpulse(
    t=np.linspace(5 / 16, 11 / 16, 2),
    xgo=f2.position()[-1],
    vgo=f2.velocity()[-1],
    ago=f2.acceleration()[-1],
    ang=f2.acceleration()[-1]
    )
f4 = gm.SineImpulse(
    t=np.linspace(11 / 16, 13 / 16, 2),
    xgo=f3.position()[-1],
    vgo=f3.velocity()[-1],
    ago=f3.acceleration()[-1],
    ang=-f3.acceleration()[-1]
    )
f5 = gm.ConstImpulse(
    t=np.linspace(13 / 16, 1.0, 2),
    xgo=f4.position()[-1],
    vgo=f4.velocity()[-1],
    ago=f4.acceleration()[-1],
    ang=f4.acceleration()[-1]
    )
f6 = gm.Stitch(f1, f2, f3, f4, f5)
f7 = gm.ConstAccRamp(t=np.linspace(0.0, 1.0, 1025), xgo=0.0, vgo=1.0, vng=1.0)
f8 = gm.Compose(f7, f6)
f9 = gm.Slice(f8, tgo=1 / 8)
f9.set_new_master(tgo=0.0)
f10 = gm.Slice(f8, tng=1 / 8)
f10.set_new_master(tng=1.0)
FollowerLaw = gm.Stitch(f9, f10)
FollowerLaw = gm.Shift(FollowerLaw)
FollowerLaw.set_new_master(0.0, 0.12)
del f1, f2, f3, f4, f5, f6, f7, f8, f9, f10

chain0 = gk.CrankSlider(
    follower=FollowerLaw,
    motor_inertia=3.49e-3, 
    gearbox_inertia=7.66e-4,
    gear_ratio=4,
    efficiency=0.85,
    mover_inertia=1.3e-2,
    follower_inertia=30,
    external_load=1200.0
    )
chain0.features()

chain1 = gk.RockerSlider(
    follower=FollowerLaw,
    distance=0.22,
    mover_inertia=1.3e-2,
    follower_inertia=30,
    external_load=1200.0)
chain1.features()

chain2 = gk.CrankRocker(
    follower=chain1.mover(),
    distance=0.165,
    motor_inertia=3.49e-3,
    gearbox_inertia=7.66e-4,
    gear_ratio=4,
    efficiency=0.85,
    mover_inertia=1.3e-2,
    follower_inertia=chain1.user_inertia(),
    external_load=chain1.torque_external(),
    )
chain2.features()

chain3 = gk.CrankRodRocker(
    follower=chain1.mover(),
    distance=0.195,
    rocker=0.12,
    motor_inertia=3.49e-3,
    gearbox_inertia=7.66e-4,
    gear_ratio=4,
    efficiency=0.85,
    mover_inertia=1.3e-2,
    follower_inertia=chain1.user_inertia(),
    external_load=chain1.torque_external(),
    )
chain3.features()

chain5 = gk.CrankRodSlider(
    follower=FollowerLaw,
    rod=0.12,
    distance=0.02,
    motor_inertia=3.49e-3,
    gearbox_inertia=7.66e-4,
    gear_ratio=4,
    efficiency=0.85,
    mover_inertia=1.3e-2,
    follower_inertia=30,
    external_load=1200.0
    )
chain5.features()

q = z_plot.Plottable()
where_user = q.add_abscissa(FollowerLaw.master() * 1000)
where_time0 = q.add_abscissa(chain0.time() * 1000)
where_time2 = q.add_abscissa(chain2.time() * 1000)
where_time3 = q.add_abscissa(chain3.time() * 1000)
where_time5 = q.add_abscissa(chain5.time() * 1000)
where_vel0 = q.add_abscissa(np.absolute(chain0.velocity()) * 30 / np.pi)
where_vel2 = q.add_abscissa(np.absolute(chain2.velocity()) * 30 / np.pi)
where_vel3 = q.add_abscissa(np.absolute(chain3.velocity()) * 30 / np.pi)
where_vel5 = q.add_abscissa(np.absolute(chain5.velocity()) * 30 / np.pi)
q.add_ordinate(where_user, 0, FollowerLaw.velocity())
q.add_ordinate(where_time0, 1, chain0.mover().velocity() * 30 / np.pi)
q.add_ordinate(where_time2, 1, chain2.mover().velocity() * 30 / np.pi)
q.add_ordinate(where_time3, 1, chain3.mover().velocity() * 30 / np.pi)
q.add_ordinate(where_time5, 1, chain5.mover().velocity() * 30 / np.pi)
q.add_ordinate(where_vel0, 2, np.absolute(chain0.torque()))
q.add_ordinate(where_vel2, 2, np.absolute(chain2.torque()))
q.add_ordinate(where_vel3, 2, np.absolute(chain3.torque()))
q.add_ordinate(where_vel5, 2, np.absolute(chain5.torque()))
q.add_ordinate(where_vel0, 3, np.absolute(chain0.power()) / 1000)
q.add_ordinate(where_vel2, 3, np.absolute(chain2.power()) / 1000)
q.add_ordinate(where_vel3, 3, np.absolute(chain3.power()) / 1000)
q.add_ordinate(where_vel5, 3, np.absolute(chain5.power()) / 1000)
q.plot(False)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import grimoir.motion as gm
import z_plot

stations = 4

m_start = 0.0
m_acc = 22.5
m_follow = 180.0

s_start = 180.0
s_span = 43.0

master = np.linspace(0.0, 360.0, 361)

anse_1 = gm.PolySpline(master, 2)
anse_1.set_startpoint(x=s_start, v=0.0)
anse_1.set_endpoint()
anse_1.set_point(
    m_acc,
    x = -(s_span * m_acc) / (2 * (m_acc+m_follow)) + s_start
)
anse_1.set_point(
    m_acc + m_follow,
    x = -(s_span * (m_acc + 2*m_follow)) / (2 * (m_acc+m_follow)) + s_start
)
anse_1.set_point(
    2*m_acc + m_follow,
    x = -(s_span * (m_acc+m_follow)) / (m_acc + m_follow) + s_start,
    v = 0.0
)
anse_1.set_leg(0, 4)
anse_1.set_leg(1, 1)
anse_1.set_leg(2, 4)
anse_1.solve()

ruota_1 = gm.PolySpline(master, 2)
ruota_1.set_endpoint(x=-90.0, cont=5)
ruota_1.set_point(
    m_acc,
    x = -(s_span * m_acc) / (2 * (m_acc+m_follow)) + s_start
)
ruota_1.set_point(
    m_acc + m_follow,
    x = -(s_span * (m_acc + 2*m_follow)) / (2 * (m_acc+m_follow)) + s_start
)
ruota_1.set_leg(1,1)
ruota_1.solve()

m_min = 0.0
m_max = m_acc
goon = True
while goon:
    m_med = (m_max+m_min) / 2
    f_min = ruota_1.position(m_min) - s_start
    f_med = ruota_1.position(m_med) - s_start
    if f_min*f_med <= 0.0:
        m_max = m_med
    else:
        m_min = m_med
    goon = abs(m_max-m_min) / 2 >= 1e-12
m_start = (m_max+m_min) / 2

gradi_macchina_2 = master[:-1].copy()
anse_2 = master[:-1].copy()
ruota_2 = master[:-1].copy()
for i in range(np.shape(gradi_macchina_2)[0]):
    g = gradi_macchina_2[i]
    m = (g+m_start) % 360
    anse_2[i] = anse_1.position(m)
    ruota_2[i] = ruota_1.position(m)
    if m < g:
        ruota_2[i] -= 360 / stations

gradi_macchina = gradi_macchina_2.copy()
anse = anse_2.copy()
ruota = ruota_2.copy()
for i in range(1, stations):
    gradi_macchina = np.block([gradi_macchina, gradi_macchina_2 + 360*i])
    anse = np.block([anse, anse_2])
    ruota = np.block([ruota, ruota_2 - (360*i)/stations])
gradi_macchina = np.block([gradi_macchina, 360*stations])
anse = np.block([anse, anse[0]])
ruota = np.block([ruota, ruota[0] - 360])

anse = gm.Import(gradi_macchina, anse, 0)
ruota = gm.Import(gradi_macchina, ruota, 0)

p_anse = anse.position()
v_anse = anse.velocity()*(180/np.pi)
a_anse = anse.acceleration()*(180/np.pi)**2
j_anse = anse.jerk()*(180/np.pi)**3
p_ruota = ruota.position()
v_ruota = ruota.velocity()*(180/np.pi)
a_ruota = ruota.acceleration()*(180/np.pi)**2
j_ruota = ruota.jerk()*(180/np.pi)**3

q = z_plot.Plottable()
q1 = q.add_abscissa(gradi_macchina)
q.add_ordinate(q1, 0, p_ruota)
q.add_ordinate(q1, 0, p_anse)
q.add_ordinate(q1, 1, v_ruota)
q.add_ordinate(q1, 1, v_anse)
q.add_ordinate(q1, 2, a_ruota)
q.add_ordinate(q1, 2, a_anse)
q.add_ordinate(q1, 3, j_ruota)
q.add_ordinate(q1, 3, j_anse)
q.plot(False)

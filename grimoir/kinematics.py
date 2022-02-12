#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Kinematics Module
a collection of kinematic chains and related tools

NOTE: Kinematics Module deals with physics, units of measure are very
      important to get consistent results!
      All classes, methods, functions, etc. are written to manipulate
      consistent International System Units:
          m  for lengths
          kg for masses
          s  for times
      and their derived, in particular
          rad      for angles
          kg m**2  for moment of inertia
          N        for forces
          N m      for torques
          W        for power

Classes for one degree of freedom kinematic chains:
    CrankCrank: ...... crank and crank kinematic chian
                       crank + RPR dyad
    CrankRocker: ..... crank and rocker kinematic chain
                       crank + RPR dyad
    CrankRodRocker: .. crank, rod and rocker kinematic chain
                       crank + RRR dyad
    CrankSlider: ..... crank and slider kinematic chain
                       crank + RPP dyad
    CrankRodSlider: .. crank, rod and slider kinematic chain
                       crank + RRP dyad
    RockerSlider: .... rocker and slider kinematic chain
                       rocker + PRP dyad
    Screw: ........... screw and nut kinematic chain
                       H joint

Classes for two degrees of freedom kinematic chains:
    OpenDyad: .. crank and crank kinematic chain
    FiveBar: ... rocker and rocker kinematic chain
"""

__version__ = '0.0.07'
__author__ = 'Luca Zambonelli'
__copyright__ = '2022, Luca Zambonelli'
__license__ = 'GPL'
__maintainer__ = 'Luca Zambonelli'
__email__ = 'luca.zambonelli@gmail.com'
__status__ = 'Prototype'

import numpy as np

# callable variable for upper level package
Kinematics = True

# private function to deal with angles
def _angle(sint, cost):
    """private function to return angle from sin and cos"""
    sint[sint > 1] = 1
    sint[sint < -1] = -1
    ang = np.empty_like(sint)
    wh = np.empty_like(sint)
    wh = (cost >= 0) & (sint >= 0)
    ang[wh] = np.arcsin(sint[wh])
    wh = (cost >= 0) & (sint < 0)
    ang[wh] = 2*np.pi + np.arcsin(sint[wh])
    wh = cost < 0
    ang[wh]= np.pi - np.arcsin(sint[wh])
    return ang

# private class to return generic law of motion
class _generic_law:
    """private class to return generic law of motion"""

    # class initialization
    def __init__(self, time, pos, vel, acc):
        self._time = time
        self._pos = pos
        self._vel = vel
        self._acc = acc
        self._jrk = np.empty_like(self._time)
        self._jrk[:] = np.NaN

    # public methods
    def master(self):
        """returns the mseter axis"""
        return self._time

    def position(self):
        """returns the slave position"""
        return self._pos

    def velocity(self):
        """returns the slave velocity"""
        return self._vel

    def acceleration(self):
        """returns the slave acceleration"""
        return self._acc

    def jerk(self):
        """returns the slave jerk"""
        return self._jrk

# private class for 1 d.o.f. kinematic chain
class _kinematic_one:
    """private class for 1 d.o.f. kinematic chain"""

    # private functions
    def _assign(self, flag):
        """private function: assign masters and slaves"""
        if flag:
            self._time = self._mover.master()
            self._mov_pos = self._mover.position()
            self._mov_vel = self._mover.velocity()
            self._mov_acc = self._mover.acceleration()
        else:
            self._time = self._follower.master()
            self._fol_pos = self._follower.position()
            self._fol_vel = self._follower.velocity()
            self._fol_acc = self._follower.acceleration()
        return flag

    def _mount(self, flag):
        """private function: mounts the kinematic chain"""
        if flag:
            # kinematic composition, direct
            self._fol_vel = self._geo_vel * self._mov_vel
            self._fol_acc = (self._geo_acc*self._mov_vel**2
                             + self._geo_vel*self._mov_acc)
        else:
            # inverse geometric derivatives
            zeros = np.where(np.absolute(self._geo_vel) <= 1e-8)[0]
            self._geo_vel[zeros] = np.NaN
            self._inv_vel = 1 / self._geo_vel
            self._inv_acc = -self._geo_acc / self._geo_vel**3
            self._geo_vel[zeros] = 0.0
            # kinematic composition, inverse
            self._mov_vel = self._inv_vel * self._fol_vel
            self._mov_acc = (self._inv_acc*self._fol_vel**2
                             + self._inv_vel*self._fol_acc)
            movi = self._flatten(self._mov_pos)
            for i in range(len(zeros)):
                if zeros[i] < 9:
                    indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                elif zeros[i] > len(self._geo_vel) - 8:
                    indexes = [-9, -8, -7, -6, -5, -4, -3, -2, -1]
                else:
                    indexes = [zeros[i] - 4, zeros[i] - 3,
                               zeros[i] - 2, zeros[i] - 1, zeros[i],
                               zeros[i] + 1, zeros[i] + 2,
                               zeros[i] + 3, zeros[i] + 4]
                self._mov_vel[indexes] = np.gradient(movi[indexes],
                                                     self._time[indexes],
                                                     edge_order=2)
                self._mov_acc[indexes] = np.gradient(self._mov_vel[indexes],
                                                     self._time[indexes],
                                                     edge_order=2)
        # load at the follower
        self._fol_load = (self._fol_j * (self._geo_vel*self._mov_acc
                          + self._geo_acc*self._mov_vel**2) + self._ext_load)
        # user inertia: contribution of follower inertia, including derivative
        self._usr_j = self._fol_j * self._geo_vel**2
        d_fol_j = np.gradient(self._usr_j, self._time, edge_order=2)
        # user inertia: contribution of mover inertia
        self._usr_j += self._mov_j
        # user inertia: contribution of gearbox, including inertia ratio
        self._usr_j /= self._gear**2
        self._usr_j += self._box_j
        if self._mot_j == 0:
            self._j_ratio = np.NaN
        else:
            self._j_ratio = max(self._usr_j / self._mot_j)
        # load at motor: contribution of external load
        self._mot_load_f = self._geo_vel * self._ext_load / self._gear
        d_load = np.gradient(self._ext_load, self._time, edge_order=2)
        # load at motor: contribution of follower inertia
        self._mot_load_j = (self._geo_vel *
                            (self._fol_j*(self._geo_vel*self._mov_acc +
                            2*self._geo_acc*self._mov_vel**2)
                            + d_fol_j*self._geo_vel*self._mov_vel))
        # load at motor: contribution of mover inertia
        self._mot_load_j += self._mov_j * self._mov_acc
        # load at motor: contribution of gearbox and motor
        self._mot_load_j /= self._gear
        self._mot_load_j += ((self._box_j + self._mot_j) * self._mov_acc
                             * self._gear)
        # load at motor: total, including rms
        self._mot_load = self._mot_load_j + self._mot_load_f
        self._mot_rms = 0
        for i in range(1, len(self._mot_load)):
            self._mot_rms += ((self._mot_load[i]+self._mot_load[i-1])**2
                              * (self._time[i]-self._time[i-1]) / 4)
        self._mot_rms /= self._time[-1] - self._time[0]
        self._mot_rms = np.sqrt(self._mot_rms)
        # velocity of the motor, including average
        self._mot_vel = self._mov_vel * self._gear
        self._mot_avg = 0
        for i in range(1, len(self._mot_vel)):
            self._mot_avg += (abs(self._mot_vel[i]+self._mot_vel[i-1])
                              * (self._time[i]-self._time[i-1]) / 2)
        self._mot_avg /= self._time[-1] - self._time[0]
        # power consumption: contribution of external load
        self._pow = (d_load*self._fol_pos
                     + self._ext_load*self._geo_vel*self._mov_vel)
        # power consumption: contribution of follower inertia
        self._pow += (self._geo_vel * self._mov_vel *
                      (self._fol_j*(self._geo_vel*self._mov_acc +
                      self._geo_acc*self._mov_vel**2)
                      + d_fol_j*self._geo_vel*self._mov_vel/2))
        # power consumption: contribution of mover inertia
        self._pow += self._mov_j * self._mov_vel * self._mov_acc
        # power consumption: contribution motor and gearbox
        self._pow += ((self._box_j+self._mot_j) * self._mov_vel
                      * self._mov_acc * self._gear**2)
        # power consumption: rms
        self._pow_rms = 0
        for i in range(1, len(self._pow)):
            self._pow_rms += ((self._pow[i]+self._pow[i-1])**2
                              * (self._time[i]-self._time[i-1]) / 4)
        self._pow_rms /= self._time[-1] - self._time[0]
        self._pow_rms = np.sqrt(self._pow_rms)
        self._max_vel = max(max(self._mot_vel), -min(self._mot_vel))
        self._max_load = max(max(self._mot_load), -min(self._mot_load))
        self._max_pow = max(max(self._pow), -min(self._pow))

    # public methods
    def time(self):
        """returns the time axis"""
        return self._time

    def mover(self):
        """returns the mover law of motion"""
        return _generic_law(
            self._time,
            self._mov_pos,
            self._mov_vel,
            self._mov_acc
            )

    def follower(self):
        """returns the follower law of motion"""
        return _generic_law(
            self._time,
            self._fol_pos,
            self._fol_vel,
            self._fol_acc
            )

    def user_inertia(self):
        """returns the inertial property seen at the motor"""
        return self._usr_j

    def velocity(self):
        """returns the motor velocity"""
        return self._mot_vel

    def torque(self):
        """returns the motor torque"""
        return self._mot_load

    def torque_external(self):
        """returns the torque requirement for external load"""
        return self._mot_load_f

    def power(self):
        """returns the motor power"""
        return self._pow

    # static methods
    @staticmethod
    def _flatten(law_in):
        """static method to remove discontinuities in mover law"""
        law_out = law_in.copy()
        for i in range(2, len(law_out)):
            if law_out[i-1]-law_out[i-2] >= 0:
                if law_out[i]-law_out[i-1] < 0:
                    law_out[i] += 2 * np.pi
            if law_out[i-1]-law_out[i-2] <= 0:
                if law_out[i]-law_out[i-1] > 0:
                    law_out[i] -= 2 * np.pi
        return law_out


# private class for 2 d.o.f. kinematic chain
class _kinematic_two:
    """private class for 2 d.o.f. kinematic chain"""
    
    # private functions
    def _assign(self, flag):
        """private function: assign master and slaves"""
        if flag:
            self._time = self._mov1.master()
            self._mov1_pos = self._mov1.position()
            self._mov1_vel = self._mov1.velocity()
            self._mov1_acc = self._mov1.acceleration()
            self._mov2_pos = self._mov2.position()
            self._mov2_vel = self._mov2.velocity()
            self._mov2_acc = self._mov2.acceleration()
        else:
            self._time = self._fol1.master()
            self._fol1_pos = self._fol1.position()
            self._fol1_vel = self._fol1.velocity()
            self._fol1_acc = self._fol1.acceleration()
            self._fol2_pos = self._fol2.position()
            self._fol2_vel = self._fol2.velocity()
            self._fol2_acc = self._fol2.acceleration()
        return flag
    
    # public methods
    def time(self):
        """returns the time axis"""
        return self._time
    
    def mover(self):
        """returns the movers laws of motion"""
        return (
            _generic_law(
                self._time,
                self._mov1_pos,
                self._mov1_vel,
                self._mov1_acc
                ),
            _generic_law(
                self._time,
                self._mov2_pos,
                self._mov2_vel,
                self._mov2_acc
                )
            )

    def follower(self):
        """returns the follower laws of motion"""
        return (
            _generic_law(
                self._time,
                self._fol1_pos,
                self._fol1_vel,
                self._fol1_acc
                ),
            _generic_law(
                self._time,
                self._mov2_pos,
                self._mov2_vel,
                self._mov2_acc
                )
            )

# classes for one degree of freedom kinematic chains
class CrankCrank(_kinematic_one):
    """crank and crank mechanism - one degree of freedom
(crank + RPR dyad)

input, set 1: direct motion
    mover: law of motion of the mover

input, set 2: inverse motion
    follower: law of motion of the follower

input, any set
    crank: length of the crank
    disctance: distance between the centers of rotation of the cranks
    [motor_inertia]: inertial property of the motor
    [gearbox_inertia]: inertial property of the gearbox
    [gear_ratio]: gear ratio between the motor and the mover
    [efficiency]: efficiency of the transmission
    [mover_inertia]: inertial property of the mover (including gear)
    [follower_inertia]: inertial property of the follower
    [external_load]: load acting on the follower

methods
    features(): get chain status description
    get_features(): returns a tuple with the features of the chain
    time(): returns the time axis
    mover(): returns the mover law of motion
    follower(): returns the follower law of motion
    user_inertia(): returns the inertial property seen at the motor
    velocity(): returns the motor velocity
    torque(): returns the motor torque
    torque_external(): returns the torque requirement for external load
    power(): returns the motor power
"""

    # class initialization
    def __init__(
        self,
        mover=None,
        follower=None,
        distance=None,
        crank=None,
        motor_inertia=0.0,
        gearbox_inertia=0.0,
        gear_ratio=1.0,
        efficiency=1.0,
        mover_inertia=0.0,
        follower_inertia=0.0,
        external_load=0.0
        ):
        """private function: class initialization"""
        if distance is None or crank is None:
            return
        elif distance >= crank:
            return
        self._crank = crank
        self._dist = distance
        if follower is None and mover is None:
            return
        elif follower is None:
            self._mover = mover
            flag = self._assign(True)
            m = self._crank
            d = self._dist
            t_1 = np.pi + np.arcsin(d/m)
            t_2 = 2*np.pi - np.arcsin(d/m)
            self._fol_pos = -(np.arctan(m * np.cos(self._mov_pos)
                              / (d + m*np.sin(self._mov_pos))))
            self._fol_pos[(self._mov_pos>=t_1)
                          & (self._mov_pos<=t_2)] += np.pi
            self._fol_pos[self._mov_pos > t_2] += 2*np.pi
        else:
            self._follower = follower
            flag = self._assign(False)
            x = np.tan(self._fol_pos)
            rt = self._crank**2 + x**2*(self._crank**2 -self._dist**2)
            rt[rt < 0] = 0
            sint = - self._dist*x**2
            inner = self._fol_pos>=(np.pi/2) & self._fol_pos<=(3*np.pi/2)
            sint[inner] -= np.sqrt(rt[inner])
            sint[~inner] += np.sqrt(rt[~inner])
            sint /= self._crank * (1+x**2)
            cost = np.ones_like(sint)
            cost[self._fol_pos>=0 & self._fol_pos <= np.pi] *= -1
            self._mov_pos = self._angle(sint, cost)
        self._geo_vel = (self._crank * (self._crank
                         + self._dist*np.sin(self._mov_pos))
                         / (self._crank**2 + self._dist**2 +
                         2*self._crank*self._dist*np.sin(self._mov_pos)))
        self._geo_acc = (self._crank * self._dist * (self._dist**2
                         - self._crank**2) * np.cos(self._mov_pos)
                         / (self._crank**2 + self._dist**2
                         + 2*self._crank*self._dist*np.sin(self._mov_pos))**2)
        self._gear = gear_ratio
        self._eta = efficiency
        self._mot_j = motor_inertia
        self._box_j = gearbox_inertia
        self._mov_j = mover_inertia
        self._fol_j = follower_inertia
        if isinstance(external_load, np.ndarray):
            self._ext_load = external_load
        else:
            self._ext_load = np.ones_like(self._time) * external_load
        self._mount(flag)

    # public methods
    def features(self):
        """get chain status description"""
        print('\ncrank and crank machanism\ncrank + RPR\n')
        print(f'crank length: {self._crank*1000:5.1f} mm')
        print(f'distance: {self._dist*1000:5.1f} mm')
        print()
        print(f'user / motor inertia ratio: {self._j_ratio:4.2f}')
        print()
        print(f'maximum motor velocity: {self._max_vel*30/np.pi:5.0f} rpm')
        print()
        print(f'maximum motor torque: {self._max_load:5.1f} N')
        print(f'nominal working point: ({self._mot_avg*30/np.pi:5.0f}'
              f' rpm, {self._mot_rms:5.1f} N m)')
        print()
        print(f'maximum power: {self._max_pow/1000:5.1f} kW')
        print(f'nominal working point: ({self._mot_avg*30/np.pi:5.0f}'
              f' rpm, {self._pow_rms/1000:5.1f} kW)')

    def get_features(self):
        """returns the features of the chian ([-1] for help)"""
        dist = self._dist
        crank = self._crank
        ratio = self._j_ratio
        vel = self._mot_avg
        trq = self._mot_rms
        powr = self._pow_rms
        label = '0 - crank length'
        label += '\n1 - distance'
        label += '\n4 - user / motor inertia ratio'
        label += '\n5 - nominal working point (velocity, torque)'
        label += '\n6 - nominal working point (velocity, power)'
        result = (crank, dist, ratio, (vel, trq), (vel, powr), label)
        return result


class CrankRocker(_kinematic_one):
    """crank and rocker mechanism - one degree of freedom
(crank + RPR dyad)

input, set 1: direct motion
    mover: law of motion of the mover
    crank: length of the crank

input, set 2: inverse motion
    follower: law of motion of the follower
    [crank]: length of the crank

input, any set
    disctance: distance between the centers of rotation of the crank
               and the rocker
    [motor_inertia]: inertial property of the motor
    [gearbox_inertia]: inertial property of the gearbox
    [gear_ratio]: gear ratio between the motor and the mover
    [efficiency]: efficiency of the transmission
    [mover_inertia]: inertial property of the mover (including gear)
    [follower_inertia]: inertial property of the follower
    [external_load]: load acting on the follower

methods
    features(): get chain status description
    get_features(): returns a tuple with the features of the chain
    time(): returns the time axis
    mover(): returns the mover law of motion
    follower(): returns the follower law of motion
    user_inertia(): returns the inertial property seen at the motor
    velocity(): returns the motor velocity
    torque(): returns the motor torque
    torque_external(): returns the torque requirement for external load
    power(): returns the motor power
"""

    # class initialization
    def __init__(
        self,
        mover=None,
        follower=None,
        distance=None,
        crank=None,
        motor_inertia=0.0,
        gearbox_inertia=0.0,
        gear_ratio=1.0,
        efficiency=1.0,
        mover_inertia=0.0,
        follower_inertia=0.0,
        external_load=0.0
        ):
        """private function: class initialization"""
        if distance is None:
            return
        elif follower is None and mover is None:
            return
        elif follower is None:
            if crank is None:
                return
            elif crank >= distance:
                return
            else:
                self._crank = crank
                self._dist = distance
                self._mover = mover
                flag = self._assign(True)
                self._fol_pos = -(np.arctan(m * np.cos(self._mov_pos)
                                  / (d + m*np.sin(self._mov_pos))))
        else:
            if (crank is None and max(follower.position())
                    -min(follower.position())) >= np.pi:
                return
            elif crank is None:
                self._crank = (distance * np.sin((max(follower.position())
                               -min(follower.position()))/2))
                rotation = True
            elif crank >= distance:
                return
            else:
                self._crank = crank
                rotation = False
            self._dist = distance
            self._follower = follower
            flag = self._assign(False)
            x = np.tan(self._fol_pos)
            rt = self._crank**2 + x**2*(self._crank**2 -self._dist**2)
            rt[rt < 0] = 0
            sint = -(self._dist*x**2 - np.sqrt(rt)) / (self._crank * (x**2+1))
            if rotation:
                closure = self._fol_vel < 0
                sint[closure] = -((self._dist*x[closure]**2
                                  + np.sqrt(rt[closure]))
                                  / (self._crank * (x[closure]**2+1)))
            cost = x*sint - x*self._dist/self._crank
            self._mov_pos = _angle(sint, cost)
        self._geo_vel = (self._crank * (self._crank
                         + self._dist*np.sin(self._mov_pos))
                         / (self._crank**2 + self._dist**2 +
                         2*self._crank*self._dist*np.sin(self._mov_pos)))
        self._geo_acc = (self._crank * self._dist * (self._dist**2
                         - self._crank**2) * np.cos(self._mov_pos)
                         / (self._crank**2 + self._dist**2
                         + 2*self._crank*self._dist*np.sin(self._mov_pos))**2)
        self._gear = gear_ratio
        self._eta = efficiency
        self._mot_j = motor_inertia
        self._box_j = gearbox_inertia
        self._mov_j = mover_inertia
        self._fol_j = follower_inertia
        if isinstance(external_load, np.ndarray):
            self._ext_load = external_load
        else:
            self._ext_load = np.ones_like(self._time) * external_load
        self._mount(flag)
        self._t_bdc = 2*np.pi - np.arcsin(self._crank/self._dist)
        self._z_bdc = -np.arcsin(self._crank/self._dist)
        self._t_tdc = np.pi + np.arcsin(self._crank/self._dist)
        self._z_tdc = np.arcsin(self._crank/self._dist)

    # public methods
    def features(self):
        """get chain status description"""
        print('\ncrank and rocker mechanism\ncrank + RPR\n')
        print(f'crank length: {self._crank*1000:5.1f} mm')
        print(f'distance: {self._dist*1000:5.1f} mm')
        print()
        print(f'bottom dead center: ({self._t_bdc*180/np.pi:5.1f} deg, '
              f'{self._z_bdc*180/np.pi:5.1f} deg)')
        print(f'top dead center:    ({self._t_tdc*180/np.pi:5.1f} deg, '
              f'{self._z_tdc*180/np.pi:5.1f} deg)')
        print()
        print(f'user / motor inertia ratio: {self._j_ratio:4.2f}')
        print()
        print(f'maximum motor velocity: {self._max_vel*30/np.pi:5.0f} rpm')
        print()
        print(f'maximum motor torque: {self._max_load:5.1f} N')
        print(f'nominal working point: ({self._mot_avg*30/np.pi:5.0f}'
              f' rpm, {self._mot_rms:5.1f} N m)')
        print()
        print(f'maximum power: {self._max_pow/1000:5.1f} kW')
        print(f'nominal working point: ({self._mot_avg*30/np.pi:5.0f}'
              f' rpm, {self._pow_rms/1000:5.1f} kW)')

    def get_features(self):
        """returns the features of the chian ([-1] for help)"""
        dist = self._dist
        crank = self._crank
        t_bdc = self._t_bdc
        z_bdc = self._z_bdc
        t_tdc = self._t_tdc
        z_tdc = self._z_tdc
        ratio = self._j_ratio
        vel = self._mot_avg
        trq = self._mot_rms
        powr = self._pow_rms
        label = '0 - crank length'
        label += '\n1 - distance'
        label += '\n2 - bottom dead center (theta, beta)'
        label += '\n3 - top dead center    (theta, beta)'
        label += '\n4 - user / motor inertia ratio'
        label += '\n5 - nominal working point (velocity, torque)'
        label += '\n6 - nominal working point (velocity, power)'
        result = (crank, dist, (t_bdc, z_bdc), (t_tdc, z_tdc), ratio,
                  (vel, trq), (vel, powr), label)
        return result


class CrankRodRocker(_kinematic_one):
    """ crank, rod and rocker mechanism - one degree of freedom
(crank + RRR dyad)

input, set 1:
    mover: law of motion of the mover
    crank: length of the crank
    rod: length of the connection rod

input, set 2:
    follower: law of motion of the follower
    [crank]: length of the crank
    [rod]: length of the connection rod

input, any set
    distance: distance between the centers of rotation of the crank
              and the rocker
    rocker: length of the rocker
    [motor_inertia]: inertial property of the motor
    [gearbox_inertia]: inertial property of the gearbox
    [gear_ratio]: gear ratio between the motor and the mover
    [efficiency]: efficiency of the transmission
    [mover_inertia]: inertial property of the mover (including gear)
    [follower_inertia]: inertial property of the follower
    [external_load]: load acting on the follower

methods
    features(): get chain status description
    get_features(): returns a tuple with the features of the chain
    time(): returns the time axis
    mover(): returns the mover law of motion
    follower(): returns the follower law of motion
    user_inertia(): returns the inertial property seen at the motor
    velocity(): returns the motor velocity
    torque(): returns the motor torque
    torque_external(): returns the torque requirement for external load
    power(): returns the motor power
"""

    # class initialization
    def __init__(
        self,
        mover=None,
        follower=None,
        distance=None,
        rocker=None,
        crank=None,
        rod=None,
        motor_inertia=0.0,
        gearbox_inertia=0.0,
        gear_ratio=1.0,
        efficiency=1.0,
        mover_inertia=0.0,
        follower_inertia=0.0,
        external_load=0.0
        ):
        if follower is None and mover is None:
            return
        elif follower is None:
            if (distance is None or rocker is None or crank is None
                    or rod is None):
                return
            else:
                case1 = ((crank < rocker < distance < rod)
                         & (crank+rod < rocker+distance))
                case2 = ((crank < rocker < rod < distance)
                         & (crank+distance < rocker+rod))
                if not (case1 or case2):
                    return
                else:
                    self._crank = crank
                    self._rod = rod
                    self._mover = mover
                    flag = self._assign(True)
        else:
            if ((crank is None or rod is None) and max(follower.position())
                    -min(follower.position())) >= np.pi:
                return
            elif crank is None or rod is None:
                r = rocker
                d = distance
                z_BDC = min(follower.position())
                z_TDC = max(follower.position())
                ka = np.sqrt(r**2 + d**2 - 2*r*d*np.cos(np.pi/2 + z_BDC))
                kb = np.sqrt(r**2 + d**2 - 2*r*d*np.cos(np.pi/2 + z_TDC))
                m = (kb-ka) / 2
                b = ka + m
                self._crank = m
                self._rod = b
                self._follower = follower
                flag = self._assign(False)
                rotation = True
            else:
                self._crank = crank
                self._rod = rod
                flag = self._assign(False)
                rotation = False
        self._rocker = rocker
        self._distance = distance
        m = self._crank
        b = self._rod
        d = self._distance
        r = self._rocker
        if not flag:
            z = self._fol_pos
            k4 = m**2 + r**2 + d**2 - b**2 + 2*r*d*np.sin(z)
            k5 = -2 * m * r * np.cos(z)
            k6 = 2 * m * (d + r*np.sin(z))
            kdelta = k5**2 + k6**2 - k4**2
            kdelta[kdelta < 0] = 0
            sint = -(k4*k5 - k6*np.sqrt(kdelta)) / (k5**2 + k6**2)
            if rotation:
                closure = self._fol_vel < 0
                sint[closure] = -((k4[closure]*k5[closure]
                                  + k6[closure]*np.sqrt(kdelta[closure]))
                                  / (k5[closure]**2 + k6[closure]**2))
            cost = -(k4 + k5*sint) / k6
            self._mov_pos = _angle(sint, cost)
        t = self._mov_pos
        k1 = m**2 + r**2 + d**2 - b**2 + 2*m*d*np.cos(t)
        dk1 = -2 * m * d * np.sin(t)
        ddk1 = -2 * m * d * np.cos(t)
        k2 = 2 * r * (d + m*np.cos(t))
        dk2 = -2 * m * r * np.sin(t)
        ddk2 = -2 * m * r * np.cos(t)
        k3 = - 2 * m * r * np.sin(t)
        dk3 = - 2 * m * r * np.cos(t)
        ddk3 = 2 * m * r * np.sin(t)
        n1 = k1 * k2
        dn1 = k1*dk2 + k2*dk1
        ddn1 = k1*ddk2 + k2*ddk1 + 2*dk1*dk2
        n2 = k2**2 + k3**2 - k1**2
        n2[n2 < 0] = 0
        dn2 = 2 * (k2*dk2 + k3*dk3 - k1*dk1)
        ddn2 = 2 * (k2*ddk2 + dk2**2 + k3*ddk3 + dk3**2 - k1*ddk1 - dk1**2)
        n3 = k3 * np.sqrt(n2)
        dn3 = (k3*dn2)/(2*np.sqrt(n2)) + dk3*np.sqrt(n2)
        ddn3 = ((2*(k3*ddn2 + dk3*dn2)*n2 - k3*dn2**2)/(4*np.sqrt(n2**3))
                +(dk3*dn2)/(2*np.sqrt(n2)) + ddk3*np.sqrt(n2))
        den = k2**2 + k3**2
        dden = 2 * (k2*dk2 + k3*dk3)
        ddden = 2 * (k2*ddk2 + dk2**2 + k3*ddk3 + dk3**2)
        arg = -(n1+n3) / den
        arg[arg > 1] = 1
        arg[arg < -1] = -1
        darg = -((dn1+dn3)*den - (n1+n3)*dden) / den**2
        ddarg = -(((ddn1+ddn3)*den - (n1+n3)*ddden) / den**2
                  -(2*dden*((dn1+dn3)*den - (n1+n3)*dden)) / den**2)
        if flag:
            self._fol_pos = np.arcsin(arg)
        self._geo_vel = darg / np.sqrt(1 - arg**2)
        self._geo_acc = ((ddarg*(1 - arg**2) + arg*darg**2)
                         / np.sqrt((1 -arg**2)**3))
        self._gear = gear_ratio
        self._eta = efficiency
        self._mot_j = motor_inertia
        self._box_j = gearbox_inertia
        self._mov_j = mover_inertia
        self._fol_j = follower_inertia
        if isinstance(external_load, np.ndarray):
            self._ext_load = external_load
        else:
            self._ext_load = np.ones_like(self._time) * external_load
        self._mount(flag)
        self._t_bdc = (2*np.pi
                       - np.arccos(((b-m)**2 + d**2 - r**2) / (2 * d * (b-m))))
        self._z_bdc = np.arccos((b-m)*np.sin(self._t_bdc) / r) - np.pi
        self._t_tdc = (np.pi
                       - np.arccos(((b+m)**2 + d**2 - r**2) / (2 * d * (b+m))))
        self._z_tdc = np.arccos((b+m)*np.sin(self._t_tdc) / r)

    # public methods
    def features(self):
        """get chain status description"""
        print('\ncrank, rod and rocker mechanism\ncrank + RPR\n')
        print(f'crank length: {self._crank*1000:5.1f} mm')
        print(f'rod length: {self._rod*1000:5.1f} mm')
        print(f'rocker length: {self._rocker*1000:5.1f} mm')
        print(f'distance: {self._distance*1000:5.1f} mm')
        print()
        print(f'bottom dead center: ({self._t_bdc*180/np.pi:5.1f} deg, '
              f'{self._z_bdc*180/np.pi:5.1f} deg)')
        print(f'top dead center:    ({self._t_tdc*180/np.pi:5.1f} deg, '
              f'{self._z_tdc*180/np.pi:5.1f} deg)')
        print()
        print(f'user / motor inertia ratio: {self._j_ratio:4.2f}')
        print()
        print(f'maximum motor velocity: {self._max_vel*30/np.pi:5.0f} rpm')
        print()
        print(f'maximum motor torque: {self._max_load:5.1f} N')
        print(f'nominal working point: ({self._mot_avg*30/np.pi:5.0f}'
              f' rpm, {self._mot_rms:5.1f} N m)')
        print()
        print(f'maximum power: {self._max_pow/1000:5.1f} kW')
        print(f'nominal working point: ({self._mot_avg*30/np.pi:5.0f}'
              f' rpm, {self._pow_rms/1000:5.1f} kW)')

    def get_features(self):
        """returns the features of the chian ([-1] for help)"""
        dist = self._distance
        crank = self._crank
        rod = self._rod
        rocker = self._rocker
        t_bdc = self._t_bdc
        z_bdc = self._z_bdc
        t_tdc = self._t_tdc
        z_tdc = self._z_tdc
        ratio = self._j_ratio
        vel = self._mot_avg
        trq = self._mot_rms
        powr = self._pow_rms
        label = '0 - crank length'
        label += '\n1 - rod length'
        label += '\n2 - rocker length'
        label += '\n3 - distance'
        label += '\n4 - bottom dead center (theta, beta)'
        label += '\n5 - top dead center    (theta, beta)'
        label += '\n6 - user / motor inertia ratio'
        label += '\n7 - nominal working point (velocity, torque)'
        label += '\n6 - nominal working point (velocity, power)'
        result = (crank, rod, rocker, dist, (t_bdc, z_bdc), (t_tdc, z_tdc),
                  ratio, (vel, trq), (vel, powr), label)
        return result


class CrankSlider(_kinematic_one):
    """crank and slider mechanism - one degree of freedom
(crank + RPP dyad)

input, set 1: direct motion
    mover: law of motion of the mover
    crank: length of the crank

input, set 2: inverse motion
    follower: law of motion of the follower
    [crank]: length of the crank

input, any set
    [motor_inertia]: inertial property of the motor
    [gearbox_inertia]: inertial property of the gearbox
    [gear_ratio]: gear ratio between the motor and the mover
    [efficiency]: efficiency of the transmission
    [mover_inertia]: inertial property of the mover (including gear)
    [follower_inertia]: inertial property of the follower
    [external_load]: load acting on the follower

methods
    features(): get chain status description
    get_features(): returns a tuple with the features of the chain
    time(): returns the time axis
    mover(): returns the mover law of motion
    follower(): returns the follower law of motion
    user_inertia(): returns the inertial property seen at the motor
    velocity(): returns the motor velocity
    torque(): returns the motor torque
    torque_external(): returns the torque requirement for external load
    power(): returns the motor power
"""

    # class initialization
    def __init__(
        self,
        mover=None,
        follower=None,
        crank=None,
        motor_inertia=0.0,
        gearbox_inertia=0.0,
        gear_ratio=1.0,
        efficiency=1.0,
        mover_inertia=0.0,
        follower_inertia=0.0,
        external_load=0.0
        ):
        """private function: class initialization"""
        if follower is None and mover is None:
            return
        elif follower is None:
            if crank is None:
                return
            else:
                self._crank = crank
                self._mover = mover
                flag = self._assign(True)
                self._fol_pos = -self._crank * np.cos(self._mov_pos)
        else:
            if crank is None:
                self._crank = abs(max(follower.position())
                                  -min(follower.position())) / 2
                rotation = True
            else:
                self._crank = crank
                rotation = False
            self._follower = follower
            flag = self._assign(False)
            cost = -self._fol_pos / self._crank
            cost[cost > 1] = 1
            cost[cost < -1] = -1
            sint = np.sqrt(1 - cost**2)
            if rotation:
                sint[self._fol_vel < 0] *= -1
            self._mov_pos = _angle(sint, cost)
        self._geo_vel = self._crank * np.sin(self._mov_pos)
        self._geo_acc = self._crank * np.cos(self._mov_pos)
        self._gear = gear_ratio
        self._eta = efficiency
        self._mot_j = motor_inertia
        self._box_j = gearbox_inertia
        self._mov_j = mover_inertia
        self._fol_j = follower_inertia
        if isinstance(external_load, np.ndarray):
            self._ext_load = external_load
        else:
            self._ext_load = np.ones_like(self._time) * external_load
        self._mount(flag)
        self._t_bdc = 0
        self._z_bdc = 0
        self._t_tdc = np.pi
        self._z_tdc = 2 * self._crank

    # public methods
    def features(self):
        """get chain status description"""
        print('\ncrank and slider mechanism\ncrank + RPP\n')
        print(f'crank length: {self._crank*1000:5.1f} mm')
        print()
        print(f'bottom dead center: ({self._t_bdc*180/np.pi:5.1f} deg, '
              f'{self._z_bdc*1000:5.1f} mm)')
        print(f'top dead center:    ({self._t_tdc*180/np.pi:5.1f} deg, '
              f'{self._z_tdc*1000:5.1f} mm)')
        print()
        print(f'user / motor inertia ratio: {self._j_ratio:4.2f}')
        print()
        print(f'maximum motor velocity: {self._max_vel*30/np.pi:5.0f} rpm')
        print()
        print(f'maximum motor torque: {self._max_load:5.1f} N')
        print(f'nominal working point: ({self._mot_avg*30/np.pi:5.0f}'
              f' rpm, {self._mot_rms:5.1f} N m)')
        print()
        print(f'maximum power: {self._max_pow/1000:5.1f} kW')
        print(f'nominal working point: ({self._mot_avg*30/np.pi:5.0f}'
              f' rpm, {self._pow_rms/1000:5.1f} kW)')

    def get_features(self):
        """returns the features of the chian ([-1] for help)"""
        crank = self._crank
        t_bdc = self._t_bdc
        z_bdc = self._z_bdc
        t_tdc = self._t_tdc
        z_tdc = self._z_tdc
        ratio = self._j_ratio
        vel = self._mot_avg
        trq = self._mot_rms
        powr = self._pow_rms
        label = '0 - crank length'
        label += '\n1 - bottom dead center (theta, zeta)'
        label += '\n2 - top dead center    (theta, zeta)'
        label += '\n3 - user / motor inertia ratio'
        label += '\n4 - nominal working point (velocity, torque)'
        label += '\n5 - nominal working point (velocity, power)'
        result = (crank, (t_bdc, z_bdc), (t_tdc, z_tdc), ratio, (vel, trq),
                  (vel, powr), label)
        return result


class CrankRodSlider(_kinematic_one):
    """crank, rod  and slider mechanism - one degree of freedom
(crank + RRP dyad)

input, set 1: direct motion
    mover: law of motion of the mover
    rod: length of the connection rod
    crank: length of the crank

input, set 2: inverse motion
    follower: law of motion of the follower
    rod: length of the connection rod
    [crank]: length of the crank

input, any set
    [distance]: distance between the center of rotation of the crank and the
                line of movement of the slider [default = 0]
    [motor_inertia]: inertial property of the motor
    [gearbox_inertia]: inertial property of the gearbox
    [gear_ratio]: gear ratio between the motor and the mover
    [efficiency]: efficiency of the transmission
    [mover_inertia]: inertial property of the mover (including gear)
    [follower_inertia]: inertial property of the follower
    [external_load]: load acting on the follower

methods
    features(): get chain status description
    get_features(): returns a tuple with the features of the chain
    time(): returns the time axis
    mover(): returns the mover law of motion
    follower(): returns the follower law of motion
    user_inertia(): returns the inertial property seen at the motor
    velocity(): returns the motor velocity
    torque(): returns the motor torque
    torque_external(): returns the torque requirement for external load
    power(): returns the motor power
"""

    # class initialization
    def __init__(
        self,
        mover=None,
        follower=None,
        rod=None,
        crank=None,
        distance=0.0,
        motor_inertia=0.0,
        gearbox_inertia=0.0,
        gear_ratio=1.0,
        efficiency=1.0,
        mover_inertia=0.0,
        follower_inertia=0.0,
        external_load=0.0
        ):
        """private function: class initialization"""
        if follower is None and mover is None:
            return
        elif rod is None:
            return
        elif follower is None:
            if crank is None:
                return
            elif rod <= crank + abs(distance):
                return
            else:
                m = crank
                b = rod
                d = distance
                t = mover.position()
                z0 = (np.sqrt((m+b)**2 - d**2) + np.sqrt((m-b)**2 - d**2)) / 2
                z = -m*np.cos(t) + np.sqrt(b**2 - (m*np.sin(t)-d)**2) - z0
                self._mover = mover
                flag = self._assign(True)
                self._crank = m
                self._rod = b
                self._dist = d
                self._fol_pos = z.copy()
        else:
            b = rod
            d = distance
            if crank is None:
                s = max(follower.position()) - min(follower.position())
                m = np.sqrt((4*(b**2 - d**2)*s**2 - s**4) / (16*b**2 - 4*s**2))
                rotation = True
            else:
                m = crank
                rotation = False
            z0 = (np.sqrt((m+b)**2 - d**2) + np.sqrt((m-b)**2 - d**2)) / 2
            x = follower.position() + z0
            q = (b**2 - m**2 - d**2 - x**2) / (2*m)
            delta = x**2 + d**2 - q**2
            delta[delta < 0] = 0
            sint = -(q*d - x*np.sqrt(delta)) / (x**2 + d**2)
            if rotation:
                closure = follower.velocity() < 0
                sint[closure] = -((q[closure]*d + x[closure]
                                  *np.sqrt(delta[closure]))
                                  / (x[closure]**2 + d**2))
            cost = (q + d*sint) / x
            self._follower = follower
            flag = self._assign(False)
            self._crank = m
            self._rod = b
            self._dist = d
            self._mov_pos = _angle(sint, cost)
            t = self._mov_pos.copy()
        self._geo_vel = (m*np.sin(t) - m*np.cos(t)*(m*np.sin(t) - d)
                         /np.sqrt(b**2 - (m*np.sin(t) - d)**2))
        self._geo_acc = (m*np.cos(t) + (m*np.sin(t)*(2*m*np.sin(t) - d) - m**2)
                         /np.sqrt(b**2 - (m*np.sin(t) - d)**2)
                         - (m*np.cos(t)*(m*np.sin(t) - d))**2
                         /np.sqrt((b**2 - (m*np.sin(t) - d)**2)**3))
        self._gear = gear_ratio
        self._eta = efficiency
        self._mot_j = motor_inertia
        self._box_j = gearbox_inertia
        self._mov_j = mover_inertia
        self._fol_j = follower_inertia
        if isinstance(external_load, np.ndarray):
            self._ext_load = external_load
        else:
            self._ext_load = np.ones_like(self._time) * external_load
        self._mount(flag)
        self._t_bdc = 2*np.pi + np.arcsin(d / (m-b))
        self._z_bdc = (np.sqrt((m-b)**2 - d**2) - np.sqrt((m+b)**2 - d**2)) / 2
        self._t_tdc = np.pi - np.arcsin(d / (m+b))
        self._z_tdc = (np.sqrt((m+b)**2 - d**2) - np.sqrt((m-b)**2 - d**2)) / 2

    # public methods
    def features(self):
        """get chain status description"""
        print('\ncrank, rod and slider mechanism\ncrank + RRP\n')
        print(f'crank length: {self._crank*1000:5.1f} mm')
        print(f'rod length: {self._rod*1000:5.1f} mm')
        print(f'distance: {self._dist*1000:5.1f} mm')
        print()
        print(f'bottom dead center: ({self._t_bdc*180/np.pi:5.1f} deg, '
              f'{self._z_bdc*1000:5.1f} mm)')
        print(f'top dead center:    ({self._t_tdc*180/np.pi:5.1f} deg, '
              f'{self._z_tdc*1000:5.1f} mm)')
        print()
        print(f'user / motor inertia ratio: {self._j_ratio:4.2f}')
        print()
        print(f'maximum motor velocity: {self._max_vel*30/np.pi:5.0f} rpm')
        print()
        print(f'maximum motor torque: {self._max_load:5.1f} N')
        print(f'nominal working point: ({self._mot_avg*30/np.pi:5.0f}'
              f' rpm, {self._mot_rms:5.1f} N m)')
        print()
        print(f'maximum power: {self._max_pow/1000:5.1f} kW')
        print(f'nominal working point: ({self._mot_avg*30/np.pi:5.0f}'
              f' rpm, {self._pow_rms/1000:5.1f} kW)')

    def get_features(self):
        """returns the features of the chian ([-1] for help)"""
        crank = self._crank
        rod = self._rod
        dist = self._dist
        t_bdc = self._t_bdc
        z_bdc = self._z_bdc
        t_tdc = self._t_tdc
        z_tdc = self._z_tdc
        ratio = self._j_ratio
        vel = self._mot_avg
        trq = self._mot_rms
        powr = self._pow_rms
        label = '0 - crank length'
        label += '\n1 - rod length'
        label += '\n2 - distance'
        label += '\n3 - bottom dead center (theta, zeta)'
        label += '\n4 - top dead center    (theta, zeta)'
        label += '\n5 - user / motor inertia ratio'
        label += '\n6 - nominal working point (velocity, torque)'
        label += '\n7 - nominal working point (velocity, power)'
        result = (crank, rod, dist, (t_bdc, z_bdc), (t_tdc, z_tdc), ratio,
                  (vel, trq), (vel, powr), label)
        return result


class RockerSlider(_kinematic_one):
    """rocker and slider mechanism - one degree of freedom
(rocker + PRP dyad)

input, set 1: direct motion
    mover: law of motion of the mover

input, set 2: inverse motion
    follower: law of motion of the follower

input, any set
    disctance: distance between the center of rotation of the rocker
               and the line of movement of the slider
    [motor_inertia]: inertial property of the motor
    [gearbox_inertia]: inertial property of the gearbox
    [gear_ratio]: gear ratio between the motor and the mover
    [efficiency]: efficiency of the transmission
    [mover_inertia]: inertial property of the mover (including gear)
    [follower_inertia]: inertial property of the follower
    [external_load]: load acting on the follower

methods
    features(): get chain status description
    get_features(): returns a tuple with the features of the chain
    time(): returns the time axis
    mover(): returns the mover law of motion
    follower(): returns the follower law of motion
    user_inertia(): returns the inertial property seen at the motor
    velocity(): returns the motor velocity
    torque(): returns the motor torque
    torque_external(): returns the torque requirement for external load
    power(): returns the motor power
"""

    # class initialization
    def __init__(
        self,
        mover=None,
        follower=None,
        distance=None,
        motor_inertia=0.0,
        gearbox_inertia=0.0,
        gear_ratio=1.0,
        efficiency=1.0,
        mover_inertia=0.0,
        follower_inertia=0.0,
        external_load=0.0
        ):
        """private function: class initialization"""
        if distance is None:
            return
        elif follower is None and mover is None:
            return
        elif follower is None:
            if (min(mover.position()) <= -np.pi/2 
                    or max(mover.position()) >= np.pi/2):
                return
            else:
                self._dist = distance
                self._mover = mover
                flag = self._assign(True)
                self._fol_pos = self._dist * np.tan(self._mov_pos)
        else:
            self._dist = distance
            self._follower = follower
            flag = self._assign(False)
            self._mov_pos = np.arctan(self._fol_pos/self._dist)
        self._geo_vel = self._dist / np.cos(self._mov_pos)**2
        self._geo_acc = (2 * self._dist * np.tan(self._mov_pos)
                         / np.cos(self._mov_pos)**2)
        self._gear = gear_ratio
        self._eta = efficiency
        self._mot_j = motor_inertia
        self._box_j = gearbox_inertia
        self._mov_j = mover_inertia
        self._fol_j = follower_inertia
        if isinstance(external_load, np.ndarray):
            self._ext_load = external_load
        else:
            self._ext_load = np.ones_like(self._time) * external_load
        self._mount(flag)

    # public methods
    def features(self):
        """get chain status description"""
        print('\nrocker and slider mechanism\nrocker + PRP\n')
        print(f'distance: {self._dist*1000:5.1f} mm')
        print()
        print(f'user / motor inertia ratio: {self._j_ratio:4.2f}')
        print()
        print(f'maximum motor velocity: {self._max_vel*30/np.pi:5.0f} rpm')
        print()
        print(f'maximum motor torque: {self._max_load:5.1f} N')
        print(f'nominal working point: ({self._mot_avg*30/np.pi:5.0f}'
              f' rpm, {self._mot_rms:5.1f} N m)')
        print()
        print(f'maximum power: {self._max_pow/1000:5.1f} kW')
        print(f'nominal working point: ({self._mot_avg*30/np.pi:5.0f}'
              f' rpm, {self._pow_rms/1000:5.1f} kW)')

    def get_features(self):
        """returns the features of the chian ([-1] for help)"""
        dist = self._dist
        ratio = self._j_ratio
        vel = self._mot_avg
        trq = self._mot_rms
        powr = self._pow_rms
        label = '0 - distance'
        label += '\n1 - user / motor inertia ratio'
        label += '\n2 - nominal working point (velocity, torque)'
        label += '\n3 - nominal working point (velocity, power)'
        result = (dist, ratio, (vel, trq), (vel, powr), label)
        return result


class Screw(_kinematic_one):
    """screw and nut mechanism - one degree of freedom
(H joint)

input, set 1: direct motion
    mover: law of motion of the mover

input, set 2: inverse motion
    follower: law of motion of the follower

input, any set
    pitch: pitch of the screw
    [motor_inertia]: inertial property of the motor
    [gearbox_inertia]: inertial property of the gearbox
    [gear_ratio]: gear ratio between the motor and the mover
    [efficiency]: efficiency of the transmission
    [mover_inertia]: inertial property of the mover (including gear)
    [follower_inertia]: inertial property of the follower
    [external_load]: load acting on the follower

methods
    features(): get chain status description
    get_features(): returns a tuple with the features of the chain
    time(): returns the time axis
    mover(): returns the mover law of motion
    follower(): returns the follower law of motion
    user_inertia(): returns the inertial property seen at the motor
    velocity(): returns the motor velocity
    torque(): returns the motor torque
    torque_external(): returns the torque requirement for external load
    power(): returns the motor power
"""

    # class initialization
    def __init__(
        self,
        mover=None,
        follower=None,
        pitch=None,
        motor_inertia=0.0,
        gearbox_inertia=0.0,
        gear_ratio=1.0,
        efficiency=1.0,
        mover_inertia=0.0,
        follower_inertia=0.0,
        external_load=0.0
        ):
        """private function: class initialization"""
        if pitch is None:
            return
        elif follower is None and mover is None:
            return
        elif follower is None:
            self._pitch = pitch
            self._mover = mover
            flag = self._assign(True)
            self._fol_pos = self._pitch * self_mov_pos / (2*np.pi)
        else:
            self._pitch = pitch
            self._follower = follower
            flag = self._assign(False)
            self._mov_pos = 2 * np.pi * self._fol_pos / self._pitch
            self._mov_pos -= min(self._mov_pos)
        self._geo_vel = np.ones_like(self._mov_pos) * self._pitch / (2*np.pi)
        self._geo_acc = np.zeros_like(self._mov_pos)
        self._gear = gear_ratio
        self._eta = efficiency
        self._mot_j = motor_inertia
        self._box_j = gearbox_inertia
        self._mov_j = mover_inertia
        self._fol_j = follower_inertia
        if isinstance(external_load, np.ndarray):
            self._ext_load = external_load
        else:
            self._ext_load = np.ones_like(self._time) * external_load
        self._mount(flag)

    # public methods
    def features(self):
        """get chain status description"""
        print('\nscrew and nut mechanism\nH joint\n')
        print(f'pitch: {self._pitch*1000:5.1f} mm')
        print()
        print(f'user / motor inertia ratio: {self._j_ratio:4.2f}')
        print()
        print(f'maximum motor velocity: {self._max_vel*30/np.pi:5.0f} rpm')
        print()
        print(f'maximum motor torque: {self._max_load:5.1f} N')
        print(f'nominal working point: ({self._mot_avg*30/np.pi:5.0f}'
              f' rpm, {self._mot_rms:5.1f} N m)')
        print()
        print(f'maximum power: {self._max_pow/1000:5.1f} kW')
        print(f'nominal working point: ({self._mot_avg*30/np.pi:5.0f}'
              f' rpm, {self._pow_rms/1000:5.1f} kW)')

    def get_features(self):
        """returns the features of the chian ([-1] for help)"""
        pitch = self._pitch
        ratio = self._j_ratio
        vel = self._mot_avg
        trq = self._mot_rms
        powr = self._pow_rms
        label = '0 - screw pitch'
        label += '\n1 - user / motor inertia ratio'
        label += '\n2 - nominal working point (velocity, torque)'
        label += '\n3 - nominal working point (velocity, power)'
        result = (pitch, ratio, (vel, trq), (vel, powr), label)
        return result


class OpenDyad(_kinematic_two):
    """"crank and crank mechanism - two degrees of freedom

input, set 1: direct motion 
    mover_1: law of motion of the first crank
    mover_2: law of motion of the second crank

input, set 2: inverse motion
    follower_x: law of motion of follower abscissa
    follower_y: law of motion of follwer ordinate
    closure: array of the closures to be used for inversion

input, any set
    crank_1: length of the first crank
    crank_2: length of the second crank

methods:
    time(): returns the time axis
    mover(): returns a tuple with the movers laws of motion
    follower(): returns a tuple with the follower laws of motion
"""

    # class initialization
    def __init__(
        self,
        mover_1=None,
        mover_2=None,
        follower_x=None,
        follower_y=None,
        closure=None,
        crank_1=None,
        crank_2=None
        ):
        """private function: class initialization"""
        if crank_1 is None or crank_2 is None:
            return
        if follower_x is None and follower_y is None:
            if mover_1 is None or mover_2 is None:
                return
            self._mov1 = mover_1
            self._mov2 = mover_2
            flag = self._assign(True)
            self._c1 = crank_1
            self._c2 = crank_2
            self._fol1_pos = (self._c1*np.cos(self._mov1_pos)
                              + self._c2*np.cos(self._mov2_pos))
            self._fol2_pos = (self._c1*np.sin(self._mov1_pos)
                              + self._c2*np.sin(self._mov2_pos))
        elif mover_1 is None and mover_2 is None:
            if follower_x is None or follower_y is None:
                return
            d = np.sqrt(follower_x.position()**2 + follower_y.position()**2)
            if max(d) > crank_1 + crank_2 or min(d) < abs(crank_1 - crank_2):
                return
            self._fol1 = follower_x
            self._fol2 = follower_y
            flag = self._assign(False)
            self._c1 = crank_1
            self._c2 = crank_2
            if closure is None:
                closure = np.empty_like(self._time)
                closure[:] = True
            lam = np.sqrt(self._fol1_pos**2 + self._fol2_pos**2)
            sind = self._fol2_pos / lam
            cosd = self._fol1_pos / lam
            delta = _angle(sind, cosd)
            gamma = np.acos((lam**2 + self._c1**2 - self._c2**2)
                            /(2*self._c1*lam))
            self._mov1_pos = delta
            self._mov1_pos[closure] += gamma[closure]
            self._mov1_pos[~closure] -= gamma[~closure]
            sint = (self._fol2_pos - self._c1*np.sin(self._mov1_pos))/self._c2
            cost = (self._fol1_pos - self._c1*np.cos(self._mov1_pos))/self._c2
            self._mov2_pos = _angle(sint, cost)
        else:
            return
        self._fol1_geo1 = -self._c1 * np.sin(self._mov1_pos)
        self._fol1_geo2 = -self._c2 * np.sin(self._mov2_pos)
        self._fol2_geo1 = self._c1 * np.cos(self._mov1_pos)
        self._fol2_geo2 = self._c2 * np.cos(self._mov2_pos)
        if flag:
            # kinematic composition, direct
            self._fol1_vel = (self._fol1_geo1*self._mov1_vel
                              + self._fol1_geo2*self._mov2_vel)
            self._fol1_acc = np.gradient(
                self._fol1_vel,
                self._time,
                edge_order=2
                )
            self._fol2_vel = (self._fol2_geo1*self._mov1_vel
                              + self._fol2_geo2*self._mov2_vel)
            self._fol2_acc = np.gradient(
                self._fol2_vel,
                self._time,
                edge_order=2
                )
        else:
            # inverse geometric derivatives
            det = (self._fol1_geo1*self.fol2_geo2
                   + self._fol1_geo2*self._fol2_geo1)
            self._mov1_geo1 = self._fol2_geo2 / det
            self._mov1_geo2 = - self._fol1_geo2 / det
            self._mov2_geo1 = - self._fol2_geo1 / det
            self._mov2_geo2 = self._fol1_geo1 / det
            # kinematic composition, inverse
            self._mov1_vel = (self._mov1_geo1*self._fol1_vel
                              + self._mov1_geo2*self._fol2_vel)
            self._mov1_acc = np.gradient(
                self._mov1_acc,
                self._time,
                edge_order=2
                )
            self._mov2_vel = (self._mov2_geo1*self._fol1_vel
                              + self._mov2_geo2*self._fol2_vel)
            self._mov2_acc = np.gradient(
                self._mov2_acc,
                self._time,
                edge_order=2
                )

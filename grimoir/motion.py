#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Motion Module
a collection of laws of motion and related tools

NOTE: Motion Module deals with pure mathematics, units of measure are
      not considered in this module

classes for strokes (position-to-position laws of motion):
    ConstVelStroke: .. S0 constant velocity stroke
    TriVelStroke: .... S1 triangular velocity stroke
    TrapVelStroke: ... S1 trapezoidal velocity stroke
    Poly3Stroke: ..... S1 polynomial, degree 3 stroke
    Poly5Stroke: ..... S2 polynomial, degree 5 stroke
    SineStroke: ...... S2 cycloid stroke
    ModSineStroke: ... S2 modified cycloid stroke
    Poly7Stroke: ..... S3 polynomial, degree 7 stroke
    SineJStroke: ..... S3 cycloid jerk stroke
    ModSineJStroke: .. S3 modified cycloid jerk stroke

classes for ramps (velocity-to-velocity laws of motion):
    ConstAccRamp: .... R0 contant acceleration ramp
    Poly3Ramp: ....... R1 polynomial, degree 3 ramp
    SineRamp: ........ R1 cycoid ramp
    Poly5Ramp: ....... R2 polynomial, degree 5 ramp
    SineJRamp: ....... R2 cycloid jerk ramp
    ModSineJRamp: .... R2 modified cycloid jerk ramp

classes for impulses (acceleration-to-acceleration laws of motion):
    ConstImpulse: .... I0 constant jerk impulse
    PolyImpulse: ..... I1 polynomial impulse
    SineImpulse: ..... I1 cycloid impulse

class for polynomial spline law of motion:
    PolySpline: ...... polynomial spline law of motion

classes for law of motion manipulation:
    Compose: ......... gives a law of motion another law as master
    Import: .......... builds a law of motion importing value lists
    Shift: ........... shifts the law of motion along the slave axis
    Slice: ........... slices one law of motion along master axis
    Stitch: .......... join several laws of motion into one
"""

__version__ = '0.1.00'
__author__ = 'Luca Zambonelli'
__copyright__ = '2022'
__license__ = 'GPL'
__maintainer__ = 'Luca Zambonelli'
__email__ = 'luca.zambonelli@gmail.com'
__status__ = 'Development'

import numpy as np

# callable variable for upper level package
Motion = True

# private functions to deal with master normalization
def _normtime(t, tgo, tng):
    """private function: master normalization routine"""
    return (t-tgo) / (tng-tgo)

def _time(t_eq, tgo, tng):
    """private function: master de-normalization routine"""
    return tgo + t_eq*(tng-tgo)


# private class to deal with strokes
class _stroke:
    """private class to deal with strokes"""

    # private functions
    def _p(self, t_eq, x_eq):
        """position routine"""
        if isinstance(t_eq, np.ndarray):
            t_in = (t_eq >= 0) & (t_eq <= 1)
            x_in = np.empty_like(t_eq)
            x_in[:] = np.NaN
            if isinstance(x_eq, np.ndarray):
                x_in[t_in] = x_eq[t_in]
            else:
                x_in[t_in] = x_eq
        else:
            if 0 <= t_eq <= 1:
                x_in = x_eq
            else:
                x_in = np.NaN
        x = self._xgo + x_in * (self._xng-self._xgo)
        return x

    def _v(self, t_eq, v_eq):
        """velocity routine"""
        if isinstance(t_eq, np.ndarray):
            t_in = (t_eq >= 0) & (t_eq <= 1)
            v_in = np.empty_like(t_eq)
            v_in[:] = np.NaN
            if isinstance(v_eq, np.ndarray):
                v_in[t_in] = v_eq[t_in]
            else:
                v_in[t_in] = v_eq
        else:
            if 0 <= t_eq <= 1:
                v_in = v_eq
            else:
                v_in = np.NaN
        v = v_in * (self._xng-self._xgo) / (self._tng-self._tgo)
        return v

    def _a(self, t_eq, a_eq):
        """acceleration routine"""
        if isinstance(t_eq, np.ndarray):
            t_in = (t_eq >= 0) & (t_eq <= 1)
            a_in = np.empty_like(t_eq)
            a_in[:] = np.NaN
            if isinstance(a_eq, np.ndarray):
                a_in[t_in] = a_eq[t_in]
            else:
                a_in[t_in] = a_eq
        else:
            if 0 <= t_eq <= 1:
                a_in = a_eq
            else:
                a_in = np.NaN
        a = a_in * (self._xng-self._xgo) / (self._tng-self._tgo)**2
        return a

    def _j(self, t_eq, j_eq):
        """jerk routine"""
        if isinstance(t_eq, np.ndarray):
            t_in = (t_eq >= 0) & (t_eq <= 1)
            j_in = np.empty_like(t_eq)
            j_in[:] = np.NaN
            if isinstance(j_eq, np.ndarray):
                j_in[t_in] = j_eq[t_in]
            else:
                j_in[t_in] = j_eq
        else:
            if 0 <= t_eq <= 1:
                j_in = j_eq
            else:
                j_in = np.NaN
        j = j_in * (self._xng-self._xgo) / (self._tng-self._tgo)**3
        return j

    # class initialization
    def __init__(self, t=np.linspace(0., 1., 257), xgo=0., xng=1.):
        """private function: class initialization"""
        self._tgo = t[0]
        self._tng = t[-1]
        self._master = _normtime(t, self._tgo, self._tng)
        self._xgo = xgo
        self._xng = xng
        self._pos = self._p(self._master, self._x_eq(self._master))
        self._vel = self._v(self._master, self._v_eq(self._master))
        self._acc = self._a(self._master, self._a_eq(self._master))
        self._jrk = self._j(self._master, self._j_eq(self._master))

    # public methods
    def master(self):
        """returns the master axis"""
        return _time(self._master, self._tgo, self._tng)

    def position(self, t=None):
        """returns the slave position"""
        if t is None:
            return self._pos
        else:
            t_eq = _normtime(t, self._tgo, self._tng)
            return self._p(t_eq, self._x_eq(t_eq))

    def velocity(self, t=None):
        """returns the slave velocity"""
        if t is None:
            return self._vel
        else:
            t_eq = _normtime(t, self._tgo, self._tng)
            return self._v(t_eq, self._v_eq(t_eq))

    def acceleration(self, t=None):
        """returns the slave acceleration"""
        if t is None:
            return self._acc
        else:
            t_eq = _normtime(t, self._tgo, self._tng)
            return self._a(t_eq, self._a_eq(t_eq))

    def jerk(self, t=None):
        """returns the slave acceleration"""
        if t is None:
            return self._jrk
        else:
            t_eq = _normtime(t, self._tgo, self._tng)
            return self._j(t_eq, self._j_eq(t_eq))


# private class to deal with ramps
class _ramp:
    """private class to deal with ramps"""

    # private functions
    def _p(self, t_eq, x_eq):
        """position routine"""
        if isinstance(t_eq, np.ndarray):
            t_in = (t_eq >= 0) & (t_eq <= 1)
            x_in = np.empty_like(t_eq)
            x_in[:] = np.NaN
            if isinstance(x_eq, np.ndarray):
                x_in[t_in] = x_eq[t_in]
            else:
                x_in[t_in] = x_eq
        else:
            if 0 <= t_eq <= 1:
                x_in = x_eq
            else:
                x_in = np.NaN
        x = (self._xgo + self._vgo*t_eq*(self._tng-self._tgo)
             + x_in*(self._vng-self._vgo)*(self._tng-self._tgo))
        return x

    def _v(self, t_eq, v_eq):
        """velocity routine"""
        if isinstance(t_eq, np.ndarray):
            t_in = (t_eq >= 0) & (t_eq <= 1)
            v_in = np.empty_like(t_eq)
            v_in[:] = np.NaN
            if isinstance(v_eq, np.ndarray):
                v_in[t_in] = v_eq[t_in]
            else:
                v_in[t_in] = v_eq
        else:
            if 0 <= t_eq <= 1:
                v_in = v_eq
            else:
                v_in = np.NaN
        v = self._vgo + v_in*(self._vng-self._vgo)
        return v

    def _a(self, t_eq, a_eq):
        """acceleration routine"""
        if isinstance(t_eq, np.ndarray):
            t_in = (t_eq >= 0) & (t_eq <= 1)
            a_in = np.empty_like(t_eq)
            a_in[:] = np.NaN
            if isinstance(a_eq, np.ndarray):
                a_in[t_in] = a_eq[t_in]
            else:
                a_in[t_in] = a_eq
        else:
            if 0 <= t_eq <= 1:
                a_in = a_eq
            else:
                a_in = np.NaN
        a = a_in * (self._vng-self._vgo) / (self._tng-self._tgo)
        return a

    def _j(self, t_eq, j_eq):
        """jerk routine"""
        if isinstance(t_eq, np.ndarray):
            t_in = (t_eq >= 0) & (t_eq <= 1)
            j_in = np.empty_like(t_eq)
            j_in[:] = np.NaN
            if isinstance(j_eq, np.ndarray):
                j_in[t_in] = j_eq[t_in]
            else:
                j_in[t_in] = j_eq
        else:
            if 0 <= t_eq <= 1:
                j_in = j_eq
            else:
                j_in = np.NaN
        j = j_in * (self._vng-self._vgo) / (self._tng-self._tgo)**2
        return j

    # class initialization
    def __init__(self, t=np.linspace(0., 1., 257), xgo=0., vgo=0., vng=1.):
        """private function: class initialization"""
        self._tgo = t[0]
        self._tng = t[-1]
        self._master = _normtime(t, self._tgo, self._tng)
        self._xgo = xgo
        self._vgo = vgo
        self._vng = vng
        self._pos = self._p(self._master, self._x_eq(self._master))
        self._vel = self._v(self._master, self._v_eq(self._master))
        self._acc = self._a(self._master, self._a_eq(self._master))
        self._jrk = self._j(self._master, self._j_eq(self._master))

    #public methods
    def master(self):
        """returns the master axis"""
        return _time(self._master, self._tgo, self._tng)

    def position(self, t=None):
        """returns the slave position"""
        if t is None:
            return self._pos
        else:
            t_eq = _normtime(t, self._tgo, self._tng)
            return self._p(t_eq, self._x_eq(t_eq))

    def velocity(self, t=None):
        """returns the slave velocity"""
        if t is None:
            return self._vel
        else:
            t_eq = _normtime(t, self._tgo, self._tng)
            return self._v(t_eq, self._v_eq(t_eq))

    def acceleration(self, t=None):
        """returns the slave velocity"""
        if t is None:
            return self._acc
        else:
            t_eq = _normtime(t, self._tgo, self._tng)
            return self._a(t_eq, self._a_eq(t_eq))

    def jerk(self, t=None):
        """returns the slave velocity"""
        if t is None:
            return self._jrk
        else:
            t_eq = _normtime(t, self._tgo, self._tng)
            return self._j(t_eq, self._j_eq(t_eq))


# private class to deal with impulses
class _impulse:
    """private class to deal with impulses"""

    # private functions
    def _p(self, t_eq, x_eq):
        """position routine"""
        if isinstance(t_eq, np.ndarray):
            t_in = (t_eq >= 0) & (t_eq <= 1)
            x_in = np.empty_like(t_eq)
            x_in[:] = np.NaN
            if isinstance(x_eq, np.ndarray):
                x_in[t_in] = x_eq[t_in]
            else:
                x_in[t_in] = x_eq
        else:
            if 0 <= t_eq <= 1:
                x_in = x_eq
            else:
                x_in = np.NaN
        x = (self._xgo + self._vgo*t_eq*(self._tng-self._tgo)
             + self._ago*((t_eq**2)/2)*(self._tng-self._tgo)**2
             + x_in*(self._ang-self._ago)*(self._tng-self._tgo)**2)
        return x

    def _v(self, t_eq, v_eq):
        """velocity routine"""
        if isinstance(t_eq, np.ndarray):
            t_in = (t_eq >= 0) & (t_eq <= 1)
            v_in = np.empty_like(t_eq)
            v_in[:] = np.NaN
            if isinstance(v_eq, np.ndarray):
                v_in[t_in] = v_eq[t_in]
            else:
                v_in[t_in] = v_eq
        else:
            if 0 <= t_eq <= 1:
                v_in = v_eq
            else:
                v_in = np.NaN
        v = (self._vgo + self._ago*t_eq*(self._tng-self._tgo)
             + v_in*(self._ang-self._ago)*(self._tng-self._tgo))
        return v

    def _a(self, t_eq, a_eq):
        """acceleration routine"""
        if isinstance(t_eq, np.ndarray):
            t_in = (t_eq >= 0) & (t_eq <= 1)
            a_in = np.empty_like(t_eq)
            a_in[:] = np.NaN
            if isinstance(a_eq, np.ndarray):
                a_in[t_in] = a_eq[t_in]
            else:
                a_in[t_in] = a_eq
        else:
            if 0 <= t_eq <= 1:
                a_in = a_eq
            else:
                a_in = np.NaN
        a = self._ago + a_in*(self._ang-self._ago)
        return a

    def _j(self, t_eq, j_eq):
        """jerk routine"""
        if isinstance(t_eq, np.ndarray):
            t_in = (t_eq >= 0) & (t_eq <= 1)
            j_in = np.empty_like(t_eq)
            j_in[:] = np.NaN
            if isinstance(j_eq, np.ndarray):
                j_in[t_in] = j_eq[t_in]
            else:
                j_in[t_in] = j_eq
        else:
            if 0 <= t_eq <= 1:
                j_in = j_eq
            else:
                j_in = np.NaN
        j = j_in * (self._ang-self._ago) / (self._tng-self._tgo)
        return j

    # class initialization
    def __init__(self, t=np.linspace(0., 1., 257), xgo=0., vgo=0., ago=0.,
                 ang=1.):
        """private function: class initialization"""
        self._tgo = t[0]
        self._tng = t[- 1]
        self._master = _normtime(t, self._tgo, self._tng)
        self._xgo = xgo
        self._vgo = vgo
        self._ago = ago
        self._ang = ang
        self._pos = self._p(self._master, self._x_eq(self._master))
        self._vel = self._v(self._master, self._v_eq(self._master))
        self._acc = self._a(self._master, self._a_eq(self._master))
        self._jrk = self._j(self._master, self._j_eq(self._master))

    # public methods
    def master(self):
        """returns the master axis"""
        return _time(self._master, self._tgo, self._tng)

    def position(self, t=None):
        """returns the slave position"""
        if t is None:
            return self._pos
        else:
            t_eq = _normtime(t, self._tgo, self._tng)
            return self._p(t_eq, self._x_eq(t_eq))

    def velocity(self, t=None):
        """returns the slave velocity"""
        if t is None:
            return self._vel
        else:
            t_eq = _normtime(t, self._tgo, self._tng)
            return self._v(t_eq, self._v_eq(t_eq))

    def acceleration(self, t=None):
        """returns the slave acceleration"""
        if t is None:
            return self._acc
        else:
            t_eq = _normtime(t, self._tgo, self._tng)
            return self._a(t_eq, self._a_eq(t_eq))

    def jerk(self, t=None):
        """returns the slave acceleration"""
        if t is None:
            return self._jrk
        else:
            t_eq = _normtime(t, self._tgo, self._tng)
            return self._j(t_eq, self._j_eq(t_eq))


# classes for strokes
class ConstVelStroke(_stroke):
    """S0 constant velocity stroke

inputs
    t: master axis
    xgo: slave at the beginning of master axis
    xng: slave at the end of master axis

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_eq = t_eq
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_eq = 1
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_eq = 0
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_eq = 0
        return j_eq


class TriVelStroke(_stroke):
    """S1 triangular velocity stroke

inputs
    t: master axis
    xgo: slave at the beginning of master axis
    xng: slave at the end of master axis

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_e1 = 2 * t_eq**2
        x_e2 = -2*t_eq**2 + 4*t_eq - 1
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/2
            t_i2 = t_eq > 1/2
            x_eq = np.empty_like(t_eq)
            x_eq[t_i1] = x_e1[t_i1]
            x_eq[t_i2] = x_e2[t_i2]
        else:
            if t_eq <= 1/2:
                x_eq = x_e1
            else:
                x_eq = x_e2
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_e1 = 4 * t_eq
        v_e2 = -4*t_eq + 4
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/2
            t_i2 = t_eq > 1/2
            v_eq = np.empty_like(t_eq)
            v_eq[t_i1] = v_e1[t_i1]
            v_eq[t_i2] = v_e2[t_i2]
        else:
            if t_eq <= 1/2:
                v_eq = v_e1
            else:
                v_eq = v_e2
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_e1 = 4
        a_e2 = -4
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/2
            t_i2 = t_eq > 1/2
            a_eq = np.empty_like(t_eq)
            a_eq[t_i1] = a_e1
            a_eq[t_i2] = a_e2
        else:
            if t_eq <= 1/2:
                a_eq = a_e1
            else:
                a_eq = a_e2
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_eq = 0
        return j_eq


class TrapVelStroke(_stroke):
    """S1 trapezoidal velocity stroke

inputs
    t: master axis
    xgo: slave at the beginning of master axis
    xng: slave at the end of master axis

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_e1 = (9/4)*t_eq**2
        x_e2 = (3/2)*t_eq - 1/4
        x_e3 = -(9/4)*t_eq**2 + (9/2)*t_eq - 5/4
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/3
            t_i2 = (t_eq > 1/3) & (t_eq <= 2/3)
            t_i3 = t_eq > 2/3
            x_eq = np.empty_like(t_eq)
            x_eq[t_i1] = x_e1[t_i1]
            x_eq[t_i2] = x_e2[t_i2]
            x_eq[t_i3] = x_e3[t_i3]
        else:
            if t_eq <= 1/3:
                x_eq = x_e1
            elif t_eq <= 2/3:
                x_eq = x_e2
            else:
                x_eq = x_e3
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_e1 = (9/2)*t_eq
        v_e2 = 3/2
        v_e3 = -(9/2)*t_eq + 9/2
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/3
            t_i2 = (t_eq > 1/3) & (t_eq <= 2/3)
            t_i3 = t_eq > 2/3
            v_eq = np.empty_like(t_eq)
            v_eq[t_i1] = v_e1[t_i1]
            v_eq[t_i2] = v_e2
            v_eq[t_i3] = v_e3[t_i3]
        else:
            if t_eq <= 1/3:
                v_eq = v_e1
            elif t_eq <= 2/3:
                v_eq = v_e2
            else:
                v_eq = v_e3
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_e1 = 9/2
        a_e2 = 0
        a_e3 = -9/2
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/3
            t_i2 = (t_eq > 1/3) & (t_eq <= 2/3)
            t_i3 = t_eq > 2/3
            a_eq = np.empty_like(t_eq)
            a_eq[t_i1] = a_e1
            a_eq[t_i2] = a_e2
            a_eq[t_i3] = a_e3
        else:
            if t_eq <= 1/3:
                a_eq = a_e1
            elif t_eq <= 2/3:
                a_eq = a_e2
            else:
                a_eq = a_e3
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_eq = 0
        return j_eq


class Poly3Stroke(_stroke):
    """S1 polynomial, degree 3 stroke

inputs
    t: master axis
    xgo: slave at the beginning of master axis
    xng: slave at the end of master axis

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_eq = -2*t_eq**3 + 3*t_eq**2
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_eq = -6*t_eq**2 + 6*t_eq
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_eq = -12*t_eq + 6
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_eq = -12
        return j_eq


class Poly5Stroke(_stroke):
    """S2 polynomial, degree 5 stroke

inputs
    t: master axis
    xgo: slave at the beginning of master axis
    xng: slave at the end of master axis

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_eq = 6*t_eq**5 - 15*t_eq**4 + 10*t_eq**3
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_eq = 30*t_eq**4 - 60*t_eq**3 + 30*t_eq**2
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_eq = 120*t_eq**3 - 180*t_eq**2 + 60*t_eq
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_eq = 360*t_eq**2 - 360*t_eq + 60
        return j_eq


class SineStroke(_stroke):
    """S2 cycloid stroke

inputs
    t: master axis
    xgo: slave at the beginning of master axis
    xng: slave at the end of master axis

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_eq = -np.sin(2*np.pi*t_eq)/(2*np.pi) + t_eq
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_eq = -np.cos(2*np.pi*t_eq) + 1
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_eq = 2*np.pi*np.sin(2*np.pi*t_eq)
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_eq = 4*np.pi**2*np.cos(2*np.pi*t_eq)
        return j_eq


class ModSineStroke(_stroke):
    """S2 modified cycloid stroke

inputs
    t: master axis
    xgo: slave at the beginning of master axis
    xng: slave at the end of master axis

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_e1 = -((np.sin(4*np.pi*t_eq) - 4*np.pi*t_eq)
                 / (4 * (4+np.pi)))
        x_e2 = -((9*np.sin((4*np.pi/3)*t_eq + np.pi/3) - 4*np.pi*t_eq - 8)
                 / (4 * (4+np.pi)))
        x_e3 = -((np.sin(4*np.pi*t_eq)  - 4*np.pi*t_eq - 16)
                 / (4 * (4+np.pi)))
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/8
            t_i2 = (t_eq > 1/8) & (t_eq <= 7/8)
            t_i3 = t_eq > 7/8
            x_eq = np.empty_like(t_eq)
            x_eq[t_i1] = x_e1[t_i1]
            x_eq[t_i2] = x_e2[t_i2]
            x_eq[t_i3] = x_e3[t_i3]
        else:
            if t_eq <= 1/8:
                x_eq = x_e1
            elif t_eq <= 7/8:
                x_eq = x_e2
            else:
                x_eq = x_e3
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_e1 = -((np.pi*np.cos(4*np.pi*t_eq) - np.pi)
                 / (4+np.pi))
        v_e2 = -((3*np.pi*np.cos((4*np.pi/3)*t_eq + np.pi/3) - np.pi)
                 / (4+np.pi))
        v_e3 = -((np.pi*np.cos(4*np.pi*t_eq) - np.pi)
                 / (4+np.pi))
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/8
            t_i2 = (t_eq > 1/8) & (t_eq <= 7/8)
            t_i3 = t_eq > 7/8
            v_eq = np.empty_like(t_eq)
            v_eq[t_i1] = v_e1[t_i1]
            v_eq[t_i2] = v_e2[t_i2]
            v_eq[t_i3] = v_e3[t_i3]
        else:
            if t_eq <= 1/8:
                v_eq = v_e1
            elif t_eq <= 7/8:
                v_eq = v_e2
            else:
                v_eq = v_e3
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_e1 = (4*np.pi**2*np.sin(4*np.pi*t_eq)) / (4+np.pi)
        a_e2 = (4*np.pi**2*np.sin((4*np.pi/3)*t_eq + np.pi/3)) / (4+np.pi)
        a_e3 = (4*np.pi**2*np.sin(4*np.pi*t_eq)) / (4+np.pi)
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/8
            t_i2 = (t_eq > 1/8) & (t_eq <= 7/8)
            t_i3 = t_eq > 7/8
            a_eq = np.empty_like(t_eq)
            a_eq[t_i1] = a_e1[t_i1]
            a_eq[t_i2] = a_e2[t_i2]
            a_eq[t_i3] = a_e3[t_i3]
        else:
            if t_eq <= 1/8:
                a_eq = a_e1
            elif t_eq <= 7/8:
                a_eq = a_e2
            else:
                a_eq = a_e3
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_e1 = (16*np.pi**3*np.cos(4*np.pi*t_eq)) / (4+np.pi)
        j_e2 = ((16*np.pi**3*np.cos((4*np.pi/3)*t_eq + np.pi/3))
                / (3 * (4+np.pi)))
        j_e3 = (16*np.pi**3*np.cos(4*np.pi*t_eq)) / (4+np.pi)
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/8
            t_i2 = (t_eq > 1/8) & (t_eq <= 7/8)
            t_i3 = t_eq > 7/8
            j_eq = np.empty_like(t_eq)
            j_eq[t_i1] = j_e1[t_i1]
            j_eq[t_i2] = j_e2[t_i2]
            j_eq[t_i3] = j_e3[t_i3]
        else:
            if t_eq <= 1/8:
                j_eq = j_e1
            elif t_eq <= 7/8:
                j_eq = j_e2
            else:
                j_eq = j_e3
        return j_eq


class Poly7Stroke(_stroke):
    """S3 polynomial, degree 7 stroke

inputs
    t: master axis
    xgo: slave at the beginning of master axis
    xng: slave at the end of master axis

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_eq = -20*t_eq**7 + 70*t_eq**6 - 84*t_eq**5 + 35*t_eq**4
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_eq = -140*t_eq**6 + 420*t_eq**5 - 420*t_eq**4 + 140*t_eq**3
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_eq = -840*t_eq**5 + 2100*t_eq**4 - 1680*t_eq**3 + 420*t_eq**2
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_eq = -4200*t_eq**4 + 8400*t_eq**3 - 5040*t_eq**2 + 840*t_eq
        return j_eq


class SineJStroke(_stroke):
    """S3 cycloid jerk stroke

inputs
    t: master axis
    xgo: slave at the beginning of master axis
    xng: slave at the end of master axis

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_e1 = ((np.cos(4*np.pi*t_eq) + 8*np.pi**2*t_eq**2 - 1)
                / (12 + 3*np.pi**2))
        x_e2 = -((16*np.sin(2*np.pi*t_eq) - 8*np.pi**2*t_eq - 12 +np.pi**2)
                 / (2 * (12 + 3*np.pi**2)))
        x_e3 = -((np.cos(4*np.pi*t_eq) +8*np.pi**2*t_eq**2 - 16*np.pi**2*t_eq
                 - 13 + 5*np.pi**2) / (12 + 3*np.pi**2))
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/4
            t_i2 = (t_eq > 1/4) & (t_eq <= 3/4)
            t_i3 = t_eq > 3/4
            x_eq = np.empty_like(t_eq)
            x_eq[t_i1] = x_e1[t_i1]
            x_eq[t_i2] = x_e2[t_i2]
            x_eq[t_i3] = x_e3[t_i3]
        else:
            if t_eq <= 1/4:
                x_eq = x_e1
            elif t_eq <= 3/4:
                x_eq = x_e2
            else:
                x_eq = x_e3
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_e1 = -((4*np.pi*np.sin(4*np.pi*t_eq) - 16*np.pi**2*t_eq)
                 / (12 + 3*np.pi**2))
        v_e2 = -(16*np.pi*np.cos(2*np.pi*t_eq) -4*np.pi**2) / (12 + 3*np.pi**2)
        v_e3 = ((4*np.pi*np.sin(4*np.pi*t_eq) - 16*np.pi**2*t_eq + 16*np.pi**2)
                / (12 + 3*np.pi**2))
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/4
            t_i2 = (t_eq > 1/4) & (t_eq <= 3/4)
            t_i3 = t_eq > 3/4
            v_eq = np.empty_like(t_eq)
            v_eq[t_i1] = v_e1[t_i1]
            v_eq[t_i2] = v_e2[t_i2]
            v_eq[t_i3] = v_e3[t_i3]
        else:
            if t_eq <= 1/4:
                v_eq = v_e1
            elif t_eq <= 3/4:
                v_eq = v_e2
            else:
                v_eq = v_e3
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_e1 = -((16*np.pi**2*np.cos(4*np.pi*t_eq) - 16*np.pi**2)
                 / (12 + 3*np.pi**2))
        a_e2 = (32*np.pi**2*np.sin(2*np.pi*t_eq)) / (12 + 3*np.pi**2)
        a_e3 = ((16*np.pi**2*np.cos(4*np.pi*t_eq) - 16*np.pi**2)
                / (12 + 3*np.pi**2))
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/4
            t_i2 = (t_eq > 1/4) & (t_eq <= 3/4)
            t_i3 = t_eq > 3/4
            a_eq = np.empty_like(t_eq)
            a_eq[t_i1] = a_e1[t_i1]
            a_eq[t_i2] = a_e2[t_i2]
            a_eq[t_i3] = a_e3[t_i3]
        else:
            if t_eq <= 1/4:
                a_eq = a_e1
            elif t_eq <= 3/4:
                a_eq = a_e2
            else:
                a_eq = a_e3
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_e1 = (64*np.pi**3*np.sin(4*np.pi*t_eq)) / (12 + 3*np.pi**2)
        j_e2 = (64*np.pi**3*np.cos(2*np.pi*t_eq)) / (12 + 3*np.pi**2)
        j_e3 = -(64*np.pi**3*np.sin(4*np.pi*t_eq)) / (12 + 3*np.pi**2)
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/4
            t_i2 = (t_eq > 1/4) & (t_eq <= 3/4)
            t_i3 = t_eq > 3/4
            j_eq = np.empty_like(t_eq)
            j_eq[t_i1] = j_e1[t_i1]
            j_eq[t_i2] = j_e2[t_i2]
            j_eq[t_i3] = j_e3[t_i3]
        else:
            if t_eq <= 1/4:
                j_eq = j_e1
            elif t_eq <= 3/4:
                j_eq = j_e2
            else:
                j_eq = j_e3
        return j_eq


class ModSineJStroke(_stroke):
    """S3 modified cycloid jerk stroke

inputs
    t: master axis
    xgo: slave at the beginning of master axis
    xng: slave at the end of master axis

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_e1 = ((np.cos(6*np.pi*t_eq) + 18*np.pi**2*t_eq**2 - 1)
                / (12 + 11*np.pi**2))
        x_e2 = ((72*np.pi**2*t_eq**2 - 12*np.pi**2* t_eq - 4 + np.pi**2)
                / (2 * (12 + 11*np.pi**2)))
        x_e3 = ((16*np.cos(3*np.pi*t_eq) + 36*np.pi**2*t_eq + 12 - 7*np.pi**2)
                / (2 * (12 + 11*np.pi**2)))
        x_e4 = -((72*np.pi**2*t_eq**2 - 132*np.pi**2*t_eq - 28 + 39*np.pi**2)
                 / (2 * (12 + 11*np.pi**2)))
        x_e5 = -((np.cos(6*np.pi*t_eq) + 18*np.pi**2*t_eq**2 - 36*np.pi**2*t_eq
                 - 13 + 7*np.pi**2) / (12 + 11*np.pi**2))
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/6
            t_i2 = (t_eq > 1/6) & (t_eq <= 1/3)
            t_i3 = (t_eq > 1/3) & (t_eq <= 2/3)
            t_i4 = (t_eq > 2/3) & (t_eq <= 5/6)
            t_i5 = t_eq > 5/6
            x_eq = np.empty_like(t_eq)
            x_eq[t_i1] = x_e1[t_i1]
            x_eq[t_i2] = x_e2[t_i2]
            x_eq[t_i3] = x_e3[t_i3]
            x_eq[t_i4] = x_e4[t_i4]
            x_eq[t_i5] = x_e5[t_i5]
        else:
            if t_eq <= 1/6:
                x_eq = x_e1
            elif t_eq <= 1/3:
                x_eq = x_e2
            elif t_eq <= 2/3:
                x_eq = x_e3
            elif t_eq <= 5/6:
                x_eq = x_e4
            else:
                x_eq = x_e5
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_e1 = -((6*np.pi*np.sin(6*np.pi*t_eq) - 36*np.pi**2*t_eq)
                 / (12 + 11*np.pi**2))
        v_e2 = (72*np.pi**2*t_eq - 6*np.pi**2) / (12 + 11*np.pi**2)
        v_e3 = -((24*np.pi*np.sin(3*np.pi* t_eq) - 18*np.pi**2)
                 / (12 + 11*np.pi**2))
        v_e4 = -(72*np.pi**2*t_eq - 66*np.pi**2) / (12 + 11*np.pi**2)
        v_e5 = ((6*np.pi*np.sin(6*np.pi*t_eq) - 36*np.pi**2*t_eq + 36*np.pi**2)
                / (12 + 11*np.pi**2))
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/6
            t_i2 = (t_eq > 1/6) & (t_eq <= 1/3)
            t_i3 = (t_eq > 1/3) & (t_eq <= 2/3)
            t_i4 = (t_eq > 2/3) & (t_eq <= 5/6)
            t_i5 = t_eq > 5/6
            v_eq = np.empty_like(t_eq)
            v_eq[t_i1] = v_e1[t_i1]
            v_eq[t_i2] = v_e2[t_i2]
            v_eq[t_i3] = v_e3[t_i3]
            v_eq[t_i4] = v_e4[t_i4]
            v_eq[t_i5] = v_e5[t_i5]
        else:
            if t_eq <= 1/6:
                v_eq = v_e1
            elif t_eq <= 1/3:
                v_eq = v_e2
            elif t_eq <= 2/3:
                v_eq = v_e3
            elif t_eq <= 5/6:
                v_eq = v_e4
            else:
                v_eq = v_e5
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_e1 = -((36*np.pi**2*np.cos(6*np.pi*t_eq) - 36*np.pi**2)
                 / (12 + 11*np.pi**2))
        a_e2 = (72*np.pi**2) / (12 + 11*np.pi**2)
        a_e3 = -(72*np.pi**2*np.cos(3*np.pi*t_eq)) / (12 + 11*np.pi**2)
        a_e4 = -(72*np.pi**2) / (12 + 11*np.pi**2)
        a_e5 = ((36*np.pi**2*np.cos(6*np.pi*t_eq) - 36*np.pi**2)
                / (12 + 11*np.pi**2))
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/6
            t_i2 = (t_eq > 1/6) & (t_eq <= 1/3)
            t_i3 = (t_eq > 1/3) & (t_eq <= 2/3)
            t_i4 = (t_eq > 2/3) & (t_eq <= 5/6)
            t_i5 = t_eq > 5/6
            a_eq = np.empty_like(t_eq)
            a_eq[t_i1] = a_e1[t_i1]
            a_eq[t_i2] = a_e2
            a_eq[t_i3] = a_e3[t_i3]
            a_eq[t_i4] = a_e4
            a_eq[t_i5] = a_e5[t_i5]
        else:
            if t_eq <= 1/6:
                a_eq = a_e1
            elif t_eq <= 1/3:
                a_eq = a_e2
            elif t_eq <= 2/3:
                a_eq = a_e3
            elif t_eq <= 5/6:
                a_eq = a_e4
            else:
                a_eq = a_e5
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_e1 = (216*np.pi**3*np.sin(6*np.pi*t_eq)) / (12 + 11*np.pi**2)
        j_e2 = 0
        j_e3 = (216*np.pi**3*np.sin(3*np.pi*t_eq)) / (12 + 11*np.pi**2)
        j_e4 = 0
        j_e5 = -(216*np.pi**3*np.sin(6*np.pi*t_eq)) / (12 + 11*np.pi**2)
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/6
            t_i2 = (t_eq > 1/6) & (t_eq <= 1/3)
            t_i3 = (t_eq > 1/3) & (t_eq <= 2/3)
            t_i4 = (t_eq > 2/3) & (t_eq <= 5/6)
            t_i5 = t_eq > 5/6
            j_eq = np.empty_like(t_eq)
            j_eq[t_i1] = j_e1[t_i1]
            j_eq[t_i2] = j_e2
            j_eq[t_i3] = j_e3[t_i3]
            j_eq[t_i4] = j_e4
            j_eq[t_i5] = j_e5[t_i5]
        else:
            if t_eq <= 1/6:
                j_eq = j_e1
            elif t_eq <= 1/3:
                j_eq = j_e2
            elif t_eq <= 2/3:
                j_eq = j_e3
            elif t_eq <= 5/6:
                j_eq = j_e4
            else:
                j_eq = j_e5
        return j_eq


# classes for ramps
class ConstAccRamp(_ramp):
    """R0 constant acceleration ramp

inputs
    t: master axis
    xgo: slave at the beginning of master axis
    vgo: slave velocity at the beginning
    vng: slave velocity at the end

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_eq = t_eq**2 / 2
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_eq = t_eq
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_eq = 1
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_eq = 0
        return j_eq


class Poly3Ramp(_ramp):
    """R1 polynomial, degree 3 ramp

inputs
    t: master axis
    xgo: slave at the beginning of master axis
    vgo: slave velocity at the beginning
    vng: slave velocity at the end

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_eq = -t_eq**4/2 + t_eq**3
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_eq = -2*t_eq**3 + 3*t_eq**2
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_eq = -6*t_eq**2 + 6*t_eq
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_eq = -12*t_eq + 6
        return j_eq


class SineRamp(_ramp):
    """R1 cycloid ramp

inputs
    t: master axis
    xgo: slave at the beginning of master axis
    vgo: slave velocity at the beginning
    vng: slave velocity at the end

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_eq = -np.sin(np.pi*t_eq)/(2*np.pi) + t_eq/2
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_eq = -np.cos(np.pi*t_eq)/2 + 1/2
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_eq = np.pi * np.sin(np.pi*t_eq) / 2
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_eq = np.pi**2 * np.cos(np.pi*t_eq) / 2
        return j_eq


class Poly5Ramp(_ramp):
    """R2 polynomial, degree 5 ramp

inputs
    t: master axis
    xgo: slave at the beginning of master axis
    vgo: slave velocity at the beginning
    vng: slave velocity at the end

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_eq = t_eq**6 - 3*t_eq**5 + (5/2)*t_eq**4
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_eq = 6*t_eq**5 - 15*t_eq**4 + 10*t_eq**3
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_eq = 30*t_eq**4 - 60*t_eq**3 + 30*t_eq**2
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_eq = 120*t_eq**3 - 180*t_eq**2 + 60*t_eq
        return j_eq


class SineJRamp(_ramp):
    """R2 cycloid jerk ramp

inputs
    t: master axis
    xgo: slave at the beginning of master axis
    vgo: slave velocity at the beginning
    vng: slave velocity at the end

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_eq = (np.cos(2*np.pi*t_eq) - 1) / (4*np.pi**2) + t_eq**2/2
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_eq = -np.sin(2*np.pi*t_eq) / (2*np.pi) + t_eq
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_eq = -np.cos(2*np.pi*t_eq) + 1
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_eq = 2*np.pi*np.sin(2*np.pi*t_eq)
        return j_eq


class ModSineJRamp(_ramp):
    """R2 modified cycloid jerk ramp

inputs
    t: master axis
    xgo: slave at the beginning of master axis
    vgo: slave velocity at the beginning
    vng: slave velocity at the end

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_e1 = (np.cos(3*np.pi*t_eq) - 1)/(12*np.pi**2) + 3*t_eq**2/8
        x_e2 = (18*t_eq**2 - 6*t_eq + 1)/24 - 1/(6*np.pi**2)
        x_e3 = (-(np.cos(3*np.pi*t_eq) + 1)/(12*np.pi**2)
                 + (3*t_eq**2 + 2*t_eq - 1)/8)
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/3
            t_i2 = (t_eq > 1/3) & (t_eq <= 2/3)
            t_i3 = t_eq > 2/3
            x_eq = np.empty_like(t_eq)
            x_eq[t_i1] = x_e1[t_i1]
            x_eq[t_i2] = x_e2[t_i2]
            x_eq[t_i3] = x_e3[t_i3]
        else:
            if t_eq <= 1/3:
                x_eq = x_e1
            elif t_eq <= 2/3:
                x_eq = x_e2
            else:
                x_eq = x_e3
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_e1 = -np.sin(3*np.pi*t_eq)/(4*np.pi) + 3*t_eq/4
        v_e2 = (6*t_eq - 1) / 4
        v_e3 = np.sin(3*np.pi*t_eq)/(4*np.pi) + (3*t_eq + 1)/4
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/3
            t_i2 = (t_eq > 1/3) & (t_eq <= 2/3)
            t_i3 = t_eq > 2/3
            v_eq = np.empty_like(t_eq)
            v_eq[t_i1] = v_e1[t_i1]
            v_eq[t_i2] = v_e2[t_i2]
            v_eq[t_i3] = v_e3[t_i3]
        else:
            if t_eq <= 1/3:
                v_eq = v_e1
            elif t_eq <= 2/3:
                v_eq = v_e2
            else:
                v_eq = v_e3
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_e1 = -(3*np.cos(3*np.pi*t_eq)-3) / 4
        a_e2 = 3 / 2
        a_e3 = (3*np.cos(3*np.pi*t_eq)+3) / 4
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/3
            t_i2 = (t_eq > 1/3) & (t_eq <= 2/3)
            t_i3 = t_eq > 2/3
            a_eq = np.empty_like(t_eq)
            a_eq[t_i1] = a_e1[t_i1]
            a_eq[t_i2] = a_e2
            a_eq[t_i3] = a_e3[t_i3]
        else:
            if t_eq <= 1/3:
                a_eq = a_e1
            elif t_eq <= 2/3:
                a_eq = a_e2
            else:
                a_eq = a_e3
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_e1 = 9 * np.pi * np.sin(3*np.pi*t_eq) / 4
        j_e2 = 0
        j_e3 = -9 * np.pi * np.sin(3*np.pi*t_eq) / 4
        if isinstance(t_eq, np.ndarray):
            t_i1 = t_eq <= 1/3
            t_i2 = (t_eq > 1/3) & (t_eq <= 2/3)
            t_i3 = t_eq > 2/3
            j_eq = np.empty_like(t_eq)
            j_eq[t_i1] = j_e1[t_i1]
            j_eq[t_i2] = j_e2
            j_eq[t_i3] = j_e3[t_i3]
        else:
            if t_eq <= 1/3:
                j_eq = j_e1
            elif t_eq <= 2/3:
                j_eq = j_e2
            else:
                j_eq = j_e3
        return j_eq


#classes for impulses
class ConstImpulse(_impulse):
    """I0 constant jerk impulse

inputs
    t: master axis
    xgo: slave at the beginning
    vgo: slave velocity at the beginning
    ago: slave acceleration at the beginning
    ang: slave acceleration at the end

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_eq = t_eq**3 / 6
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_eq = t_eq**2 / 2
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_eq = t_eq
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_eq = 1
        return j_eq


class PolyImpulse(_impulse):
    """I1 polynomial impulse

inputs
    t: master axis
    xgo: slave at the beginning
    vgo: slave velocity at the beginning
    ago: slave acceleration at the beginning
    ang: slave acceleration at the end

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_eq = -t_eq**5/10 + t_eq**4/4
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_eq = -t_eq**4/2 + t_eq**3
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_eq = -2*t_eq**3 + 3*t_eq**2
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_eq = -6*t_eq**2 + 6*t_eq
        return j_eq


class SineImpulse(_impulse):
    """I1 cycloid impulse

inputs
    t: master axis
    xgo: slave at the beginning
    vgo: slave velocity at the beginning
    ago: slave acceleration at the beginning
    ang: slave acceleration at the end

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _x_eq(self, t_eq):
        """private function: position"""
        x_eq = (np.cos(np.pi*t_eq)-1)/(2*np.pi**2) + t_eq**2/4
        return x_eq

    def _v_eq(self, t_eq):
        """private function: velocity"""
        v_eq = -np.sin(np.pi*t_eq)/(2*np.pi) + t_eq/2
        return v_eq

    def _a_eq(self, t_eq):
        """private function: acceleration"""
        a_eq = -(np.cos(np.pi*t_eq)-1) / 2
        return a_eq

    def _j_eq(self, t_eq):
        """private function: jerk"""
        j_eq = np.pi * np.sin(np.pi*t_eq) / 2
        return j_eq


# class for polynomial spline
class PolySpline:
    """polynomial spline law of motion

inputs
    t: master axis
    cont: minimum continuous derivatives at each node

methods to define the polynomial spline
    set_startpoint(x, [v, a, j]): slave at the beginning of master axis
    set_endpoint(x, [v, a, j]): slave at the end of master axis
    set_point(t0, [x, v, a, j, cont]): slave, continuity at given point
    set_leg(leg_no, deg): maximum degree of given leg

method to solve the linear problem
    solve(): generates the slave with derivatives

method to get class status description
    selfcheck(): get class status description

methods to get master and slave
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _eq0(self, t, deg):
        """private function: write equations"""
        eq = np.array([t**7, t**6, t**5, t**4, t**3, t**2, t, 1])
        return eq[-deg-1:]

    def _eq1(self, t, deg):
        """private function: write first derivative of equation"""
        eq = np.array([7*t**6, 6*t**5, 5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0])
        return eq[-deg-1:]

    def _eq2(self, t, deg):
        """private function: write second derivative of equation"""
        eq = np.array([42*t**5, 30*t**4, 20*t**3, 12*t**2, 6*t, 2, 0, 0])
        return eq[-deg-1:]

    def _eq3(self, t, deg):
        """private function: write third derivative of equation"""
        eq = np.array([210*t**4, 120*t**3, 60*t**2, 24*t, 6, 0, 0, 0])
        return eq[-deg-1:]

    def _eq4(self, t, deg):
        """private function: write fourth derivative of equation"""
        eq = np.array([840*t**3, 360*t**2, 120*t, 24, 0, 0, 0, 0])
        return eq[-deg-1:]

    def _eq5(self, t, deg):
        """private function: write fifth derivative of equation"""
        eq = np.array([2520*t**2, 720*t, 120, 0, 0, 0, 0, 0])
        return eq[-deg-1:]

    def _eq6(self, t, deg):
        """private function: write sixth derivative of equation"""
        eq = np.array([5040*t, 720, 0, 0, 0, 0, 0, 0])
        return eq[-deg-1:]

    # class initialization
    def __init__(self, t=np.linspace(0., 1., 257), cont=3):
        """private function: class initialization"""
        self._tgo = t[0]
        self._tng = t[-1]
        self._master = _normtime(t, self._tgo, self._tng)
        self._pos = np.empty_like(self._master)
        self._vel = np.empty_like(self._master)
        self._acc = np.empty_like(self._master)
        self._jrk = np.empty_like(self._master)
        self._continuity = int(max(0, cont))
        self._degree = (2*self._continuity) + 1
        self._nodes = {self._master[0]: [(np.NaN, np.NaN, np.NaN, np.NaN),
                       np.NaN], self._master[-1]: [(np.NaN, np.NaN, np.NaN,
                       np.NaN), np.NaN]}
        self._legs = {(self._master[0], self._master[-1]): self._degree}
        self._eq_no = 0
        self._cond_no = 0
        self._cf = []
        self._tn = []
        self._sl = None
        self._ks = []

    # methods to define the spline
    def set_startpoint(self, x=np.NaN, v=np.NaN, a=np.NaN, j=np.NaN):
        """sets slave with derivatives at the beginning of master axis"""
        if self._sl is not None:
            return
        if np.isnan(x) and np.isnan(v) and np.isnan(a) and np.isnan(j):
            return
        v *= self._tng - self._tgo
        a *= (self._tng - self._tgo)**2
        j *= (self._tng - self._tgo)**3
        self._nodes[self._master[0]][0] = (x, v, a, j)

    def set_endpoint(self, x=np.NaN, v=np.NaN, a=np.NaN, j=np.NaN, cont=None):
        """sets slave with derivatives at the end of master axis"""
        if self._sl is not None:
            return
        if np.isnan(x) and np.isnan(v) and np.isnan(a) and np.isnan(j):
            if cont is None:
                cont = self._continuity
            else:
                cont = max(self._continuity, cont)
        elif cont is None:
            cont = np.NaN
        if np.isnan(cont):
            v *= self._tng - self._tgo
            a *= (self._tng - self._tgo)**2
            j *= (self._tng - self._tgo)**3
        else:
            v = np.NaN
            a = np.NaN
            j = np.NaN
        self._nodes[self._master[-1]] = [(x, v, a, j), cont]

    def set_point(self, t0, x=np.NaN, v=np.NaN, a=np.NaN, j=np.NaN, cont=None):
        """sets slave with derivatives at given point in master axis"""
        if self._sl is not None:
            return
        if cont is None:
            cont = self._continuity
        v *= self._tng - self._tgo
        a *= (self._tng - self._tgo)**2
        j *= (self._tng - self._tng)**3
        min_cont = 1
        max_cont = 3
        if np.isnan(x):
            max_cont += 1
        if np.isnan(v):
            max_cont += 1
        if np.isnan(a):
            max_cont += 1
        else:
            min_cont = 2
        if np.isnan(j):
            max_cont += 1
        else:
            min_continuity = 3
        t0 = _normtime(t0, self._tgo, self._tng)
        cont = int(max(max(min_cont, min(cont, max_cont)), self._continuity))
        self._nodes[t0] = [(x, v, a, j), cont]
        times = sorted(self._nodes)
        where = times.index(t0)
        if list(self._legs).count((times[where-1], times[where+1])) == 1:
            self._legs.pop((times[where-1], times[where+1]))
        self._legs[(times[where-1], times[where])] = self._degree
        self._legs[(times[where], times[where+1])] = self._degree

    def set_leg(self, leg_no, deg=None):
        """sets maximum degree of given leg"""
        if self._sl is not None:
            return
        if deg is None:
            return
        if leg_no >= len(list(self._legs)):
            return
        if leg_no < 0:
            leg_no += len(list(self._legs))
        t0 = sorted(self._nodes)[leg_no]
        t1 = sorted(self._nodes)[leg_no+1]
        deg = int(min(max(1, min(deg, 7)), self._degree))
        self._legs[(t0, t1)] = deg

    # method to solve the problem
    def solve(self):
        """sets and solves the linear problem"""
        if self._sl is not None:
            return
        order = min(max(3, self._degree), 7)
        self._eq_no = 0
        for l in list(self._legs):
            self._legs[l] = min(self._legs[l], order)
            self._eq_no += self._legs[l] + 1
        self._cond_no = 0
        times = sorted(self._nodes)
        for t in times:
            for i in range(4):
                if not np.isnan(self._nodes[t][0][i]):
                    self._cond_no += 1
            if not np.isnan(self._nodes[t][1]):
                self._cond_no += self._nodes[t][1] + 1
        if not np.isnan(self._nodes[times[-1]][1]):
            if not np.isnan(self._nodes[times[-1]][0][0]):
                self._cond_no -= 1
        if self._eq_no != self._cond_no:
            return
        # build the matrix ad the vector
        columnsbefore = 0
        for l in sorted(self._legs):
            columnsafter = self._eq_no - columnsbefore - self._legs[l] - 1
            if l[0] == sorted(self._nodes)[0]:
                if not np.isnan(self._nodes[l[0]][0][0]):
                    self._cf.append(
                        np.block(
                            [
                                np.zeros(columnsbefore),
                                self._eq0(l[0], self._legs[l]),
                                np.zeros(columnsafter)
                            ]
                        )
                    )
                    self._tn.append(self._nodes[l[0]][0][0])
                if not np.isnan(self._nodes[l[0]][0][1]):
                    self._cf.append(
                        np.block(
                            [
                                np.zeros(columnsbefore),
                                self._eq1(l[0], self._legs[l]),
                                np.zeros(columnsafter)
                            ]
                        )
                    )
                    self._tn.append(self._nodes[l[0]][0][1])
                if not np.isnan(self._nodes[l[0]][0][2]):
                    self._cf.append(
                        np.block(
                            [
                                np.zeros(columnsbefore),
                                self._eq2(l[0], self._legs[l]),
                                np.zeros(columnsafter)
                            ]
                        )
                    )
                    self._tn.append(self._nodes[l[0]][0][2])
                if not np.isnan(self._nodes[l[0]][0][3]):
                    self._cf.append(
                        np.block(
                            [
                                np.zeros(columnsbefore),
                                self._eq3(l[0], self._legs[l]),
                                np.zeros(columnsafter)
                            ]
                        )
                    )
                    self._tn.append(self._nodes[l[0]][0][3])
            if not np.isnan(self._nodes[l[1]][0][0]):
                if (
                    l[1] != sorted(self._nodes)[-1]
                    or np.isnan(self._nodes[l[1]][1])
                ):
                    self._cf.append(
                        np.block(
                            [
                                np.zeros(columnsbefore),
                                self._eq0(l[1], self._legs[l]),
                                np.zeros(columnsafter)
                            ]
                        )
                    )
                else:
                    deg1 = self._legs[sorted(self._legs)[0]]
                    deg2 = self._legs[sorted(self._legs)[-1]]
                    self._cf.append(
                        np.block(
                            [
                                -self._eq0(sorted(self._nodes)[0], deg1),
                                np.zeros(self._eq_no - deg1 - deg2 - 2),
                                self._eq0(sorted(self._nodes)[-1], deg2)
                            ]
                        )
                    )
                self._tn.append(self._nodes[l[1]][0][0])
            if not np.isnan(self._nodes[l[1]][0][1]):
                self._cf.append(
                    np.block(
                        [
                            np.zeros(columnsbefore),
                            self._eq1(l[1], self._legs[l]),
                            np.zeros(columnsafter)
                        ]
                    )
                )
                self._tn.append(self._nodes[l[1]][0][1])
            if not np.isnan(self._nodes[l[1]][0][2]):
                self._cf.append(
                    np.block(
                        [
                            np.zeros(columnsbefore),
                            self._eq2(l[1], self._legs[l]),
                            np.zeros(columnsafter)
                        ]
                    )
                )
                self._tn.append(self._nodes[l[1]][0][2])
            if not np.isnan(self._nodes[l[1]][0][3]):
                self._cf.append(
                    np.block(
                        [
                            np.zeros(columnsbefore),
                            self._eq3(l[1], self._legs[l]),
                            np.zeros(columnsafter)
                        ]
                    )
                )
                self._tn.append(self._nodes[l[1]][0][3])
            if not np.isnan(self._nodes[l[1]][1]):
                if l[1] == sorted(self._nodes)[-1]:
                    deg1 = self._legs[sorted(self._legs)[0]]
                    deg2 = self._legs[sorted(self._legs)[-1]]
                    if np.isnan(self._nodes[l[1]][0][0]):
                        self._cf.append(
                            np.block(
                                [
                                    -self._eq0(sorted(self._nodes)[0], deg1),
                                    np.zeros(self._eq_no - deg1 - deg2 - 2),
                                    self._eq0(sorted(self._nodes)[-1], deg2)
                                ]
                            )
                        )
                        self._tn.append(0)
                    if self._nodes[l[1]][1] > 0:
                        self._cf.append(
                            np.block(
                                [
                                    -self._eq1(sorted(self._nodes)[0], deg1),
                                    np.zeros(self._eq_no - deg1 - deg2 - 2),
                                    self._eq1(sorted(self._nodes)[-1], deg2)
                                ]
                            )
                        )
                        self._tn.append(0)
                    if self._nodes[l[1]][1] > 1:
                        self._cf.append(
                            np.block(
                                [
                                    -self._eq2(sorted(self._nodes)[0], deg1),
                                    np.zeros(self._eq_no - deg1 - deg2 - 2),
                                    self._eq2(sorted(self._nodes)[-1], deg2)
                                ]
                            )
                        )
                        self._tn.append(0)
                    if self._nodes[l[1]][1] > 2:
                        self._cf.append(
                            np.block(
                                [
                                    -self._eq3(sorted(self._nodes)[0], deg1),
                                    np.zeros(self._eq_no - deg1 - deg2 - 2),
                                    self._eq3(sorted(self._nodes)[-1], deg2)
                                ]
                            )
                        )
                        self._tn.append(0)
                    if self._nodes[l[1]][1] > 3:
                        self._cf.append(
                            np.block(
                                [
                                    -self._eq4(sorted(self._nodes)[0], deg1),
                                    np.zeros(self._eq_no - deg1 - deg2 - 2),
                                    self._eq4(sorted(self._nodes)[-1], deg2)
                                ]
                            )
                        )
                        self._tn.append(0)
                    if self._nodes[l[1]][1] > 4:
                        self._cf.append(
                            np.block(
                                [
                                    -self._eq5(sorted(self._nodes)[0], deg1),
                                    np.zeros(self._eq_no - deg1 - deg2 - 2),
                                    self._eq5(sorted(self._nodes)[-1], deg2)
                                ]
                            )
                        )
                        self._tn.append(0)
                    if self._nodes[l[1]][1] > 5:
                        self._cf.append(
                            np.block(
                                [
                                    -self._eq6(sorted(self._nodes)[0], deg1),
                                    np.zeros(self._eq_no - deg1 - deg2 - 2),
                                    self._eq6(sorted(self._nodes)[-1], deg2)
                                ]
                            )
                        )
                        self._tn.append(0)
                else:
                    m = sorted(self._legs)[sorted(self._legs).index(l)+1]
                    self._cf.append(
                        np.block(
                            [
                                np.zeros(columnsbefore),
                                self._eq0(l[1], self._legs[l]),
                                -self._eq0(m[0], self._legs[m]),
                                np.zeros(columnsafter - self._legs[m] - 1)
                            ]
                        )
                    )
                    self._tn.append(0)
                    self._cf.append(
                        np.block(
                            [
                                np.zeros(columnsbefore),
                                self._eq1(l[1], self._legs[l]),
                                -self._eq1(m[0], self._legs[m]),
                                np.zeros(columnsafter - self._legs[m] - 1)
                            ]
                        )
                    )
                    self._tn.append(0)
                    if self._nodes[l[1]][1] > 1:
                        self._cf.append(
                            np.block(
                                [
                                    np.zeros(columnsbefore),
                                    self._eq2(l[1], self._legs[l]),
                                    -self._eq2(m[0], self._legs[m]),
                                    np.zeros(columnsafter-self._legs[m]-1)
                                ]
                            )
                        )
                        self._tn.append(0)
                    if self._nodes[l[1]][1] > 2:
                        self._cf.append(
                            np.block(
                                [
                                    np.zeros(columnsbefore),
                                    self._eq3(l[1], self._legs[l]),
                                    -self._eq3(m[0], self._legs[m]),
                                    np.zeros(columnsafter-self._legs[m]-1)
                                ]
                            )
                        )
                        self._tn.append(0)
                    if self._nodes[l[1]][1] > 3:
                        self._cf.append(
                            np.block(
                                [
                                    np.zeros(columnsbefore),
                                    self._eq4(l[1], self._legs[l]),
                                    -self._eq4(m[0], self._legs[m]),
                                    np.zeros(columnsafter-self._legs[m]-1)
                                ]
                            )
                        )
                        self._tn.append(0)
                    if self._nodes[l[1]][1] > 4:
                        self._cf.append(
                            np.block(
                                [
                                    np.zeros(columnsbefore),
                                    self._eq5(l[1], self._legs[l]),
                                    -self._eq5(m[0], self._legs[m]),
                                    np.zeros(columnsafter-self._legs[m]-1)
                                ]
                            )
                        )
                        self._tn.append(0)
                    if self._nodes[l[1]][1] > 5:
                        self._cf.append(
                            np.block(
                                [
                                    np.zeros(columnsbefore),
                                    self._eq6(l[1], self._legs[l]),
                                    -self._eq6(m[0], self._legs[m]),
                                    np.zeros(columnsafter-self._legs[m]-1)
                                ]
                            )
                        )
                        self._tn.append(0)
            columnsbefore += self._legs[l] + 1
        self._cf = np.array(self._cf)
        self._tn = np.array(self._tn)
        # solve the linear problem
        self._sl = np.linalg.solve(self._cf, self._tn)
        # generate slave and derivatives
        for l in sorted(self._legs):
            the_slice = (self._master >= l[0]) & (self._master <= l[1])
            self._ks.append(self._sl[:self._legs[l]+1])
            self._sl = self._sl[self._legs[l]+1:]
            the_x = np.zeros_like(self._master)
            the_v = np.zeros_like(self._master)
            the_a = np.zeros_like(self._master)
            the_j = np.zeros_like(self._master)
            for i in range(len(self._ks[-1])):
                ka = self._ks[-1][i]
                es = self._legs[l] - i
                the_x += ka * self._master**max(es, 0)
                the_v += es * ka * self._master**max(es-1, 0)
                the_a += (es-1) * es * ka * self._master**max(es-2, 0)
                the_j += (es-2) * (es-1) * es * ka * self._master**max(es-3, 0)
            self._pos[the_slice] = the_x[the_slice]
            self._vel[the_slice] = the_v[the_slice]
            self._acc[the_slice] = the_a[the_slice]
            self._jrk[the_slice] = the_j[the_slice]

    # method to get class status description
    def selfcheck(self):
      """returns the status description of the class"""
      print('\nnodes')
      print('|     t    | cont |     x    |     v    |      a     |',
            '      j      |')
      times = sorted(self._nodes)
      for t in times:
          print(f'| {_time(t,self._tgo,self._tng):8.3f} |',
                f'{self._nodes[t][1]:3.0f}  |',
                f'{self._nodes[t][0][0]:8.2f} |',
                f'{self._nodes[t][0][1] / (self._tng-self._tgo):8.0f} |',
                f'{self._nodes[t][0][2] / (self._tng-self._tgo)**2:10.0f} |',
                f'{self._nodes[t][0][3] / (self._tng-self._tgo)**3:12.0f} |')
      legs = sorted(self._legs)
      leg_no = 0
      for l in legs:
          print(f'leg #{leg_no}: degree {self._legs[l]}')
          leg_no += 1
      if self._eq_no == 0 and self._cond_no == 0:
          print('\nlinear problem not solved yet')
      elif self._eq_no == self._cond_no:
          result = f'\n {self._eq_no} equations'
          result += f'- {self._cond_no} conditions: well defined'
          print(result)
      else:
          result = f'\n {self._eq_no} equations'
          result += f'- {self._cond_no} conditions: not square'
          print(result)

    # methods to get master and slave
    def master(self):
        """returns the master axis"""
        if self._sl is None:
            return
        return _time(self._master, self._tgo, self._tng)

    def position(self, t=None):
        """returns the slave position"""
        if self._sl is None:
            return
        elif t is None:
            return self._pos
        else:
            t_eq = _normtime(t, self._tgo, self._tng)
            for l in sorted(self._legs):
                if l[0] <= t_eq <= l[1]:
                    x = 0
                    for i in range(len(self._ks[sorted(self._legs).index(l)])):
                        ka = self._ks[sorted(self._legs).index(l)][i]
                        es = self._legs[l] - i
                        x += ka * t_eq**max(es, 0)
                    return x
            return np.NaN

    def velocity(self, t=None):
        """returns the slave velocity"""
        if self._sl is None:
            return
        elif t is None:
            return self._vel / (self._tng-self._tgo)
        else:
            t_eq = _normtime(t, self._tgo, self._tng)
            for l in sorted(self._legs):
                if l[0] <= t_eq <= l[1]:
                    v = 0
                    for i in range(len(self._ks[sorted(self._legs).index(l)])):
                        ka = self._ks[sorted(self._legs).index(l)][i]
                        es = self._legs[l] - i
                        v += es * ka * t_eq**max(es-1, 0)
                    return v / (self._tng-self._tgo)
            return np.NaN

    def acceleration(self, t=None):
        """returns the slave acceleration"""
        if self._sl is None:
            return
        elif t is None:
            return self._acc / (self._tng-self._tgo)**2
        else:
            t_eq = _normtime(t, self._tgo, self._tng)
            for l in sorted(self._legs):
                if l[0] <= t_eq <= l[1]:
                    a = 0
                    for i in range(len(self._ks[sorted(self._legs).index(l)])):
                        ka = self._ks[sorted(self._legs).index(l)][i]
                        es = self._legs[l] - i
                        a += (es-1) * es * ka * t_eq**max(es-2, 0)
                    return a / (self._tng-self._tgo)**2
            return np.NaN

    def jerk(self, t=None):
        """returns the slave jerk"""
        if self._sl is None:
            return
        if t is None:
            return self._jrk / (self._tng-self._tgo)**3
        else:
            t_eq = _normtime(t, self._tgo, self._tng)
            for l in sorted(self._legs):
                if l[0] <= t_eq <= l[1]:
                    j = 0
                    for i in range(len(self._ks[sorted(self._legs).index(l)])):
                        ka = self._ks[sorted(self._legs).index(l)][i]
                        es = self._legs[l] - i
                        j += (es-2) * (es-1) * es * ka * t_eq**max(es-3, 0)
                    return j / (self._tng-self._tgo)**3
            return np.NaN


# classes for law of motion manipulation
class Compose:
    """gives a law of motion another law as master

inputs
    master: law of motion to be used as master
    slave: slave law of motion

methods
    master(): returns the independant axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # class initialization
    def __init__(self, master, slave):
        self._master_law = master
        self._slave_law = slave
        self._master = master.master()
        self._pos = np.empty_like(self._master)
        self._vel = np.empty_like(self._master)
        self._acc = np.empty_like(self._master)
        self._jrk = np.empty_like(self._master)
        for i in range(len(self._master)):
            t = self._master[i]
            mx = self._master_law.position(t)
            mv = self._master_law.velocity(t)
            ma = self._master_law.acceleration(t)
            mj = self._master_law.jerk(t)
            sx = self._slave_law.position(mx)
            sv = self._slave_law.velocity(mx)
            sa = self._slave_law.acceleration(mx)
            sj = self._slave_law.jerk(mx)
            self._pos[i] = sx
            self._vel[i] = sv * mv
            self._acc[i] = sa*mv**2 + sv*ma
            self._jrk[i] = sj*mv**3 + 3*sa*mv*ma + sv*mj

    # public methods
    def master(self):
        """returns the independant axis"""
        return self._master

    def master_law(self):
        """returns the law of motion used as master"""
        return self._master_law

    def slave_law(self):
        """returns the law of motion used as slave"""
        return self._slave_law

    def position(self, t=None):
        """returns the slave position"""
        if t is None:
            return self._pos
        else:
            mx = self._master_law.position(t)
            sx = self._slave_law.position(mx)
            x = sx
            return x

    def velocity(self, t=None):
        """returns the slave velocity"""
        if t is None:
            return self._vel
        else:
            mx = self._master_law.position(t)
            mv = self._master_law.velocity(t)
            sv = self._slave_law.velocity(mx)
            v = sv * mv
            return v

    def acceleration(self, t=None):
        """returns the slave acceleration"""
        if t is None:
            return self._acc
        else:
            mx = self._master_law.position(t)
            mv = self._master_law.velocity(t)
            ma = self._master_law.acceleration(t)
            sv = self._slave_law.velocity(mx)
            sa = self._slave_law.acceleration(mx)
            a = sa*mv**2 + sv*ma
            return a

    def jerk(self, t=None):
        """returns the slave jerk"""
        if t is None:
            return self._jrk
        else:
            mx = self._master_law.position(t)
            mv = self._master_law.velocity(t)
            ma = self._master_law.acceleration(t)
            mj = self._master_law.jerk(t)
            sv = self._slave_law.velocity(mx)
            sa = self._slave_law.acceleration(mx)
            sj = self._slave_law.jerk(mx)
            j = sj*mv**3 + 3*sa*mv*ma + sv*mj
            return j


class Import:
    """builds a law of motion importing value lists

inputs
    master: list of values to be used as a master
    slave: list of values to be used as slave
    [order]: order of the slave [default: 0]
              0: position
              1: velocity
              2: acceleration
              3: jerk

methods
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # private functions
    def _interpolate(self, t):
        "private function: interpolate values with poly7"
        ts = np.where(self._master <= t, True, False)
        mbelow = np.empty_like(self._master)
        mbelow[:] = np.NaN
        mbelow[ts] = self._master[ts]
        mbelow = mbelow[~ np.isnan(mbelow)]
        mgo = mbelow[-1]
        pbelow = np.empty_like(self._master)
        pbelow[:] = np.NaN
        pbelow[ts] = self._pos[ts]
        pbelow = pbelow[~ np.isnan(pbelow)]
        pgo = pbelow[-1]
        vbelow = np.empty_like(self._master)
        vbelow[:] = np.NaN
        vbelow[ts] = self._vel[ts]
        vbelow = vbelow[~ np.isnan(vbelow)]
        vgo = vbelow[-1]
        abelow = np.empty_like(self._master)
        abelow[:] = np.NaN
        abelow[ts] = self._acc[ts]
        abelow = abelow[~ np.isnan(abelow)]
        ago = abelow[-1]
        jbelow = np.empty_like(self._master)
        jbelow[:] = np.NaN
        jbelow[ts] = self._jrk[ts]
        jbelow = jbelow[~ np.isnan(jbelow)]
        jgo = jbelow[-1]
        mabove = np.empty_like(self._master)
        mabove[:] = np.NaN
        mabove[~ ts] = self._master[~ ts]
        mabove = mabove[~ np.isnan(mabove)]
        mng = mabove[0]
        pabove = np.empty_like(self._master)
        pabove[:] = np.NaN
        pabove[~ ts] = self._pos[~ ts]
        pabove = pabove[~ np.isnan(pabove)]
        png = pabove[0]
        vabove = np.empty_like(self._master)
        vabove[:] = np.NaN
        vabove[~ ts] = self._vel[~ ts]
        vabove = vabove[~ np.isnan(vabove)]
        vng = mabove[0]
        aabove = np.empty_like(self._master)
        aabove[:] = np.NaN
        aabove[~ ts] = self._acc[~ ts]
        aabove = aabove[~ np.isnan(aabove)]
        ang = aabove[0]
        jabove = np.empty_like(self._master)
        jabove[:] = np.NaN
        jabove[~ ts] = self._jrk[~ ts]
        jabove = jabove[~ np.isnan(jabove)]
        jng = jabove[0]
        funct = PolySpline(np.array([mgo, mng]), cont=3)
        funct.set_startpoint(pgo, vgo, ago, jgo)
        funct.set_endpoint(png, vng, ago, jng)
        funct.solve()
        return funct

    # class initialization
    def __init__(self, master, slave, order=0):
        """private function: class initialization"""
        self._master= np.array(master)
        if order == 3:
            self._pos = np.zeros_like(master)
            self._vel = np.zeros_like(master)
            self._acc = np.zeros_like(master)
            self._jrk = np.array(slave)
            for i in range(1, len(self._master)):
                self._acc[i] = (self._acc[i-1] + (self._jrk[i]+self._jrk[i-1])
                                *(self._master[i]-self._master[i-1])/2)
                self._vel[i] = (self._vel[i-1] + (self._acc[i]+self._acc[i-1])
                                *(self._master[i]-self._master[i-1])/2)
                self._pos[i] = (self._pos[i-1] + (self._vel[i]+self._vel[i-1])
                                *(self._master[i]-self._master[i-1])/2)
        elif order == 2:
            self._pos = np.zeros_like(master)
            self._vel = np.zeros_like(master)
            self._acc = np.array(slave)
            self._jrk = np.gradient(self._acc, self._master, edge_order=2)
            for i in range(1, len(self._master)):
                self._vel[i] = (self._vel[i-1] + (self._acc[i]+self._acc[i-1])
                                *(self._master[i]-self._master[i-1])/2)
                self._pos[i] = (self._pos[i-1] + (self._vel[i]+self._vel[i-1])
                                *(self._master[i]-self._master[i-1])/2)
        elif order == 1:
            self._pos = np.zeros_like(master)
            self._vel = np.array(slave)
            self._acc = np.gradient(self._vel, self._master, edge_order=2)
            self._jrk = np.gradient(self._acc, self._master, edge_order=2)
            for i in range(1, len(self._master)):
                self._pos[i] = (self._pos[i-1] + (self._vel[i]+self._vel[i-1])
                                *(self._master[i]-self._master[i-1])/2)
        else:
            self._pos = np.array(slave)
            self._vel = np.gradient(self._pos, self._master, edge_order=2)
            self._acc = np.gradient(self._vel, self._master, edge_order=2)
            self._jrk = np.gradient(self._acc, self._master, edge_order=2)

    # public methods
    def master(self):
        """returns the independant axis"""
        return self._master

    def position(self, t=None):
        """returns the slave position"""
        if t is None:
            return self._pos
        elif t < self._master[0] or t > self._master[-1]:
            return np.NaN
        elif t == self._master[0]:
            return self._pos[0]
        elif t == self._master[-1]:
            return self._pos[-1]
        else:
            return self._interpolate(t).position(t)

    def velocity(self, t=None):
        """returns the slave velocity"""
        if t is None:
            return self._vel
        elif t < self._master[0] or t > self._master[-1]:
            return np.NaN
        elif t == self._master[0]:
            return self._vel[0]
        elif t == self._master[-1]:
            return self._vel[-1]
        else:
            return self._interpolate(t).velocity(t)

    def acceleration(self, t=None):
        """returns the slave acceleration"""
        if t is None:
            return self._acc
        elif t < self._master[0] or t > self._master[-1]:
            return np.NaN
        elif t == self._master[0]:
            return self._acc[0]
        elif t == self._master[-1]:
            return self._acc[-1]
        else:
            return self._interpolate(t).acceleration(t)

    def jerk(self, t=None):
        """returns the slave jerk"""
        if t is None:
            return self._jrk
        elif t < self._master[0] or t > self._master[-1]:
            return np.NaN
        elif t == self._master[0]:
            return self._jrk[0]
        elif t == self._master[-1]:
            return self._jrk[-1]
        else:
            return self._interpolate(t).jerk(t)


class Shift:
    """shifts the law of motion along the slave axis

inputs
    law: law of motion to be translated
    [amount]: entity of the translation
        plus-minus around zero if omitted

methods
    set_new_master([tgo, tng]): changes master axis, start and end points
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    #class initialization
    def __init__(self, law, amount=None):
        """private function: class initialziation"""
        self._law = law
        if amount is None:
            self._amount = -(max(law.position())+min(law.position())) / 2
        else:
            self._amount = amount
        self._master = self._law.master()
        self._pos = self._law.position() + self._amount
        self._vel = self._law.velocity()
        self._acc = self._law.acceleration()
        self._jrk = self._law.jerk()
        self._oldgo = self._master[0]
        self._oldng = self._master[-1]
        self._tgo = self._oldgo
        self._tng = self._oldng

    #public methods
    def set_new_master(self, tgo, tng):
        """changes master axis, start and end points"""
        if tgo is None and tng is None:
            pass
        elif tgo is None:
            self._tgo = tng - (self._oldng-self._oldgo)
            self._tng = tng
        elif tng is None:
            self._tgo = tgo
            self._tng = tgo + (self._oldng-self._oldgo)
        else:
            self._tgo = tgo
            self._tng = tng

    def master(self):
        """returns the master axis"""
        m = (self._tgo + (self._master-self._oldgo)*(self._tng-self._tgo)
             /(self._oldng-self._oldgo))
        return m

    def position(self, t=None):
        """returns the slave position"""
        if t is None:
            return self._pos
        else:
            t = (self._oldgo + (t-self._tgo)*(self._oldng-self._oldgo)
                 /(self._tng-self._tgo))
            if self._master[0] <= t <= self._master[-1]:
                return self._law.position(t) +self._amount
            else:
              return np.NaN

    def velocity(self, t=None):
        """returns the slave velocity"""
        if t is None:
            v = self._vel * (self._oldng-self._oldgo) / (self._tng-self._tgo)
            return v
        else:
            t = (self._oldgo + (t-self._tgo)*(self._oldng-self._oldgo)
                 /(self._tng-self._tgo))
            if self._oldgo <= t <= self._oldng:
                v = (self._law.velocity(t) * (self._oldng-self._oldgo)
                     / (self._tng-self._tgo))
                return v
            else:
                return np.NaN

    def acceleration(self, t=None):
        """returns the slave acceleration"""
        if t is None:
            a = (self._acc * ((self._oldng-self._oldgo)
                 /(self._tng-self._tgo))**2)
            return a
        else:
            t = (self._oldgo + (t-self._tgo)*(self._oldng-self._oldgo)
                 /(self._tng-self._tgo))
            if self._oldgo <= t <= self._oldng:
                a = (self._law.acceleration(t) * ((self._oldng-self._oldgo)
                     /(self._tng-self._tgo))**2)
                return a
            else:
                return np.NaN

    def jerk(self, t=None):
        """returns the slave jerk"""
        if t is None:
            j = (self._jrk * ((self._oldng-self._oldgo)
                 /(self._tng-self._tgo))**3)
            return j
        else:
            t = (self._oldgo + (t-self._tgo)*(self._oldng-self._oldgo)
                 /(self._tng-self._tgo))
            if self._oldgo <= t <= self._oldng:
                j = (self._law.jerk(t) * ((self._oldng-self._oldgo)
                     /(self._tng-self._tgo))**3)
                return j
            else:
                return np.NaN


class Slice:
    """slices one law of motion along master axis

inputs
    law: law of motion to be sliced
    tgo: beginning of the slice (rounded to existing master)
    tng: end of the slice (rounded to existing master)

methods
    set_new_master([tgo, tng]): changes master axis, start and end points
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # class initialization
    def __init__(self, law, tgo=None, tng=None):
        """private function: class initialization"""
        self._law = law
        if tgo is None:
            tgo = self._law.master()[0]
        else:
            tgo = max(self._law.master()[0], tgo)
        if tng is None:
            tng = self._law.master()[-1]
        else:
            tng = min(self._law.master()[-1], tng)
        where = np.array(list(tgo <= t <= tng for t in self._law.master()))
        self._master = np.empty_like(self._law.master())
        self._master[:] = np.NaN
        self._master[where] = self._law.master()[where]
        self._master = self._master[~ np.isnan(self._master)]
        self._pos = np.empty_like(self._law.position())
        self._pos[:] = np.NaN
        self._pos[where] = self._law.position()[where]
        self._pos = self._pos[~ np.isnan(self._pos)]
        self._vel = np.empty_like(self._law.velocity())
        self._vel[:] = np.NaN
        self._vel[where] = self._law.velocity()[where]
        self._vel = self._vel[~ np.isnan(self._vel)]
        self._acc = np.empty_like(self._law.acceleration())
        self._acc[:] = np.NaN
        self._acc[where] = self._law.acceleration()[where]
        self._acc = self._acc[~ np.isnan(self._acc)]
        self._jrk = np.empty_like(self._law.jerk())
        self._jrk[:] = np.NaN
        self._jrk[where] = self._law.jerk()[where]
        self._jrk = self._jrk[~ np.isnan(self._jrk)]
        if self._master[0] > self._law.master()[0]:
            i = np.where(self._law.master() == self._master[0])[0][0] - 1
            t_ext = self._law.master()[i]
            if tgo-t_ext < self._master[0]-tgo:
                self._master = np.append(t_ext, self._master)
                self._pos = np.append(self._law.position()[i], self._pos)
                self._vel = np.append(self._law.velocity()[i], self._vel)
                self._acc = np.append(self._law.acceleration()[i], self._acc)
                self._jrk = np.append(self._law.jerk()[i], self._jrk)
        if self._master[-1] < self._law.master()[-1]:
            i = np.where(self._law.master() == self._master[-1])[0][0] + 1
            i = self._law.master().tolist().index(self._master[-1]) + 1
            t_ext = self._law.master()[i]
            if t_ext-tng < tng-self._master[-1]:
                self._master = np.append(self._master, t_ext)
                self._pos = np.append(self._pos, self._law.position()[i])
                self._vel = np.append(self._vel, self._law.velocity()[i])
                self._acc = np.append(self._acc, self._law.acceleration()[i])
                self._jrk = np.append(self._jrk, self._law.jerk()[i])
        self._oldgo = self._master[0]
        self._oldng = self._master[-1]
        self._tgo = self._oldgo
        self._tng = self._oldng

    # public methods
    def set_new_master(self, tgo=None, tng=None):
        """changes master axis, start and end points"""
        if tgo is None and tng is None:
            pass
        elif tgo is None:
            self._tgo = tng - (self._oldng-self._oldgo)
            self._tng = tng
        elif tng is None:
            self._tgo = tgo
            self._tng = tgo + (self._oldng-self._oldgo)
        else:
            self._tgo = tgo
            self._tng = tng

    def master(self):
        """returns the master axis"""
        m = (self._tgo + (self._master-self._oldgo)*(self._tng-self._tgo)
             /(self._oldng-self._oldgo))
        return m

    def position(self, t=None):
        """returns the slave position"""
        if t is None:
            return self._pos
        else:
            t = (self._oldgo + (t-self._tgo)*(self._oldng-self._oldgo)
                 /(self._tng-self._tgo))
            if self._master[0] <= t <= self._master[-1]:
                return self._law.position(t)
            else:
              return np.NaN

    def velocity(self, t=None):
        """returns the slave velocity"""
        if t is None:
            v = self._vel * (self._oldng-self._oldgo) / (self._tng-self._tgo)
            return v
        else:
            t = (self._oldgo + (t-self._tgo)*(self._oldng-self._oldgo)
                 /(self._tng-self._tgo))
            if self._oldgo <= t <= self._oldng:
                v = (self._law.velocity(t) * (self._oldng-self._oldgo)
                     / (self._tng-self._tgo))
                return v
            else:
                return np.NaN

    def acceleration(self, t=None):
        """returns the slave acceleration"""
        if t is None:
            a = (self._acc * ((self._oldng-self._oldgo)
                 /(self._tng-self._tgo))**2)
            return a
        else:
            t = (self._oldgo + (t-self._tgo)*(self._oldng-self._oldgo)
                 /(self._tng-self._tgo))
            if self._oldgo <= t <= self._oldng:
                a = (self._law.acceleration(t) * ((self._oldng-self._oldgo)
                     /(self._tng-self._tgo))**2)
                return a
            else:
                return np.NaN

    def jerk(self, t=None):
        """returns the slave jerk"""
        if t is None:
            j = (self._jrk * ((self._oldng-self._oldgo)
                 /(self._tng-self._tgo))**3)
            return j
        else:
            t = (self._oldgo + (t-self._tgo)*(self._oldng-self._oldgo)
                 /(self._tng-self._tgo))
            if self._oldgo <= t <= self._oldng:
                j = (self._law.jerk(t) * ((self._oldng-self._oldgo)
                     /(self._tng-self._tgo))**3)
                return j
            else:
                return np.NaN


class Stitch:
    """join several laws of motion into one

inputs
    legs: laws of motion to be joined together

methods
    set_new_master([tgo, tng]): changes master axis, start and end points
    master(): returns the master axis
    position([t]): returns the slave position [at given master]
    velocity([t]): returns the slave velocity [at given master]
    acceleration([t]): returns the slave acceleration [at given master]
    jerk([t]): returns the slave jerk [at given master]
"""

    # class initialization
    def __init__(self, *legs):
        """private function: class initialization"""
        self._legs = legs
        self._master = self._legs[0].master()
        self._pos = self._legs[0].position()
        self._vel = self._legs[0].velocity()
        self._acc = self._legs[0].acceleration()
        self._jrk = self._legs[0].jerk()
        for i in range(len(self._legs) - 1):
            self._master = np.append(self._master,
                                     self._legs[i+1].master()[1:])
            self._pos = np.append(self._pos,
                                  self._legs[i+1].position()[1:])
            self._vel = np.append(self._vel,
                                  self._legs[i+1].velocity()[1:])
            self._acc = np.append(self._acc,
                                  self._legs[i+1].acceleration()[1:])
            self._jrk = np.append(self._jrk,
                                  self._legs[i+1].jerk()[1:])
        self._oldgo = self._master[0]
        self._oldng = self._master[-1]
        self._tgo = self._oldgo
        self._tng = self._oldng

    # public methods
    def set_new_master(self, tgo, tng):
        """changes master axis, start and end points"""
        if tgo is None and tng is None:
            pass
        elif tgo is None:
            self._tgo = tng - (self._oldng-self._oldgo)
            self._tng = tng
        elif tng is None:
            self._tgo = tgo
            self._tng = tgo + (self._oldng-self._oldgo)
        else:
            self._tgo = tgo
            self._tng = tng

    def master(self):
        """returns the master axis"""
        if self._legs is None:
            return
        else:
            m = (self._tgo + (self._master-self._oldgo)*(self._tng-self._tgo)
                 /(self._oldng-self._oldgo))
            return m

    def position(self, t=None):
        """returns the slave position"""
        if self._legs is None:
            return
        elif t is None:
            return self._pos
        else:
            t = (self._oldgo + (t-self._tgo)*(self._oldng-self._oldgo)
                 /(self._tng-self._tgo))
            for l in self._legs:
                if l.master()[0] <= t <= l.master()[-1]:
                    return l.position(t)
            return np.NaN

    def velocity(self, t=None):
        """returns the slave velocity"""
        if self._legs is None:
            return
        elif t is None:
            v = self._vel * (self._oldng-self._oldgo) / (self._tng-self._tgo)
            return v
        else:
            t = (self._oldgo + (t-self._tgo)*(self._oldng-self._oldgo)
                 /(self._tng-self._tgo))
            for l in self._legs:
                if l.master()[0] <= t <= l.master()[-1]:
                    v = (l.velocity(t) * (self._oldng-self._oldgo)
                         / (self._tng-self._tgo))
                    return v
            return np.NaN

    def acceleration(self, t=None):
        """returns the slave acceleration"""
        if self._legs is None:
            return
        elif t is None:
            a = (self._acc * ((self._oldng-self._oldgo)
                 /(self._tng-self._tgo))**2)
            return a
        else:
            t = (self._oldgo + (t-self._tgo)*(self._oldng-self._oldgo)
                 /(self._tng-self._tgo))
            for l in self._legs:
                if l.master()[0] <= t <= l.master()[-1]:
                    a = (l.acceleration(t) * ((self._oldng-self._oldgo)
                         /(self._tng-self._tgo))**2)
                    return a
            return np.NaN

    def jerk(self, t=None):
        """returns the slave jerk"""
        if self._legs is None:
            return
        elif t is None:
            j = (self._jrk * ((self._oldng-self._oldgo)
                 /(self._tng-self._tgo))**3)
            return j
        else:
            t = (self._oldgo + (t-self._tgo)*(self._oldng-self._oldgo)
                 /(self._tng-self._tgo))
            for l in self._legs:
                if l.master()[0] <= t <= l.master()[- 1]:
                    j = (l.jerk(t) * ((self._oldng-self._oldgo)
                         /(self._tng-self._tgo))**3)
                    return j
            return np.NaN

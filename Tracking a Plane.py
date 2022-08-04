#Tracking a Flying Airplane
#We will start by simulating tracking an airplane by using ground based radar. Radars work
by emitting a beam of radio waves and scanning for a return bounce. Anything in the beam's
path will re
ects some of the signal back to the radar. By timing how long it takes for the
re
ected signal to get back to the radar the system can compute the slant distance - the
straight line distance from the radar installation to the object.
For this example we want to take the slant range measurement from the radar and compute
the horizontal position (distance of aircraft from the radar measured over the ground)
and altitude of the aircraft, as in the diagram below.

import ekf_internal
ekf_internal.show_radar_chart()
As discussed in the introduction, our measurement model is the nonlinear function x =
slant2 􀀀 altitude2. Therefore we will need a nonlinear
Predict step:
Linear Nonlinear
x = Fx x = f(x)
P = FPFT + Q P = FPFT + Q
Update step:
Linear Nonlinear
K = PHT (HPHT + R)􀀀1 K = PHT (HPHT + R)􀀀1
x = x + K(z 􀀀 Hx) x = x + K(z 􀀀 h(x))
P = P(I 􀀀 KH) P = P(I 􀀀 KH)

In [13]: xs = np.arange(0,2,0.01)
ys = [x**2 - 2*x for x in xs]
plt.plot (xs, ys)
plt.show()


Suppose we want to linearlize this equation so we can evaluate it's value at 1.5. In other
words, we want to create a linear function of the form yl(x) = ax + b such that yl(1:5) gives
the same value as y(1:5). Obviously there is not single linear equation that will do this. But
if we linearize y(x) at 1.5, then we will have a perfect answer for yl(1:5), and a progressively
worse answer as our evaluation point gets further away from 1.5.
The simplest way to linearize a function is to take a partial derivative of it. In geometic
terms, the derivative of a function at a point is just the slope of the function. Let's just look
at that, and then reason about why this is a good choice.
The derivative of f(x) = x2 􀀀 2x is @f
@x = 2x 􀀀 2, so the slope at 1.5 is 2  1:5 􀀀 2 = 1.
Let's plot that.
In [14]: def y(x):
return x - 2.25
plt.plot (xs, ys)
plt.plot ([1,2], [y(1),y(2)], c='r')
232
plt.ylim([-1.5, 1])
plt.show()
This

In [15]: def y(x):
return 8*x - 12.75
plt.plot (xs, ys)
plt.plot ([1.25, 1.75], [y(1.25), y(1.75)], c='r')
plt.ylim([-1.5, 1])
plt.show()


#writing a simulation for the radar.
import random
import math
class Radar(object):
def __init__(self, pos, vel, alt, dt):
self.pos = pos
self.vel = vel
self.alt = alt
self.dt = dt
def get(self):
""" Simulate radar range to object at 1K altidue and moving at 100m/s.
Adds about 5% measurement noise. Returns slant range to the object.
Call once for each new measurement at dt time from last call.
"""
# add some process noise to the system
vel = self.vel + 5*random.gauss(0,1)
alt = self.alt + 10*random.gauss(0,1)
self.pos = self.pos + vel*self.dt
# add measurment noise
234
err = self.pos * 0.05*random.gauss(0,1)
return math.sqrt(self.pos**2 + alt**2) + err
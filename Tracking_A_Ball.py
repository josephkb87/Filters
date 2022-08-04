
#Things to Consider
# Laws of Physics of the object(A ball thrown in a vacuum must obey Newtonian laws). 
# In a constant gravitational eld it(the ball) will travel in a parabola. 
# We assume familiarity with the derivation of the formula:

y = ((g/2)t*2 + vt + y ((At zero, we subscript with 0)
x = vt + x (At zero, we subscript with 0)

vx0 = v cos(Angle)
vy0 = v sin (Angle)
Because we don't have real data we will start by writing a simulator for a ball. As always,
we add a noise term independent of time so we can simulate noise sensors.

from math import radians, sin, cos
import math
def rk4(y, x, dx, f):
"""computes 4th order Runge-Kutta for dy/dx.
y is the initial value for y
x is the initial value for x
dx is the difference in x (e.g. the time step)
f is a callable function (y, x) that you supply to compute dy/dx for
the specified values.
"""
k1 = dx * f(y, x)
k2 = dx * f(y + 0.5*k1, x + 0.5*dx)
k3 = dx * f(y + 0.5*k2, x + 0.5*dx)
k4 = dx * f(y + k3, x + dx)
return y + (k1 + 2*k2 + 2*k3 + k4) / 6.
def fx(x,t):
return fx.vel
def fy(y,t):
return fy.vel - 9.8*t
class BallTrajectory2D(object):
def __init__(self, x0, y0, velocity, theta_deg=0., g=9.8, noise=[0.0,0.0]):
self.x = x0
self.y = y0
self.t = 0
theta = math.radians(theta_deg)
fx.vel = math.cos(theta) * velocity
fy.vel = math.sin(theta) * velocity
self.g = g
self.noise = noise
201
def step (self, dt):
self.x = rk4 (self.x, self.t, dt, fx)
self.y = rk4 (self.y, self.t, dt, fy)
self.t += dt
return (self.x +random.randn()*self.noise[0], self.y+random.randn()*self.noise[1])
So to create a trajectory starting at (0,15) with a velocity of 60m
s and an angle of 65 we
would write:
traj = BallTrajectory2D (x0=0, y0=15, velocity=100, theta_deg=60)
and then call traj.position(t) for each time step. Let's test this
In [15]: def test_ball_vacuum(noise):
y = 15
x = 0
ball = BallTrajectory2D(x0=x, y0=y, theta_deg=60., velocity=100., noise=noise)
t = 0
dt = 0.25
while y >= 0:
x,y = ball.step(dt)
t += dt
if y >= 0:
plt.scatter(x,y)
plt.axis('equal')
plt.show()
test_ball_vacuum([0,0]) # plot ideal ball position
test_ball_vacuum([1,1]) # plot with noise

Step 1: Choose the State Variables

xt = vx(t􀀀1)t
vxt = vxt􀀀1
yt = 􀀀
g
2
t2 + vyt􀀀1t + yt􀀀1
vyt = 􀀀gt + vy(t􀀀1)
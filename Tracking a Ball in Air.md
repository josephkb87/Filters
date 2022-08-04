# Tracking a Ball in Air

# Assumptions;
#  Ball traveling through the Earth's atmosphere therefore path of the ball is infuenced by wind, drag, and the rotation of the ball. 
# our sensor is a camera; code that we will not implement will perform some type of image processing to detect the position of the ball.

# Implementing Air Drag
# implement the math for a ball moving through air.
Magnus eect (spin causes one
side of the ball to have higher velocity relative to the air vs the opposite side, so the coecient
of drag diers on opposite sides), the eect of lift, humidity, air density, and so on


A ball moving through air encounters wind resistance. This imparts a force on the wall,
called drag, which alters the 
ight of the ball. In Giordano this is denoted as
Fdrag = 􀀀B2v2
where B2 is a coecient derived experimentally, and v is the velocity of the object. Fdrag
can be factored into x and y components with
Fdrag;x = 􀀀B2vvxFdrag;y = 􀀀B2vvy
If m is the mass of the ball, we can use F = ma to compute the acceleration as
ax = 􀀀
B2
m
vvxay = 􀀀
B2
m
vvy
Giordano provides the following function for B2
m , which takes air density, the cross section
of a baseball, and its roughness into account. Understand that this is an approximation based
on wind tunnel tests and several simplifying assumptions. It is in SI units: velocity is in
meters/sec and time is in seconds.
B2
m
= 0:0039 +
0:0058
1 + exp [(v 􀀀 35)=5]
210
Starting with this Euler discretation of the ball path in a vacuum:
x = vxt
y = vyt
vx = vx
vy = vy 􀀀 9:8t
We can incorporate this force (acceleration) into our equations by incorporating accelt
into the velocity update equations. We should subtract this component because drag will
reduce the velocity. The code to do this is quite straightforward, we just need to break out
the Force into x and y components.
I will not belabor this issue further because the computational physics is beyond the scope
of this book. Recognize that a higher delity simulation would require incorporating things
like altitude, temperature, ball spin, and several other factors. My intent here is to impart
some real-world behavior into our simulation to test how our simpler prediction model used
by the Kalman lter reacts to this behavior. Your process model will never exactly model
what happens in the world, and a large factor in designing a good Kalman lter is carefully
testing how it performs against real world data.
The code below computes the behavior of a baseball in air, at sea level, in the presence
of wind. I plot the same initial hit with no wind, and then with a tail wind at 10
mph. Baseball statistics are universally done in US units, and we will follow suit here
(http://en.wikipedia.org/wiki/United States customary units). Note that the velocity of
110 mph is a typical exit speed for a baseball for a home run hit.

# from math import sqrt, exp, cos, sin, radians
def mph_to_mps(x):
return x * .447
def drag_force(velocity):
""" Returns the force on a baseball due to air drag at
the specified velocity. Units are SI"""
return (0.0039 + 0.0058 / (1. + exp((velocity-35.)/5.))) * velocity
v = mph_to_mps(110.)
y = 1
x = 0
dt = .1
theta = radians(35)
def solve(x, y, vel, v_wind, launch_angle):
xs = []
ys = []
v_x = vel*cos(launch_angle)
211
v_y = vel*sin(launch_angle)
while y >= 0:
# Euler equations for x and y
x += v_x*dt
y += v_y*dt
# force due to air drag
velocity = sqrt ((v_x-v_wind)**2 + v_y**2)
F = drag_force(velocity)
# euler's equations for vx and vy
v_x = v_x - F*(v_x-v_wind)*dt
v_y = v_y - 9.8*dt - F*v_y*dt
xs.append(x)
ys.append(y)
return xs, ys
x,y = solve(x=0, y=1, vel=v, v_wind=0, launch_angle=theta)
p1 = plt.scatter(x, y, color='blue')
x,y = solve(x=0, y=1,vel=v, v_wind=mph_to_mps(10), launch_angle=theta)
p2 = plt.scatter(x, y, color='green', marker="v")
plt.legend([p1,p2], ['no wind', '10mph wind'])
plt.show()
# Print Must Output Plot 
from math import radians, sin, cos, sqrt, exp
class BaseballPath(object):
def __init__(self, x0, y0, launch_angle_deg, velocity_ms, noise=(1.0,1.0)):
""" Create 2D baseball path object
(x = distance from start point in ground plane, y=height above ground)
x0,y0 initial position
launch_angle_deg angle ball is travelling respective to ground plane
velocity_ms speeed of ball in meters/second
noise amount of noise to add to each reported position in (x,y)
"""
omega = radians(launch_angle_deg)
self.v_x = velocity_ms * cos(omega)
self.v_y = velocity_ms * sin(omega)
self.x = x0
self.y = y0
self.noise = noise

def drag_force (self, velocity):
""" Returns the force on a baseball due to air drag at
the specified velocity. Units are SI
"""
B_m = 0.0039 + 0.0058 / (1. + exp((velocity-35.)/5.))
return B_m * velocity
def update(self, dt, vel_wind=0.):
213
""" compute the ball position based on the specified time step and
wind velocity. Returns (x,y) position tuple.
"""
# Euler equations for x and y
self.x += self.v_x*dt
self.y += self.v_y*dt
# force due to air drag
v_x_wind = self.v_x - vel_wind
v = sqrt (v_x_wind**2 + self.v_y**2)
F = self.drag_force(v)
# Euler's equations for velocity
self.v_x = self.v_x - F*v_x_wind*dt
self.v_y = self.v_y - 9.81*dt - F*self.v_y*dt
return (self.x + random.randn()*self.noise[0],
self.y + random.randn()*self.noise[1])
Now we can test the Kalman lter against measurements created by this model.
In [22]: y = 1.
x = 0.
theta = 35. # launch angle
v0 = 50.
dt = 1/10. # time step
ball = BaseballPath(x0=x, y0=y, launch_angle_deg=theta, velocity_ms=v0, noise=[.3,.3])
f1 = ball_kf(x,y,theta,v0,dt,r=1.)
f2 = ball_kf(x,y,theta,v0,dt,r=10.)
t = 0
xs = []
ys = []
xs2 = []
ys2 = []
while f1.x[2,0] > 0:
t += dt
x,y = ball.update(dt)
z = np.mat([[x,y]]).T
f1.update(z)
f2.update(z)
xs.append(f1.x[0,0])
214
ys.append(f1.x[2,0])
xs2.append(f2.x[0,0])
ys2.append(f2.x[2,0])
f1.predict()
f2.predict()
p1 = plt.scatter(x, y, color='green', marker='o', s=75, alpha=0.5)
p2, = plt.plot (xs, ys, lw=2)
p3, = plt.plot (xs2, ys2, lw=4, c='#e24a33')
plt.legend([p1,p2, p3],
# Print Must Output Plot 
['Measurements', 'Kalman filter(R=0.5)', 'Kalman filter(R=10)'],
loc='best')
plt.show()
# Print Must Output Plot 

def plot_ball_with_q(q, r=1., noise=0.3):
y = 1.
x = 0.
theta = 35. # launch angle
v0 = 50.
dt = 1/10. # time step
ball = BaseballPath(x0=x,
y0=y,
launch_angle_deg=theta,
velocity_ms=v0,
noise=[noise,noise])
f1 = ball_kf(x,y,theta,v0,dt,r=r, q=q)
t = 0
xs = []
ys = []
while f1.x[2,0] > 0:
t += dt
x,y = ball.update(dt)
z = np.mat([[x,y]]).T
f1.update(z)
xs.append(f1.x[0,0])
ys.append(f1.x[2,0])
f1.predict()
p1 = plt.scatter(x, y, color='green', marker='o', s=75, alpha=0.5)
p2, = plt.plot (xs, ys,lw=2)
plt.legend([p1,p2], ['Measurements', 'Kalman filter'])
plt.show()
216
plot_ball_with_q(0.01)
# Print Must Output Plot 
plot_ball_with_q(0.1)
# Print Must Output Plot 
The

plot_ball_with_q(0.01, r=3, noise=3.)
# Print Must Output Plot 
plot_ball_with_q(0.1, r=3, noise=3.)
# Print Must Output Plot 
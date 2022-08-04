Tracking a Robot


multidimensional problem we will track a robot in a 2D space, such as a
eld. We will start with a simple noisy sensor that outputs noisy (x; y) coordinates which
we will need to lter to generate a 2D track. Once we have mastered this concept, we will
extend the problem signicantly with more sensors and then adding control inputs. blah
blah
9.2 Tracking a Robot
This rst attempt at tracking a robot will closely resemble the 1-D dog tracking problem of
previous chapters. This will allow us to `get our feet wet' with Kalman ltering. So, instead
of a sensor that outputs position in a hallway, we now have a sensor that supplies a noisy
measurement of position in a 2-D space, such as an open eld. That is, at each time T it
will provide an (x; y) coordinate pair specifying the measurement of the sensor's position in
the eld.
Implementation of code to interact with real sensors is beyond the scope of this book,
so as before we will program simple simulations in Python to represent the sensors. We will
develop several of these sensors as we go, each with more complications, so as I program
them I will just append a number to the function name. pos sensor1() is the rst sensor
we write, and so on.
So let's start with a very simple sensor, one that travels in a straight line. It takes as
input the last position, velocity, and how much noise we want, and returns the new position.

Step 1::Choose the State Variables
Step 2: Design State Transition Function
Step 3: Design the Motion Function
Step 4: Design the Measurement Function
Step 5: Design the Measurement Noise Matrix
Step 6: Design the Process Noise Matrix
Step 7: Design Initial Conditions
Implement the Filter Code



Step 1: Choose the State Variables As always, the rst step is to choose our state
variables. We are tracking in two dimensions and have a sensor that gives us a reading in
each of those two dimensions, so we know that we have the two observed variables x and y.
If we created our Kalman lter using only those two variables the performance would not
191
be very good because we would be ignoring the information velocity can provide to us. We
will want to incorporate velocity into our equations as well. I will represent this as
x =
2
664
x
vx
y
vy
3
775
There is nothing special about this organization. I could have listed the (xy) coordinates
rst followed by the velocities, and/or I could done this as a row matrix instead of a column
matrix. For example, I could have chosen:
x =

x y vx vy

All that matters is that the rest of my derivation uses this same scheme. However, it is
typical to use column matrices for state variables, and I prefer it, so that is what we will
use.
It might be a good time to pause and address how you identify the unobserved variables.
This particular example is somewhat obvious because we already worked through the 1D
case in the previous chapters. Would it be so obvious if we were ltering market data,
population data from a biology experiment, and so on? Probably not. There is no easy
answer to this question. The rst thing to ask yourself is what is the interpretation of the
rst and second derivatives of the data from the sensors. We do that because obtaining
the rst and second derivatives is mathematically trivial if you are reading from the sensors
using a xed time step. The rst derivative is just the dierence between two successive
readings. In our tracking case the rst derivative has an obvious physical interpretation: the
dierence between two successive positions is velocity.
Beyond this you can start looking at how you might combine the data from two or more
dierent sensors to produce more information. This opens up the eld of sensor fusion, and
we will be covering examples of this in later sections. For now, recognize that choosing the
appropriate state variables is paramount to getting the best possible performance from your
lter.
Step 2: Design State Transition Function Our next step is to design the state
transition function. Recall that the state transition function is implemented as a matrix F
that we multipy with the previous state of our system to get the next state, like so.
x0 = Fx
I will not belabor this as it is very similar to the 1-D case we did in the previous chapter.
Our state equations for position and velocity would be:
x0 = (1  x) + (t  vx) + (0  y) + (0  vy)
vx = (0  x) + (1  vx) + (0  y) + (0  vy)
y0 = (0  x) + (0  vx) + (1  y) + (t  vy)
vy = (0  x) + (0  vx) + (0  y) + (1  vy)
192
Laying it out that way shows us both the values and row-column organization required
for F. In linear algebra, we would write this as:
2
664
x
vx
y
vy
3
775
0
=
2
664
1 t 0 0
0 1 0 0
0 0 1 t
0 0 0 1
3
775
2
664
x
vx
y
vy
3
775
So, let's do this in Python. It is very simple; the only thing new here is setting dim z to
2. We will see why it is set to 2 in step 4.
In [4]: from filterpy.kalman import KalmanFilter
import numpy as np
f1 = KalmanFilter(dim_x=4, dim_z=2)
dt = 1. # time step
f1.F = np.array ([[1, dt, 0, 0],
[0, 1, 0, 0],
[0, 0, 1, dt],
[0, 0, 0, 1]])
Step 3: Design the Motion Function We have no control inputs to our robot (yet!),
so this step is trivial - set the motion input u to zero. This is done for us by the class when
it is created so we can skip this step, but for completeness we will be explicit.
In [5]: f1.u = 0.
Step 4: Design the Measurement Function The measurement function denes
how we go from the state variables to the measurements using the equation z = Hx. At rst
this is a bit counterintuitive, after all, we use the Kalman lter to go from measurements to
state. But the update step needs to compute the residual between the current measurement
and the measurement represented by the prediction step. Therefore H is multiplied by the
state x to produce a measurement z.
In this case we have measurements for (x,y), so z must be of dimension 21. Our state
variable is size 4  1. We can deduce the required size for H by recalling that multiplying a
matrix of size m  n by n  p yields a matrix of size m  p. Thus,
(2  1) = (a  b)(4  1)
= (a  4)(4  1)
= (2  4)(4  1)
So, H is of size 2  4.
Filling in the values for H is easy in this case because the measurement is the position of
the robot, which is the x and y variables of the state x. Let's make this just slightly more
interesting by deciding we want to change units. So we will assume that the measurements
193
are returned in feet, and that we desire to work in meters. Converting from feet to meters
is a simple as multiplying by 0.3048. However, we are converting from state (meters) to
measurements (feet) so we need to divide by 0.3048. So
H =
 1
0:3048 0 0 0
0 0 1
0:3048 0

which corresponds to these linear equations
z0
x = (
x
0:3048
) + (0  vx) + (0  y) + (0  vy)
z0
y = (0  x) + (0  vx) + (
y
0:3048
) + (0  vy)
To be clear about my intentions here, this is a pretty simple problem, and we could have
easily found the equations directly without going through the dimensional analysis that I did
above. In fact, an earlier draft did just that. But it is useful to remember that the equations
of the Kalman lter imply a specic dimensionality for all of the matrices, and when I start
to get lost as to how to design something it is often extremely useful to look at the matrix
dimensions. Not sure how to design H? Here is the Python that implements this:
In [6]: f1.H = np.array ([[1/0.3048, 0, 0, 0],
[0, 0, 1/0.3048, 0]])
print(f1.H)
[[ 3.2808399 0. 0. 0. ]
[ 0. 0. 3.2808399 0. ]]
Step 5: Design the Measurement Noise Matrix In this step we need to mathematically
model the noise in our sensor. For now we will make the simple assumption that
the x and y variables are independent Gaussian processes. That is, the noise in x is not in
any way dependent on the noise in y, and the noise is normally distributed about the mean.
For now let's set the variance for x and y to be 5 for each. They are independent, so there
is no covariance, and our o diagonals will be 0. This gives us:
R =

5 0
0 5

It is a 22 matrix because we have 2 sensor inputs, and covariance matrices are always
of size nn for n variables. In Python we write:
In [7]: f1.R = np.array([[5,0],
[0, 5]])
print (f1.R)
[[ 5. 0.]
[ 0. 5.]]
194
Step 6: Design the Process Noise Matrix Finally, we design the process noise. We
don't yet have a good way to model process noise, so for now we will assume there is a small
amount of process noise, say 0.1 for each state variable. Later we will tackle this admittedly
dicult topic in more detail. We have 4 state variables, so we need a 44 covariance matrix:
Q =
2
664
0:1 0 0 0
0 0:1 0 0
0 0 0:1 0
0 0 0 0:1
3
775
In Python I will use the numpy eye helper function to create an identity matrix for us,
and multipy it by 0.1 to get the desired result.
In [8]: f1.Q = np.eye(4) * 0.1
print(f1.Q)
[[ 0.1 0. 0. 0. ]
[ 0. 0.1 0. 0. ]
[ 0. 0. 0.1 0. ]
[ 0. 0. 0. 0.1]]
Step 7: Design Initial Conditions For our simple problem we will set the initial
position at (0,0) with a velocity of (0,0). Since that is a pure guess, we will set the covariance
matrix P to a large value.
x =
2
664
0
0
0
0
3
775
P =
2
664
500 0 0 0
0 500 0 0
0 0 500 0
0 0 0 500
3
775
In Python we implement that with
In [9]: f1.x = np.array([[0,0,0,0]]).T
f1.P = np.eye(4) * 500.
print(f1.x)
print()
print (f1.P)
[[ 0.]
[ 0.]
[ 0.]
[ 0.]]
[[ 500. 0. 0. 0.]
[ 0. 500. 0. 0.]
[ 0. 0. 500. 0.]
[ 0. 0. 0. 500.]]
195
9.3 Implement the Filter Code
Design is complete, now we just have to write the Python code to run our lter, and output
the data in the format of our choice. To keep the code clear, let's just print a plot of the
track. We will run the code for 30 iterations.
In [10]: f1 = KalmanFilter(dim_x=4, dim_z=2)
dt = 1.0 # time step
f1.F = np.array ([[1, dt, 0, 0],
[0, 1, 0, 0],
[0, 0, 1, dt],
[0, 0, 0, 1]])
f1.u = 0.
f1.H = np.array ([[1/0.3048, 0, 0, 0],
[0, 0, 1/0.3048, 0]])
f1.R = np.eye(2) * 5
f1.Q = np.eye(4) * .1
f1.x = np.array([[0,0,0,0]]).T
f1.P = np.eye(4) * 500.
# initialize storage and other variables for the run
count = 30
xs, ys = [],[]
pxs, pys = [],[]
s = PosSensor1 ([0,0], (2,1), 1.)
for i in range(count):
pos = s.read()
z = np.array([[pos[0]],[pos[1]]])
f1.predict ()
f1.update (z)
xs.append (f1.x[0,0])
ys.append (f1.x[2,0])
pxs.append (pos[0]*.3048)
pys.append(pos[1]*.3048)
p1, = plt.plot (xs, ys, 'r--')
p2, = plt.plot (pxs, pys)
plt.legend([p1,p2], ['filter', 'measurement'], 2)
196
plt.show()
I encourage you to play with this, setting Q and R to various values. However, we did a
fair amount of that sort of thing in the last chapters, and we have a lot of material to cover,
so I will move on to more complicated cases where we will also have a chance to experience
changing these values.
Now I will run the same Kalman lter with the same settings, but also plot the covariance
ellipse for x and y. First, the code without explanation, so we can see the output. I print the
last covariance to see what it looks like. But before you scroll down to look at the results,
what do you think it will look like? You have enough information to gure this out but this
is still new to you, so don't be discouraged if you get it wrong.
In [27]: import stats
f1 = KalmanFilter(dim_x=4, dim_z=2)
dt = 1.0 # time step
f1.F = np.array ([[1, dt, 0, 0],
[0, 1, 0, 0],
[0, 0, 1, dt],
[0, 0, 0, 1]])
f1.u = 0.
f1.H = np.array ([[1/0.3048, 0, 0, 0],
[0, 0, 1/0.3048, 0]])
f1.R = np.eye(2) * 5
f1.Q = np.eye(4) * .1
197
f1.x = np.array([[0,0,0,0]]).T
f1.P = np.eye(4) * 500.
# initialize storage and other variables for the run
count = 30
xs, ys = [],[]
pxs, pys = [],[]
s = PosSensor1 ([0,0], (2,1), 1.)
for i in range(count):
pos = s.read()
z = np.array([[pos[0]],[pos[1]]])
f1.predict ()
f1.update (z)
xs.append (f1.x[0,0])
ys.append (f1.x[2,0])
pxs.append (pos[0]*.3048)
pys.append(pos[1]*.3048)
# plot covariance of x and y
cov = np.array([[f1.P[0,0], f1.P[2,0]],
[f1.P[0,2], f1.P[2,2]]])
#e = stats.sigma_ellipse (cov=cov, x=f1.x[0,0], y=f1.x[2,0])
#stats.plot_sigma_ellipse(ellipse=e)
stats.plot_covariance_ellipse((f1.x[0,0], f1.x[2,0]), cov=cov,
facecolor='g', alpha=0.2)
p1, = plt.plot (xs, ys, 'r--')
p2, = plt.plot (pxs, pys)
plt.legend([p1,p2], ['filter', 'measurement'], 2)
plt.show()
print("final P is:")
print(f1.P)
198
final P is:
[[ 0.30660483 0.12566239 0. 0. ]
[ 0.12566239 0.24399092 0. 0. ]
[ 0. 0. 0.30660483 0.12566239]
[ 0. 0. 0.12566239 0.24399092]]
Did you correctly predict what the covariance matrix and plots would look like? Perhaps
you were expecting a tilted ellipse, as in the last chapters. If so, recall that in those chapters
we were not plotting x against y, but x against x_ . x is correlated to x_ , but x is not correlated
or dependent on y. Therefore our ellipses are not tilted. Furthermore, the noise for both x
and y are modeled to have the same value, 5, in R. If we were to set R to, for example,
R =

10 0
0 5

we would be telling the Kalman lter that there is more noise in x than y, and our ellipses
would be longer than they are tall.
The nal P tells us everything we need to know about the correlation between the state
variables. If we look at the diagonal alone we see the variance for each variable. In other
words P0;0 is the variance for x, P1;1 is the variance for x_ , P2;2 is the variance for y, and P3;3
is the variance for y_. We can extract the diagonal of a matrix using numpy.diag().
In [12]: print(np.diag(f1.P))
[ 0.30660483 0.24399092 0.30660483 0.24399092]
The covariance matrix contains four 22 matrices that you should be able to easily pick
out. This is due to the correlation of x to x_ , and of y to y_. The upper left hand side shows
the covariance of x to x_ . Let's extract and print, and plot it.
199
In [26]: c = f1.P[0:2,0:2]
print(c)
stats.plot_covariance_ellipse((0,0), cov=c, facecolor='g', alpha=0.2)
[[ 0.08204134 0.02434904]
[ 0.02434904 0.00955614]]
The covariance contains the data for x and x_ in the upper left because of how it is
organized. Recall that entries Pi;j and Pj;i contain p12.
Finally, let's look at the lower left side of P, which is all 0s. Why 0s? Consider P3;0. That
stores the term p30, which is the covariance between y_ and x. These are independent, so
the term will be 0. The rest of the terms are for similarly independent variables.

In [2]: import numpy.random as random
import copy
class PosSensor1(object):
def __init__(self, pos = [0,0], vel = (0,0), noise_scale = 1.):
self.vel = vel
190
self.noise_scale = noise_scale
self.pos = copy.deepcopy(pos)
def read(self):
self.pos[0] += self.vel[0]
self.pos[1] += self.vel[1]
return [self.pos[0] + random.randn() * self.noise_scale,
self.pos[1] + random.randn() * self.noise_scale]
A quick test to verify that it works as we expect.
In [3]: pos = [4,3]
s = PosSensor1 (pos, (2,1), 1)
for i in range (50):
pos = s.read()
plt.scatter(pos[0], pos[1])
plt.show()

from filterpy.kalman import KalmanFilter
import numpy as np
f1 = KalmanFilter(dim_x=4, dim_z=2)
dt = 1. # time step
f1.F = np.array ([[1, dt, 0, 0],
[0, 1, 0, 0],
[0, 0, 1, dt],
[0, 0, 0, 1]])
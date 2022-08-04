#This program will implement a Kalman Filter in Python that Tracks a Dog
#Using a simple 1D Kalmann Filter

"""name = "@clydekingkid"  #The code author#
twitter = ", @clydekingkid"""

#Python libraries to import#
#In [3]: from __future__ import print_function, division#
import numpy as np
import matplotlib.pyplot as plt #Python libraries#
import numpy.random as random  #Python libraries
import math     #Python libraries
class DogSensor(object):
def __init__(self, x0=0, velocity=1, noise=0.0):
    # 75
    """ x0 - initial position
    velocity - (+=right, -=left)
    noise - scaling factor for noise, 0== no noise
    """
    self.x = x0
    self.velocity = velocity
    self.noise = math.sqrt(noise)

    def sense(self):
        self.x = self.x + self.velocity

    return self.x + random.randn() * self.noise
    ##The constructor init() initializes the DogSensor class with an initial position x0,
    ##velocity vel, and an noise scaling factor. The sense() function has the dog move by the set
    ##velocity and returns its new position, with noise added. If you look at the code for sense()
    ##you will see a call to numpy.random.randn(). This returns a number sampled from a normal
    ##distribution with a mean of 0.0. Let’s look at some example output for that.


    In [4]: for i in range(20):
    print(’{: 5.4f}’.format(random.randn()),end=’\t’)
    if (i+1) % 5 == 0:
    print (’’)

    ##Results#
    ##0.7405 0.7229 -0.5326 1.2238 -0.1799
    # 0.0760 -0.5359 1.0663 0.5362 -0.6931
    # 1.9148 0.5687 1.1368 -0.7581 1.7235
    # -1.1808 0.9561 0.4169 0.1081 -1.5491
    # You should see a sequence of numbers near 0, some negative and some positive. Most are
    # probably between -1 and 1, but a few might lie somewhat outside that range. This is what
    # we expect from a normal distribution - values are clustered around the mean, and there are
    # fewer values the further you get from the mean.
    # Okay, so lets look at the output of the DogSensor class. We will start by setting the
    # noise to 0 to check that the class does what we think it does

    ##
    In[5]:
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pylab
    dog = DogSensor(noise=0.0)
    xs = []
    for i in range(10):
        x = dog.sense()
    xs.append(x)
    print("%.4f" % x, end=’ ’),
    plt.plot(xs, label=’dog
    position’)
    plt.legend(loc=’best’)
    plt.show()

    ##Results
    # 76
    # 1.0000 2.0000 3.0000 4.0000 5.0000 6.0000 7.0000 8.0000 9.0000 10.000

    In [6]: def test_sensor(noise_scale):
    dog = DogSensor(noise=noise_scale)
    xs = []
    for i in range(100):
        x = dog.sense()
    xs.append(x)
    plt.plot(xs, label=’sensor’)
    plt.plot([0, 99], [1, 100], ’r - -’, label =’actual’)
    plt.xlabel(’time’)
    plt.ylabel(’pos’)
    plt.ylim([0, 100])
    plt.title(’noise = ’ + str(noise_scale))
    plt.legend(loc=’best’)
    plt.show()
    test_sensor(4.0)

    #####
# In [10]: dog = DogSensor(23, 0, 5)
xs = range(100)
80
ys = []
for i in xs:
ys.append(dog.sense())
plt.plot(xs,ys, label=’dog position’)
plt.legend(loc=’best’)
plt.show()


##Recall the histogram code for adding a measurement to a preexisting belief:
def update(pos, measure, p_hit, p_miss):
q = array(pos, dtype=float)
for i in range(len(hallway)):
if hallway[i] == measure:
q[i] = pos[i] * p_hit
else:
q[i] = pos[i] * p_miss
normalize(q)
return q

##
In [11]: from __future__ import division
import numpy as np
def multiply(mu1, var1, mu2, var2):
mean = (var1*mu2 + var2*mu1) / (var1+var2)
variance = 1 / (1/var1 + 1/var2)
return (mean, variance)
xs = np.arange(16, 30, 0.1)
m1,v1 = 23, 5
m, v = multiply(m1,v1,m1,v1)
82
ys = [stats.gaussian(x,m1,v1) for x in xs]
plt.plot (xs, ys, label=’original’)
ys = [stats.gaussian(x,m,v) for x in xs]
plt.plot (xs, ys, label=’multiply’)
plt.legend(loc=’best’)
plt.show()

##
In [12]: xs = np.arange(16, 30, 0.1)
83
m1, v1 = 23, 5
m2, v2 = 25, 5
m, s = multiply(m1,v1,m2,v2)
ys = [stats.gaussian(x,m1,v1) for x in xs]
plt.plot (xs, ys, label=’measure 1’)
ys = [stats.gaussian(x,m2,v2) for x in xs]
plt.plot (xs, ys, label=’measure 2’)
ys = [stats.gaussian(x,m,v) for x in xs]
plt.plot(xs, ys,label=’multiply’)
plt.legend()
plt.show()

In [13]: xs = np.arange(0, 60, 0.1)
m1, v1 = 10, 5
m2, v2 = 50, 5
m, v = multiply(m1,v1,m2,v2)
ys = [stats.gaussian(x,m1,v1) for x in xs]
plt.plot (xs, ys, label=’measure 1’)
ys = [stats.gaussian(x,m2,v2) for x in xs]
plt.plot (xs, ys, label=’measure 2’)
ys = [stats.gaussian(x,m,v) for x in xs]
plt.plot(xs, ys, label=’multiply’)
plt.legend()
plt.show()

##Implementing the Update Step##
In [14]: def update(mean, variance, measurement, measurement_variance):
return multiply(mean, variance, measurement, measurement_variance)

In [15]: def update_dog(dog_pos, dog_variance, measurement, measurement_variance):
return multiply(dog_pos, dog_sigma, measurement, measurement_variance)


In [16]: dog = DogSensor(velocity=0, noise=1)
pos,s = 2, 5
for i in range(20):
pos,s = update(pos, s, dog.sense(), 5)
print(’time:’, i, ’\tposition =’, "%.3f" % pos, ’\tvariance =’, "%.3f" % s)

###time: 0 position = 1.616 variance = 2.500
###time: 1 position = 0.501 variance = 1.667
###time: 2 position = 0.388 variance = 1.250
###time: 3 position = 0.105 variance = 1.000
###time: 4 position = 0.222 variance = 0.833
###time: 5 position = 0.154 variance = 0.714
###time: 6 position = 0.194 variance = 0.625
###time: 7 position = 0.024 variance = 0.556
###time: 8 position = -0.019 variance = 0.500
###time: 9 position = -0.079 variance = 0.455
###time: 10 position = -0.064 variance = 0.417
###time: 11 position = -0.046 variance = 0.385
87
###time: 12 position = -0.108 variance = 0.357
###time: 13 position = -0.138 variance = 0.333
###time: 14 position = -0.159 variance = 0.312
###time: 15 position = -0.151 variance = 0.294
###time: 16 position = -0.212 variance = 0.278
###time: 17 position = -0.212 variance = 0.263
###time: 18 position = -0.266 variance = 0.250
###time: 19 position = -0.241 variance = 0.238

##Implementing Predictions##
def predict(pos, move, p_correct, p_under, p_over):
n = len(pos)
result = array(pos, dtype=float)
for i in range(n):
result[i] = \
pos[(i-move) % n] * p_correct + \
pos[(i-move-1) % n] * p_over + \
88
pos[(i-move+1) % n] * p_under
return result

###In [17]: def predict(pos, variance, movement, movement_variance):
return (pos + movement, variance + movement_variance)
##hat is left? Just calling these functions. The histogram did nothing more than loop
# ##over the update() and predict() functions, so let’s do the same.

###In [18]: # assume dog is always moving 1m to the right
movement = 1
movement_error = 2
sensor_error = 10
pos = (0, 500) # gaussian N(0,50)
dog = DogSensor(pos[0], velocity=movement, noise=sensor_error)
89
zs = []
ps = []
for i in range(10):
pos = predict(pos[0], pos[1], movement, movement_error)
print(’PREDICT: {: 10.4f} {: 10.4f}’.format(pos[0], pos[1]),end=’\t’)
Z = dog.sense()
zs.append(Z)
pos = update(pos[0], pos[1], Z, sensor_error)
ps.append(pos[0])
print(’UPDATE: {: 10.4f} {: 10.4f}’.format(pos[0], pos[1]))
plt.plot(ps, label=’filter’)
plt.plot(zs, c=’r’, linestyle=’dashed’, label=’measurement’)
plt.legend(loc=’best’)
plt.show()
#####
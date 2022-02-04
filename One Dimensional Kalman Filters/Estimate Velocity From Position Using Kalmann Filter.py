#This program will implement a Kalman Filter in Python that estimates velocity from position measurements
#Using a simple 1D Kalmann Filter

"""name = "@clydekingkid"  #This is the author#
twitter = ", @clydekingkid"""

#Python libraries to import#
#In [3]: from __future__ import print_function, division#
import numpy as np
import matplotlib.pyplot as plt #Python libraries#
import numpy.random as random  #Python libraries
import math     #Python libraries
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

##Compute Measurements##
def getMeasurement(updateNumber):
    if updateNumber == 1:
      getMeasurement.currentPosition = 0
      getMeasurement.currentVelocity = 60 # m/s

    dt = 0.1

    w = 8 * np.random.randn(1)
    v = 8 * np.random.randn(1)

    z = getMeasurement.currentPosition + getMeasurement.currentVelocity*dt + v
    getMeasurement.currentPosition = z - v
    getMeasurement.currentVelocity = 60 + w
    return [z, getMeasurement.currentPosition, getMeasurement.currentVelocity]

##Filter Measurements##
## initialized. The measurement is in the following structures##.
## z is the position measurement, R is the position variance##,
## and t is the timestamp of the measurement.##
 ##initialized. The measurement is in the following structures##
# z is the position measurement,
# R is the position variance,
# and t is the timestamp of the measurement.##
## For this example, it is assumed that measurements have a 10/second and therefore a delta time,
# dt, of 0.1 seconds is used.
# Next the Kalman Gain, K, is computed using the input measurement uncertainty,
# R, and the predicted state covariance matrix, PP.
# This computation was broken down into two steps, first compute the innovation, S,
# and then compute the Kalman Gain, K, with the innovation##
## Lastly, the Kalman Gain, K, is used to compute the new state, x, and state covariance estimate, P.##
def filter(z, updateNumber):
    dt = 0.1
    # Initialize State
    if updateNumber == 1:
        filter.x = np.array([[0],
                            [20]])
        filter.P = np.array([[5, 0],
                                 [0, 5]])

        filter.A = np.array([[1, dt],
                             [0, 1]])
        filter.H = np.array([[1, 0]])
        filter.HT = np.array([[1],
                              [0]])
        filter.R = 10
        filter.Q = np.array([[1, 0],
                             [0, 3]])

    # Predict State Forward
    x_p = filter.A.dot(filter.x)
    # Predict Covariance Forward
    P_p = filter.A.dot(filter.P).dot(filter.A.T) + filter.Q
    # Compute Kalman Gain
    S = filter.H.dot(P_p).dot(filter.HT) + filter.R
    K = P_p.dot(filter.HT)*(1/S)

    # Estimate State
    residual = z - filter.H.dot(x_p)
    filter.x = x_p + K*residual

    # Estimate Covariance
    filter.P = P_p - K.dot(filter.H).dot(P_p)

    return [filter.x[0], filter.x[1], filter.P];
##Test Kalman Filter##
#Now that you have both the input measurements to process and your Kalman Filter,
#its time to write a test program so that you see how your filter performs.
def testFilter():
    dt = 0.1
    t = np.linspace(0, 10, num=300)
    numOfMeasurements = len(t)

    measTime = []
    measPos = []
    measDifPos = []
    estDifPos = []
    estPos = []
    estVel = []
    posBound3Sigma = []

    for k in range(1,numOfMeasurements):
        z = getMeasurement(k)
        # Call Filter and return new State
        f = filter(z[0], k)
        # Save off that state so that it could be plotted
        measTime.append(k)
        measPos.append(z[0])
        measDifPos.append(z[0]-z[1])
        estDifPos.append(f[0]-z[1])
        estPos.append(f[0])
        estVel.append(f[1])
        posVar = f[2]
        posBound3Sigma.append(3*np.sqrt(posVar[0][0]))

    return [measTime, measPos, estPos, estVel, measDifPos, estDifPos, posBound3Sigma];

##Plot Kalman Filter Results##
##The first plot below shows the position measurement error and estimate error relative to the actual position of the vehicle. This plot shows how the Kalman Filter smooths the input measurements and reduces the positional error.##
##The second plot shows the velocity estimate for the vehicle based on the input measurements. It can be seen that after the first five or so measurements the filter starts to settle on the vehicles actual speed which is 60 meters per second.##
t = testFilter()

plot1 = plt.figure(1)
plt.scatter(t[0], t[1])
plt.plot(t[0], t[2])
plt.ylabel('Position')
plt.xlabel('Time')
plt.grid(True)

plot2 = plt.figure(2)
plt.plot(t[0], t[3])
plt.ylabel('Velocity (meters/seconds)')
plt.xlabel('Update Number')
plt.title('Velocity Estimate On Each Measurement Update \n', fontweight="bold")
plt.legend(['Estimate'])
plt.grid(True)

plot3 = plt.figure(3)
plt.scatter(t[0], t[4], color = 'red')
plt.plot(t[0], t[5])
plt.legend(['Estimate', 'Measurement'])
plt.title('Position Errors On Each Measurement Update \n', fontweight="bold")
#plt.plot(t[0], t[6])
plt.ylabel('Position Error (meters)')
plt.xlabel('Update Number')
plt.grid(True)
plt.xlim([0, 300])
plt.show()



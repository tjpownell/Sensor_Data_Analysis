# Importing necessary libraries
from scipy import signal
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd

# Importing the data from our test file

data = pd.read_excel(r'W:\DATAFILES\TPOWNELL\Differential Speed Sensor Project\Testing\6-25-23\straight-1_filtered.xlsx') 
TimeData10ms = pd.DataFrame(data, columns=['TransmissionOutputShaftSpeed(Time)'])
TimeData50ms = pd.DataFrame(data, columns=['LH_Frequency(Time)'])
Left_Wheel_Speed = pd.DataFrame(data, columns=['LH_Frequency(Hz)'])
Right_Wheel_Speed = pd.DataFrame(data, columns=['RH_Frequency(Hz)'])
Transmission_Speed = pd.DataFrame(data, columns=['TransmissionOutputShaftSpeed(rpm)'])
Engage_Conditions = pd.DataFrame(data, columns=['Engage_Conditions(Y)'])
Input_Validation = pd.DataFrame(data, columns=['Input_Validation(Y)'])
Wheel_Speeds_Match = pd.DataFrame(data, columns=['Wheel_Speeds_Match(Y)'])
No_Spin_Condition = pd.DataFrame(data, columns=['No_Spin_Condition(Y)'])
Switch_Status = pd.DataFrame(data, columns=['Diff_Lock_Switch(Hz)'])
#print(Left_Wheel_Speed)

#rpm_to_mph = (33.6*60*2*3.1415)/(12*5280*5.4)
rpm_to_mph = 1
#rpm_to_hz = 2/60
rpm_to_hz = 1

#Convert the pandas dataframes to arrays
Engage_Conditions = Engage_Conditions.to_numpy()*0.75
Input_Validation = Input_Validation.to_numpy()*1.125
Wheel_Speeds_Match = Wheel_Speeds_Match.to_numpy()*0.5
No_Spin_Condition = No_Spin_Condition.to_numpy()*1
Switch_Status = Switch_Status.to_numpy()*1.15

Left_Wheel_Speed =  Left_Wheel_Speed.to_numpy()*(60/2)*rpm_to_mph*rpm_to_hz # Converting raw Hz data into RPM via the 60/2 conversion factor
Right_Wheel_Speed =  Right_Wheel_Speed.to_numpy()*(60/2)*rpm_to_mph*rpm_to_hz
Transmission_Speed = Transmission_Speed.to_numpy()*(1/3.36)*rpm_to_mph*rpm_to_hz # Converting raw rpm data to the differential output via the correction factor

# Also divide by 1000 to get the time in seconds
TimeData50ms = TimeData50ms.to_numpy()/1000
TimeData10ms = TimeData10ms.to_numpy()/1000

# Secondary Calculations

# Wheel Slippage Condition:
Wheel_Speed_Average = (Right_Wheel_Speed+Left_Wheel_Speed)/2
Wheel_Speed_Difference = Right_Wheel_Speed - Left_Wheel_Speed

# Algorithm for computing the X-Y coordinates of the machine during a test:
theta = [0]
C = 120
X_a_coord = [0]
Y_a_coord = [0]

X_b_coord = [C]
Y_b_coord = [0]


T_0 = 0
Theta_0 = 0

R = 33.6

#print(Left_Wheel_Speed[2]*2)

Left_Wheel_Speed = Left_Wheel_Speed[np.logical_not(np.isnan(Left_Wheel_Speed))]
Right_Wheel_Speed = Right_Wheel_Speed[np.logical_not(np.isnan(Right_Wheel_Speed))]
TimeData50ms = TimeData50ms[np.logical_not(np.isnan(TimeData50ms))]


#print(Left_Wheel_Speed)

LEFTSPEEDS = Left_Wheel_Speed.tolist()
RIGHTSPEEDS = Right_Wheel_Speed.tolist()
TIMES = TimeData50ms.tolist()



for x in range(len(Left_Wheel_Speed)):
    #print(x)
    # These velocities are magnitudes. The orientation will be determined by the Theta computed in the previous loop. Initial Theta is 0.
    
    V_a = LEFTSPEEDS[x]*R*0.1047198/5.4 # Convert rpm to rad/s, then divide by dropbox factor and multiply by tire radius to get tangential velocity
    V_b = RIGHTSPEEDS[x]*R*0.1047198/5.4


    Theta = np.arcsin(((V_b-V_a)*(TIMES[x] - T_0))/C)
    #print(V_b-V_a,"vel")
        
    #print(TIMES[x] - T_0)
    #print(V_a)
    #print(Theta*180/3.1415)

    # The velocity extraction and angular calculations are working correctly.

    deltaX_a = V_a*(TIMES[x] - T_0)*np.cos(3.1415/2 + (Theta_0))
    deltaY_a = V_a*(TIMES[x] - T_0)*np.sin(3.1415/2 + (Theta_0))

    deltaX_b = V_b*(TIMES[x] - T_0)*np.cos(3.1415/2 + (Theta_0))
    deltaY_b = V_b*(TIMES[x] - T_0)*np.sin(3.1415/2 + (Theta_0))

    X_a_coord.append(X_a_coord[x]+deltaX_a)
    Y_a_coord.append(Y_a_coord[x]+deltaY_a)

    X_b_coord.append(X_b_coord[x]+deltaX_b)
    Y_b_coord.append(Y_b_coord[x]+deltaY_b)

    T_0 = TIMES[x]
    Theta_0 = Theta + Theta_0
    print(deltaX_b)
    #print(Theta*180/3.1415)

# Convert Coordinates to ft, miles, or any other desired unit here:
for i in range(len(X_a_coord)):
    X_a_coord[i] = X_a_coord[i]/12
    X_b_coord[i] = X_b_coord[i]/12
    Y_a_coord[i] = Y_a_coord[i]/12
    Y_b_coord[i] = Y_b_coord[i]/12



# Set plot boundaries:
MAXES = [max(X_b_coord),max(X_a_coord),max(Y_a_coord),max(Y_b_coord)]
MINS = [min(X_b_coord),min(X_a_coord),min(Y_a_coord),min(Y_b_coord)]

print(len(X_a_coord))
print(len(Y_a_coord))
extra = 100

#print(X_a_coord)

# Plot the Results
#plot.plot(TimeData50ms,Left_Wheel_Speed,TimeData50ms,Right_Wheel_Speed,TimeData10ms,Transmission_Speed)
plot.plot(X_a_coord, Y_a_coord, "-b", label="Left Wheel Position")
plot.plot(X_b_coord, Y_b_coord, "-r", label="Right Wheel Position")

# Give x,y, title axis label
plot.xlabel('X [Feet]')
plot.ylabel('Y [Feet]')
plot.title('Machine Path [Overhead View]')
plot.xlim(min(MINS)-extra,max(MAXES)+extra)
plot.ylim(min(MINS)-extra,max(MAXES)+extra)
plot.legend(loc="upper right")
# Provide x axis and black line color
#plot.axhline(y=0, color='k')

# Display
plot.savefig(r'W:\DATAFILES\TPOWNELL\Differential Speed Sensor Project\6-14-23\PNG Charts\chart.png',dpi ='figure')
plot.show()
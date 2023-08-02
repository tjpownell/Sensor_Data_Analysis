# Importing necessary libraries
from scipy import signal
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd

# Importing the data from our test file

data = pd.read_excel(r'W:\DATAFILES\TPOWNELL\Differential Speed Sensor Project\Testing\6-27-23\full-test-two-turn_ept.xlsx') 
TimeData10ms = pd.DataFrame(data, columns=['TransmissionOutputShaftSpeed(Time)'])
TimeData50ms = pd.DataFrame(data, columns=['Engage_Conditions(Time)'])
TimeData250ms = pd.DataFrame(data, columns =['TransmissionSelectedGear(Time)'])
Left_Wheel_Speed = pd.DataFrame(data, columns=['LH_Frequency(Hz)'])
Right_Wheel_Speed = pd.DataFrame(data, columns=['RH_Frequency(Hz)'])
Transmission_Speed = pd.DataFrame(data, columns=['TransmissionOutputShaftSpeed(rpm)'])
Engage_Conditions = pd.DataFrame(data, columns=['Engage_Conditions(Y)'])
Disengage_Conditions = pd.DataFrame(data, columns=['Disengage_Conditions(Y)'])
Input_Validation = pd.DataFrame(data, columns=['Input_Validation(Y)'])
Wheel_Speeds_Match = pd.DataFrame(data, columns=['Wheel_Speeds_Match(Y)'])
No_Spin_Condition = pd.DataFrame(data, columns=['No_Spin_Condition(Y)'])
Switch_Status = pd.DataFrame(data, columns=['Diff_Lock_Switch(Hz)'])
WAS_vel = pd.DataFrame(data,columns=['YawRate(deg/s)'])
WAS_pos = pd.DataFrame(data,columns=['SteeringWheelAngle(deg)'])
Engine_Torque = pd.DataFrame(data,columns=['ActualEnginePercentTorque(%)'])
TimeData20ms = pd.DataFrame(data,columns=['ActualEnginePercentTorque(Time)'])
Current_Gear = pd.DataFrame(data,columns=['TransmissionSelectedGear(Y)'])
Shift_In_Process = pd.DataFrame(data,columns=['TransmissionShiftInProcess(Y)'])
#print(Left_Wheel_Speed)

#rpm_to_mph = (33.6*60*2*3.1415)/(12*5280*5.4)
rpm_to_mph = 1
rpm_to_hz = 2/60
#rpm_to_hz = 1



#Convert the pandas dataframes to arrays
# Scale so that they show up on graph:
scalar = 1
Engage_Conditions = Engage_Conditions.to_numpy()*0.75*scalar
#Input_Validation = Input_Validation.to_numpy()*1.125*scalar
#Wheel_Speeds_Match = Wheel_Speeds_Match.to_numpy()*0.5*scalar
#No_Spin_Condition = No_Spin_Condition.to_numpy()*1*scalar
#Switch_Status = Switch_Status.to_numpy()*1.15*scalar
#Disengage_Conditions = Disengage_Conditions.to_numpy()*1.3*scalar

#Left_Wheel_Speed =  Left_Wheel_Speed.to_numpy()*(60/2)*rpm_to_mph*rpm_to_hz # Converting raw Hz data into RPM via the 60/2 conversion factor
#Right_Wheel_Speed =  Right_Wheel_Speed.to_numpy()*(60/2)*rpm_to_mph*rpm_to_hz
Transmission_Speed = Transmission_Speed.to_numpy()*(1/3.36)*rpm_to_mph*rpm_to_hz # Converting raw rpm data to the differential output via the correction factor
Vehicle_Speed = Transmission_Speed*0.011018*60/2*3.36
Engine_Torque = Engine_Torque.to_numpy()/100

# Also divide by 1000 to get the time in seconds
TimeData50ms = TimeData50ms.to_numpy()/1000
TimeData10ms = TimeData10ms.to_numpy()/1000
TimeData20ms = TimeData20ms.to_numpy()/1000
TimeData250ms = TimeData250ms.to_numpy()/1000

# Secondary Calculations
# Wheel Slippage Condition:
Wheel_Speed_Average = (Right_Wheel_Speed+Left_Wheel_Speed)/2
Wheel_Speed_Difference = Right_Wheel_Speed - Left_Wheel_Speed

top_bar = []
lower_bar = []
for j in range(len(TimeData50ms)):
    top_bar.append(0.4)
    lower_bar.append(-0.5)


fig, (ax0, ax1, ax2) = plot.subplots(3, 1)

# Plot the Results
#ax0.plot(TimeData10ms,Transmission_Speed, label="Transmission Expected Speed")
#ax0.plot(TimeData50ms, Left_Wheel_Speed, "-m", label="Left Wheel Speed")
#ax0.plot(TimeData50ms, Right_Wheel_Speed, "-r", label="Right Wheel Speed")
ax0.plot(TimeData10ms, Vehicle_Speed, "k", label = "Vehicle Ground Speed")
#plot.plot(TimeData10ms, Transmission_Speed, "-g", label="Transmission Based Expected Speed")
#ax2.plot(TimeData50ms, Input_Validation, "-m", label="Input Validation Passed")
#ax2.plot(TimeData50ms, Engage_Conditions, "-b", label="Engage Conditions")
#ax2.plot(TimeData50ms, Disengage_Conditions, "-r", label="Disengage Conditions")
#plot.plot(TimeData50ms, Wheel_Speeds_Match, "-b", label="Wheel Speed Match")
#plot.plot(TimeData50ms, No_Spin_Condition, "-k", label="No Slip")
#plot.plot(TimeData50ms, Wheel_Speed_Average, "-c", label="Average Differential Output Speed")
#plot.plot(TimeData50ms, Wheel_Speed_Difference, "-c", label="Left Wheel Speed")
#plot.plot(TimeData50ms, top_bar,":k",label="Top Bar")
#plot.plot(TimeData50ms, lower_bar,':k',label="Low Bar")
#ax2.plot(TimeData50ms, Switch_Status, "-k", label = "Switch Status")
ax2.plot(TimeData20ms, Engine_Torque, "-r", label = "ECM % Torque")
ax2.plot(TimeData50ms,top_bar,"k", label = "Software Lock Limit")
ax1.plot(TimeData250ms,Current_Gear,"-k", label="Current_Gear")
ax1.plot(TimeData10ms,Shift_In_Process,"-g", label="Shift_In_Process")
ax0.legend(loc="upper right")
ax1.legend(loc="upper right")
ax2.legend(loc="lower right")


# Give x,y, title axis label
ax2.set_xlabel('Time [seconds]')
ax0.set_ylabel('Amplitude [MPH]')
ax1.set_ylabel('Amplitude [Bool] + [Gear]')
ax2.set_ylabel('% Engine Torque')

ax0.set_title('Differential Output Measurements')
ax1.set_title('Transmission Measurements')
ax2.set_title('Logical Response')

# Provide x axis and black line color
#plot.axhline(y=0, color='k')
# Display
#ax0.savefig(r'W:\DATAFILES\TPOWNELL\Differential Speed Sensor Project\6-14-23\PNG Charts\chart.png',dpi ='figure')
plot.show()
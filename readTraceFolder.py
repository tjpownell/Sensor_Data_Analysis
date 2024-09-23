import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import utils
from VariableData import *
from collections import deque
from numpy.fft import fft

class Traces:
	def __init__(self,file_path,res_path):
		self.file_path = file_path
		self.res_path = res_path
		self.trace_vars = []
		self.kinematicHeightRight = {}
		self.kinematicHeightLeft = {}
		self.kinematicHeightRack = {}
		self.heightStats = {}
		self.filter_size = 10
		self.groundHeightRightFiltered = {}
		self.groundHeightLeftFiltered = {}
		self.groundHeightRackFiltered = {}

		self.RightHeightError = {}
		self.RightPressureAdjust = {}
		self.Righttargetcurrent = {}
		self.RightPWM = {}

		self.LeftHeightError = {}
		self.LeftPressureAdjust = {}
		self.Lefttargetcurrent = {}
		self.LeftPWM = {}

		targetHeight = 60
		self.setpoint = targetHeight * 25 / 25.4

		self.file_name = os.path.basename(os.path.splitext(file_path)[0])

	def addTraceVariable(self,data,sensor_configData=None):
		trace_var = VariableData(data,self.file_path,self.res_path,sensor_configData)
		trace_var.extract_messages_from_trc()
		# trace_var.extract_messages_from_trc(printBool=True)
		# trace_var.getStats()
		trace_var.getStats(printBool=False)
		trace_var.generate_plot()
		trace_var.save_timestampedDataBH()

		self.trace_vars.append(trace_var)

	def apply_moving_average_filter(self, data_dict):
		timestamps = list(data_dict.keys())
		values = list(data_dict.values())
		
		queue = deque(maxlen=self.filter_size)
		sum_values = 0
		filtered_values = []

		for _, value in enumerate(values):
			if len(queue) == self.filter_size:
				sum_values -= queue.popleft()
			queue.append(value)
			sum_values += value

			if len(queue) == self.filter_size:
				filtered_values.append(sum_values / self.filter_size)
			else:
				filtered_values.append(np.nan)

		return dict(zip(timestamps, filtered_values))


	def saveSensorStats(self):
		fields = ["Variable Name", "Units", "Total Messages", "Message Frequency", "Min Val", "Max Val", "Avg Val", "Std Dev"]
		csv_filename = os.path.join(self.res_path, f"{self.file_name}_sensorStats.csv")

		with open(csv_filename, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(fields)

			for bh_var in self.trace_vars:
				if bh_var.stats:
					total_msgs, msg_freq, min_val, max_val, mean_val, stddev = bh_var.stats
					writer.writerow([
						bh_var.variable_name,
						bh_var.plotUnitsY,
						total_msgs,
						msg_freq,
						min_val,
						max_val,
						mean_val,
						stddev
					])

		# print(f"CSV file saved for {self.variable_name} in {csv_filename}")

	def saveHeightStats(self):
		height_stats = self.computeHeightStats()
		csv_filename = os.path.join(self.res_path, f"{self.file_name}_heightStats.csv")

		with open(csv_filename, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(height_stats[0].keys())

			for rows in height_stats:
				writer.writerow(rows.values())

		# print(f"CSV file saved for {self.variable_name} in {csv_filename}")

	def computeHeightStats(self):
		res = []
		for bh_var in self.trace_vars:
			if bh_var.variable_name == "RightRadarGroundHeight":
				right_sensor_stats = utils.compute_sensor_stats(self.setpoint,bh_var.int_messages.values())
				res.append({"Variable Name": "RightRadarGroundHeight", "Unit": "inch", **right_sensor_stats})
			elif bh_var.variable_name == "LeftRadarGroundHeight":
				left_sensor_stats = utils.compute_sensor_stats(self.setpoint,bh_var.int_messages.values())
				res.append({"Variable Name": "LeftRadarGroundHeight", "Unit": "inch", **left_sensor_stats})
			elif bh_var.variable_name == "RackRadarGroundHeight":
				rack_sensor_stats = utils.compute_sensor_stats(self.setpoint,bh_var.int_messages.values())
				res.append({"Variable Name": "RackRadarGroundHeight", "Unit": "inch", **rack_sensor_stats})
			else:
				continue
		return res
				
	def plot_ActiveLoopDiagnostics(self):
		
		var_dict = {
			"RightHeightError": "RightHeightError",
			"RightPressureAdjust": "RightPressureAdjust",
			"RightPWM": "RightPWM",
			"LeftHeightError": "LeftHeightError",
			"LeftPressureAdjust": "LeftPressureAdjust",
			"LeftPWM": "LeftPWM",
			"LeftRadarGroundHeight": "groundHeightLeft",
			"RightRadarGroundHeight": "groundHeightRight",
		}

		for bh_var in self.trace_vars:
			if bh_var.variable_name in var_dict:
				setattr(self, var_dict[bh_var.variable_name], bh_var)

		fig, (ax1, ax3, ax5) = plt.subplots(3,1,figsize=(12, 10))
		fig.suptitle(f"{self.file_name} Sensor Reading, PWM, Target = {self.setpoint:.2f} inch")

		plot_details = [
			(ax1, self.groundHeightRight, "-b", "Right Height (in)"),
			(ax3, self.groundHeightLeft, "-m", "Left Height (in)"),
			(ax5, self.groundHeightLeft, "-m", "Left Height (in)"),
			(ax5, self.groundHeightRight, "-b", "Right Height (in)")
		]

		for ax, data, color, label in plot_details:
			if data:
				ax.plot(data.int_messages.keys(), data.int_messages.values(), color, label=label)
    
		def compute_values(data):
			if data:
				values = data.int_messages.values()
				return (
					round(utils.compute_f_value(values, self.setpoint, utils.cm2inch(10)), 3),
					round(utils.compute_f_value(values, self.setpoint, utils.cm2inch(25)), 3),
					round(utils.compute_rmsd(values, self.setpoint), 2)
				)
			return (None, None, None)

		f10_left, f25_left, rmsd_left = compute_values(self.groundHeightLeft)
		f10_right, f25_right, rmsd_right = compute_values(self.groundHeightRight)
		
		rightheights = list(self.groundHeightRight.int_messages.values())
		rightheights = np.array(rightheights)
		avgRight = round(np.average(rightheights),2)

		leftheights = list(self.groundHeightLeft.int_messages.values())
		leftheights = np.array(leftheights)
		avgLeft = round(np.average(leftheights),2)

		ax1.axhline(y = self.setpoint, color = 'b', linestyle = '-') 
		ax3.axhline(y = self.setpoint, color = 'm', linestyle = '-') 
		
		annotations = [
			(f'F10 Left: {f10_left}', 'm') if f10_left else None,
			(f'F10 Right: {f10_right}', 'b') if f10_right else None,
			(f'F25 Left: {f25_left}', 'm') if f25_left else None,
			(f'F25 Right: {f25_right}', 'b') if f25_right else None,
			(f'RMSD L: {rmsd_left}', 'm') if rmsd_left else None,
			(f'RMSD R: {rmsd_right}', 'b') if rmsd_right else None,
			(f'Avg L: {avgLeft}', 'm') if avgLeft else None,
			(f'Avg R: {avgRight}', 'b') if avgRight else None
		]

		xlim = int(ax5.get_xlim()[1])
		
		def annotate(ax, annotations):
			y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
			y_min = ax.get_ylim()[0]

			start_fraction = 0.85
			spacing_fraction = 0.15
			y_start = y_min + y_range * start_fraction

			for text, color in annotations:
				if text:
					ax.text(xlim, y_start, text, fontsize=10, bbox=dict(facecolor=color, alpha=0.6))
					y_start -= y_range * spacing_fraction

		if not any(x is None for x in annotations):
			annotate(ax5, annotations)

		def highlight_range(ax, center, width, color, alpha):
			y1 = center + width
			y2 = center - width
			ax.axhspan(y1, y2, color=color, alpha=alpha, lw=0)
		
		highlight_range(ax1, self.setpoint, utils.cm2inch(10), 'green', 0.35)
		highlight_range(ax1, self.setpoint, utils.cm2inch(25), 'yellow', 0.25)
		highlight_range(ax3, self.setpoint, utils.cm2inch(10), 'green', 0.35)
		highlight_range(ax3, self.setpoint, utils.cm2inch(25), 'yellow', 0.25)
		highlight_range(ax5, self.setpoint, utils.cm2inch(10), 'green', 0.35)
		highlight_range(ax5, self.setpoint, utils.cm2inch(25), 'yellow', 0.25)

		for ax in [ax1, ax3, ax5]:
			ax.set_xlabel('Time (sec)')
			ax.legend()
			ax.grid()

		plot_filename = os.path.join(self.trace_vars[0].plot_path, f"Current_Loop_Results.png")
		plt.savefig(plot_filename)
		plt.close()

	def plot_SensorReadings(self):

		var_dict = {
			"RightRadarCanopyHeight": "RightCanopyHeight",
			"RightRadarCanopyConf": "RightCanopyConf",
			"LeftRadarCanopyHeight": "LeftCanopyHeight",
			"LeftRadarCanopyConf": "LeftCanopyConf",
			"LeftRadarGroundHeight": "groundHeightLeft",
			"RightRadarGroundHeight": "groundHeightRight",
			"LeftRadarGroundConf": "goundConfLeft",
			"RightRadarGroundConf": "goundConfRight",
		}

		for bh_var in self.trace_vars:
			if bh_var.variable_name in var_dict:
				setattr(self, var_dict[bh_var.variable_name], bh_var)
		
		fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize=(12, 10))
		fig.suptitle(f"{self.file_name} Kinematic Prediction vs Sensor Reading")

		ax1.plot(self.goundConfRight.int_messages.keys(), self.goundConfRight.int_messages.values(), "-k", label="Right Ground Conf (%)")
		ax1.plot(self.RightCanopyConf.int_messages.keys(), self.RightCanopyConf.int_messages.values(), "-r", label="Right Canopy Conf (%)")

		ax2.plot(self.RightCanopyHeight.int_messages.keys(), self.RightCanopyHeight.int_messages.values(), "-r", label="Right Canopy Dist (mm)")
		ax2.plot(self.groundHeightRight.int_messages.keys(), self.groundHeightRight.int_messages.values(), "-k", label="Right Ground Dist (mm)")

		ax3.plot(self.LeftCanopyHeight.int_messages.keys(), self.LeftCanopyHeight.int_messages.values(), "-m", label="Left Canopy Dist (mm)")
		ax3.plot(self.groundHeightLeft.int_messages.keys(), self.groundHeightLeft.int_messages.values(), "-b", label="Left Ground Dist (mm)")
		
		ax4.plot(self.goundConfLeft.int_messages.keys(), self.goundConfLeft.int_messages.values(), "-b", label="Left Ground Conf (%)")
		ax4.plot(self.LeftCanopyConf.int_messages.keys(), self.LeftCanopyConf.int_messages.values(), "-m", label="Left Canopy Conf (%)")
		
		setpoint = 100

		f50_left_gnd = round((utils.compute_f_value(self.goundConfLeft.int_messages.values(),setpoint,50)),3)
		f50_right_gnd = round(utils.compute_f_value(self.goundConfRight.int_messages.values(),setpoint,50),3)
		f50_left_can = round(utils.compute_f_value(self.LeftCanopyConf.int_messages.values(),setpoint,50),3)
		f50_right_can = round(utils.compute_f_value(self.RightCanopyConf.int_messages.values(),setpoint,50),3)

		for ax in [ax1, ax2, ax3, ax4]:
			ax.set_xlabel('Time (sec)')
			ax.legend()
			ax.grid()
		xlim = int(ax4.get_xlim()[1])-2

		ax4.text(int(xlim), 100, 'F50 Left Ground: '+str(f50_left_gnd), fontsize = 8, bbox = dict(facecolor = 'blue', alpha = 0.6))
		ax1.text(int(xlim), 100, 'F50 Right Ground: '+str(f50_right_gnd), fontsize = 8, bbox = dict(facecolor = 'black', alpha = 0.6))
		ax4.text(int(xlim), 88, 'F50 Left Canopy: '+str(f50_left_can), fontsize = 8,bbox = dict(facecolor = 'red', alpha = 0.6))
		ax1.text(int(xlim), 88, 'F50 Right Canopy: '+str(f50_right_can), fontsize = 8,bbox = dict(facecolor = 'red', alpha = 0.6))
		
		plot_filename = os.path.join(self.trace_vars[0].plot_path, f"Sensor_Readings.png")
		plt.savefig(plot_filename)
		plt.close()
	
	def plot_3DSensorReadings(self):

		var_dict = {
			"RightRadarCanopyHeight": "RightCanopyHeight",
			"RightRadarCanopyConf": "RightCanopyConf",
			"LeftRadarCanopyHeight": "LeftCanopyHeight",
			"LeftRadarCanopyConf": "LeftCanopyConf",
			"LeftRadarGroundHeight": "groundHeightLeft",
			"RightRadarGroundHeight": "groundHeightRight",
			"LeftRadarGroundConf": "goundConfLeft",
			"RightRadarGroundConf": "goundConfRight",
			"RackRadarGroundHeight": "groundHeightRack",
			"RackRadarCanopyHeight": "canopyHeightRack",
			
		}

		for bh_var in self.trace_vars:
			if bh_var.variable_name in var_dict:
				setattr(self, var_dict[bh_var.variable_name], bh_var)

		fig = plt.figure(figsize=(12, 10))
		ax = fig.add_subplot(projection='3d')

		colors = ['r', 'g', 'b']
		yticks = [0, 1, 2]

		rightheights = list(self.groundHeightRight.int_messages.values())
		rightheights = np.array(rightheights)
		rightTime = list(self.groundHeightRight.int_messages.keys())
		rightTime = np.array(rightTime)

		rackheights = list(self.groundHeightRack.int_messages.values())
		rackheights = np.array(rackheights)
		rackTime = list(self.groundHeightRack.int_messages.keys())
		rackTime = np.array(rackTime)
		
		leftheights = list(self.groundHeightLeft.int_messages.values())
		leftheights = np.array(leftheights)
		leftTime = list(self.groundHeightLeft.int_messages.keys())
		leftTime = np.array(leftTime)


		Tvals = [leftTime, rackTime, rightTime]
		Zvals = [leftheights, rackheights, rightheights]
		i=0
		for c, k in zip(colors, yticks):

			xs = Tvals[i]
			ys = Zvals[i]
			
			ax.plot(xs, ys, zs=k, zdir='y', color=c, alpha=0.5)
			i = i + 1

		ax.set_xlabel('Time [s]')
		ax.set_ylabel('Sensor')
		ax.set_zlabel('Radar Height [mm]')
		ax.set_zlim(0,100)
		ax.set_title(f"{self.file_name} 3D plot of Sensor Reading")

		# On the y-axis let's only label the discrete values that we have data for.
		ax.set_yticks(yticks)
		
		plot_filename = os.path.join(self.trace_vars[0].plot_path, f"Sensor_Readings_3D.png")
		plt.savefig(plot_filename)
		plt.close()

	def plot_LowPassFiltering(self):

		var_dict = {
			"RightHeightError": "RightError",
			"LeftHeightError": "LeftError",
			"RightTargetCurrent": "RightCurrent",
			"LeftTargetCurrent": "LeftCurrent",
			"RightPWM": "RightPWM",
			"LeftPWM": "LeftPWM",
		}
		e_binwidth = 40
		c_binwidth = 20

		for bh_var in self.trace_vars:
			if bh_var.variable_name in var_dict:
				setattr(self, var_dict[bh_var.variable_name], bh_var)
		
		fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize=(14, 12))
		fig.suptitle(f"{self.file_name} Error and Current Plots")

		# define the cut-off frequency
		RightCurrent = list(self.RightCurrent.int_messages.values())
		RightCurrent = np.array(RightCurrent)

		LeftCurrent = list(self.LeftCurrent.int_messages.values())
		LeftCurrent = np.array(LeftCurrent)
		
		RightError = list(self.RightError.int_messages.values())
		RightError = np.array(RightError)

		LeftError = list(self.LeftError.int_messages.values())
		LeftError = np.array(LeftError)

		previous = 0
		alpha = [0.25,0.125,0.0635,0.0317]
		alpha_cols = ["-b","-y","-g","-k"]
		k=0

		for j in alpha:
			LeftCurrentFiltered = []
			RightCurrentFiltered = []

			previous = 0
			for i in LeftCurrent:
				LeftCurrentFiltered.append(i*j + (1-j)*previous)
				previous = i*j + (1-j)*previous
			
			previous = 0
			for i in RightCurrent:
				RightCurrentFiltered.append(i*j + (1-j)*previous)
				previous = i*j + (1-j)*previous

			#print(len(self.RightCurrent.int_messages.keys()))
			#print(len(RightCurrentFiltered))

			ax2.plot(self.RightCurrent.int_messages.keys(), RightCurrentFiltered, alpha_cols[k], label="Right Current Filtered: " + str(alpha[k]))
			ax4.plot(self.LeftCurrent.int_messages.keys(), LeftCurrentFiltered, alpha_cols[k], label="Left Current Filtered:" + str(alpha[k]))

			k = k+1
		
		previous = 0
		alpha = [0.25,0.125,0.0635,0.0317]
		alpha_cols = ["-b","-y","-g","-k"]
		k=0

		for j in alpha:
			LeftCurrentFiltered = []
			RightCurrentFiltered = []

			previous = 0
			for i in LeftError:
				LeftCurrentFiltered.append(i*j + (1-j)*previous)
				previous = i*j + (1-j)*previous
			
			previous = 0
			for i in RightError:
				RightCurrentFiltered.append(i*j + (1-j)*previous)
				previous = i*j + (1-j)*previous

			#print(len(self.RightCurrent.int_messages.keys()))
			#print(len(RightCurrentFiltered))

			ax1.plot(self.RightCurrent.int_messages.keys(), RightCurrentFiltered, alpha_cols[k], label="Right Error Filtered: " + str(alpha[k]))
			ax3.plot(self.LeftCurrent.int_messages.keys(), LeftCurrentFiltered, alpha_cols[k], label="Left Error Filtered:" + str(alpha[k]))

			k = k+1

		
		ax1.plot(self.RightError.int_messages.keys(), RightError, "-r", label="Right Error (mm)")
		ax1.set_ylim(-500,500)
		
		ax2.plot(self.RightCurrent.int_messages.keys(), RightCurrent, "-r", label="Right Current (mA)")
		ax2.set_ylim(300,900)

		ax3.plot(self.LeftError.int_messages.keys(), LeftError, "-r", label="Left Error (mm)")
		ax3.set_ylim(-500,500)

		ax4.plot(self.LeftPWM.int_messages.keys(),  LeftCurrent, "-r", label="Left Current (mA)")
		ax4.set_ylim(300,900)

		for ax in [ax1, ax2, ax3, ax4]:
			ax.legend()
			ax.grid()
		
		plot_filename = os.path.join(self.trace_vars[0].plot_path, f"LowPass_Sensors_Current.png")
		plt.savefig(plot_filename)
		plt.close()

	def plot_SensorReadingsFFT(self):

		var_dict = {
			"RightRadarCanopyHeight": "RightCanopyHeight",
			"RightRadarCanopyConf": "RightCanopyConf",
			"LeftRadarCanopyHeight": "LeftCanopyHeight",
			"LeftRadarCanopyConf": "LeftCanopyConf",
			"LeftRadarGroundHeight": "groundHeightLeft",
			"RightRadarGroundHeight": "groundHeightRight",
			"LeftRadarGroundConf": "goundConfLeft",
			"RightRadarGroundConf": "goundConfRight",
		}

		for bh_var in self.trace_vars:
			if bh_var.variable_name in var_dict:
				setattr(self, var_dict[bh_var.variable_name], bh_var)
		
		fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize=(12, 10))
		fig.suptitle(f"{self.file_name} FFT Plots of Sensor Reading")

		# define the cut-off frequency
		
		rightheights = list(self.groundHeightRight.int_messages.values())
		rightheights = np.array(rightheights)
		X_right = fft(rightheights)
		X_right = np.delete(X_right, 0)
		
		sampling_rate = 50
		N = len(X_right)
		n = np.arange(N)
		T = N/sampling_rate
		freq = n/T 

		ax1.stem(freq, np.abs(X_right), 'k', markerfmt=" ", basefmt="-k",label="Right Ground FFT")
		ax1.set_xlim(0,5)
		ax1.set_ylim(0,np.max(np.abs(X_right))*1.2)
		ax1.set_xlabel("Freq (Hz)")
		
		ax2.plot(self.RightCanopyHeight.int_messages.keys(), self.RightCanopyHeight.int_messages.values(), "-r", label="Right Canopy Dist (mm)")
		ax2.plot(self.groundHeightRight.int_messages.keys(), self.groundHeightRight.int_messages.values(), "-k", label="Right Ground Dist (mm)")
		ax2.set_xlabel("Time (s)")

		ax3.plot(self.LeftCanopyHeight.int_messages.keys(), self.LeftCanopyHeight.int_messages.values(), "-m", label="Left Canopy Dist (mm)")
		ax3.plot(self.groundHeightLeft.int_messages.keys(), self.groundHeightLeft.int_messages.values(), "-b", label="Left Ground Dist (mm)")
		ax3.set_xlabel("Time (s)")
		
		leftheights = list(self.groundHeightLeft.int_messages.values())
		leftheights = np.array(leftheights)
		X_left = fft(leftheights)
		X_left = np.delete(X_left, 0)
		
		N = len(X_left)
		n = np.arange(N)
		T = N/sampling_rate
		freq = n/T 

		
		ax4.stem(freq, np.abs(X_left), 'b', markerfmt=" ", basefmt="-b", label="Left Ground FFT")
		ax4.set_xlim(0,5)
		ax4.set_ylim(0,np.max(np.abs(X_left))*1.2)
		
		for ax in [ax1, ax2, ax3, ax4]:
			# ax.set_ylim(0,1.1*y_max)
			# ax.set_xlim(0,)
			# ax.set_xlabel('Time (sec)')
			# ax.set_ylabel('Height (inch)')
			ax.legend()
			ax.grid()
		xlim = int(ax4.get_xlim()[1])-2
		#print(xlim)
		
		plot_filename = os.path.join(self.trace_vars[0].plot_path, f"FFT.png")
		plt.savefig(plot_filename)
		plt.close()

	def plot_Histograms(self):

		var_dict = {
			"RightHeightError": "RightError",
			"LeftHeightError": "LeftError",
			"RightTargetCurrent": "RightCurrent",
			"LeftTargetCurrent": "LeftCurrent",
			"RightPWM": "RightPWM",
			"LeftPWM": "LeftPWM",
		}
		e_binwidth = 40
		c_binwidth = 20

		for bh_var in self.trace_vars:
			if bh_var.variable_name in var_dict:
				setattr(self, var_dict[bh_var.variable_name], bh_var)
		
		fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize=(14, 12))
		fig.suptitle(f"{self.file_name} Histogram of Output and Sensor Reading")

		# define the cut-off frequency
		
		RightError = list(self.RightError.int_messages.values())
		RightError = np.array(RightError)
		
		ax1.hist(RightError, bins=range(min(RightError), max(RightError) + e_binwidth, e_binwidth), color='teal', edgecolor='black', label="Right Errors")
		ax1.set_ylabel("Frequency")
		ax1.set_xlabel("Error Value [mm]")

		RightPWM = list(self.RightPWM.int_messages.values())
		RightPWM = np.array(RightPWM)
		
		ax2.hist(RightPWM, bins=range(min(RightPWM), max(RightPWM) + c_binwidth, c_binwidth), color='skyblue', edgecolor='black', label="Right PWM")
		ax2.set_ylabel("Frequency")
		ax2.set_xlabel("PWM Value [0.01%]")

		LeftPWM = list(self.LeftPWM.int_messages.values())
		LeftPWM = np.array(LeftPWM)
		
		ax3.hist(LeftPWM, bins=range(min(LeftPWM), max(LeftPWM) + c_binwidth, c_binwidth), color='magenta', edgecolor='black', label="Left PWM")
		ax3.set_ylabel("Frequency")
		ax3.set_xlabel("PWM Value [0.01%]")

		LeftError = list(self.LeftError.int_messages.values())
		LeftError = np.array(LeftError)

		ax4.hist(LeftError, bins=range(min(LeftError), max(LeftError) + e_binwidth, e_binwidth), color='crimson', edgecolor='black', label="Left Errors")
		ax4.set_ylabel("Frequency")
		ax4.set_xlabel("Error Value [mm]")
		
		for ax in [ax1, ax2, ax3, ax4]:
			ax.legend()
			ax.grid()
		
		plot_filename = os.path.join(self.trace_vars[0].plot_path, f"Histograms.png")
		plt.savefig(plot_filename)
		plt.close()

	def plot_PWM_FFT(self):

		var_dict = {
			"RightPWM": "RightPWM",
			"LeftPWM": "LeftPWM",
			"RightTargetCurrent": "RightCurrent",
			"LeftTargetCurrent": "LeftCurrent",
		}

		for bh_var in self.trace_vars:
			if bh_var.variable_name in var_dict:
				setattr(self, var_dict[bh_var.variable_name], bh_var)
		
		fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize=(12, 10))
		fig.suptitle(f"{self.file_name} FFT Plots of Sensor Reading")

		# define the cut-off frequency
		
		rightCurrent = list(self.RightCurrent.int_messages.values())
		rightCurrent = np.array(rightCurrent)
		X_right = fft(rightCurrent)
		X_right = np.delete(X_right, 0)
		
		sampling_rate = 50
		N = len(X_right)
		n = np.arange(N)
		T = N/sampling_rate
		freq = n/T 

		ax1.stem(freq, np.abs(X_right), 'k', markerfmt=" ", basefmt="-k",label="Right Current FFT")
		ax1.set_xlim(0,10)
		ax1.set_ylim(0,np.max(np.abs(X_right))*1.2)
		ax1.set_xlabel("Freq (Hz)")
		
		ax2.plot(self.RightCurrent.int_messages.keys(), self.RightCurrent.int_messages.values(), "-r", label="Right Current (mA)")
		ax2.set_xlabel("Time (s)")
		#ax2.set_ylim(2800,5000)

		ax3.plot(self.LeftCurrent.int_messages.keys(), self.LeftCurrent.int_messages.values(), "-m", label="Left Current (mA)")
		ax3.set_xlabel("Time (s)")
		#ax3.set_ylim(2800,5000)

		leftCurrent = list(self.LeftCurrent.int_messages.values())
		leftCurrent = np.array(leftCurrent)
		X_left = fft(leftCurrent)
		X_left = np.delete(X_left, 0)
		
		N = len(X_left)
		n = np.arange(N)
		T = N/sampling_rate
		freq = n/T 

		
		ax4.stem(freq, np.abs(X_left), 'b', markerfmt=" ", basefmt="-b", label="Left Current FFT")
		ax4.set_xlim(0,10)
		ax1.set_ylim(0,np.max(np.abs(X_left))*1.2)
		ax4.set_xlabel("Freq (Hz)")
		
		for ax in [ax1, ax2, ax3, ax4]:
			ax.legend()
			ax.grid()
		xlim = int(ax4.get_xlim()[1])-2
		#print(xlim)
		
		plot_filename = os.path.join(self.trace_vars[0].plot_path, f"Current_FFT.png")
		plt.savefig(plot_filename)
		plt.close()


def main():
	if len(sys.argv) != 2:
		print("Usage: python readTraceFile.py <trc_file_folder>")
		sys.exit(1)

	machine_config_fname = 'SensorConfig.csv'
	machine_data = utils.read_machineConfig(machine_config_fname)
	machineVariableList = list(machine_data.keys())
	print("Variable List from machine config file:\n",machineVariableList)

	config_fname = 'PlotterConfig.csv'
	config_data = utils.read_configfile(config_fname)

	variableList = list(config_data.keys())
	# print("Variable List from config file:\n",variableList)

	folder_name = sys.argv[1]
	folder_path = os.path.join(os.getcwd(), folder_name)
	
	result_path = os.path.join(folder_path,"Results")
	os.makedirs(result_path, exist_ok=True)
		
	file_name_list = sorted([x for x in os.listdir(folder_name)])
	for file_name in file_name_list:
		if file_name.endswith(".trc"):
			file_path = os.path.join(folder_path,file_name)
			print(f"Processing {os.path.basename(file_path)}")
			trace_manager = Traces(file_path,result_path)
			for target_name in variableList:
				sensor_configData = machine_data[target_name] if target_name in machineVariableList else None
				trace_manager.addTraceVariable(config_data[target_name],
												sensor_configData
				)

			trace_manager.plot_ActiveLoopDiagnostics()
			trace_manager.plot_SensorReadings()
			trace_manager.plot_SensorReadingsFFT()
			trace_manager.saveSensorStats()
			trace_manager.saveHeightStats()
			trace_manager.plot_Histograms()
			trace_manager.plot_PWM_FFT()
			trace_manager.plot_LowPassFiltering()
			trace_manager.plot_3DSensorReadings()

	output_file = os.path.join(folder_path, 'cumulative_heightStats.csv')

	utils.create_cumulative_csv(result_path, output_file)

if __name__ == "__main__":
	main()

import os
import numpy as np
import matplotlib.pyplot as plt
from utils import data_from_pgn, mm2inch, mv2unit, save_timestampedData,get_rackPitchHeight

class VariableData:
	def __init__(self,data,file_path,res_path,sensor_configData=None):
		self.variable_name = data['Variable_name']
		self.msgIDrx = data['msgIDrx']
		self.msgBits = data['msgBits']
		self.endianness = data['Endianness']
		self.max_value = data['Max']
		self.min_value = data['Min']
		self.unit = data['Unit']
		self.plotUnitsY = data['PlotUnitsY']
		self.plotUnitsX = data['PlotUnitsX']
		self.plotBool = data['PlotBool']
		self.file_path = file_path
		self.sensor_configData = self.SensorConfig(sensor_configData) if sensor_configData is not None else None
		self.hex_messages = {}
		self.int_messages = {}
		self.stats = None
		self.fname = os.path.basename(os.path.splitext(file_path)[0])
		self.res_path = os.path.join(res_path,f"{self.fname}_results")
		self.mk_resultsDirs()

	def extract_messages_from_trc(self,printBool=False):
		min_timestamp = float('inf') 

		with open(self.file_path, 'r') as file:
			for line in file:
				if line.strip().startswith(';') or line.strip() == "":
					continue  
				
				fields = line.strip().split()

				if len(fields) >= 5 and fields[4].startswith('['):
					try:
						timestamp = float(fields[0])
						if timestamp < min_timestamp:
							min_timestamp = timestamp
					except ValueError:
						continue  

		with open(self.file_path, 'r') as file:
			for line in file:
				if line.strip().startswith(';') or line.strip() == "":
					continue  
				
				fields = line.strip().split()

				try:
					if fields[4].startswith('['):
						# Logged with the ActiveControl UI
						message_id = fields[3]
						raw_msg = fields[5:]  
						timestamp = (float(fields[0]) - min_timestamp) / 1000000  	# microseconds to milliseconds 
					else:
						# Logged with PCAN
						message_id = fields[4]
						raw_msg = fields[7:]  
						timestamp = float(fields[1]) / 1000  						# milliseconds to seconds
				except IndexError:
					continue  
				
				if len(fields) >= 9:

					try:
						if int(message_id.lower(),16) == self.msgIDrx:
							int_message = self.convert_data(data_from_pgn(raw_msg,self.msgBits,self.endianness))
							self.hex_messages[timestamp] = raw_msg
							self.int_messages[timestamp] = int_message
						if printBool:
							self.printTypeVariable()
					except ValueError:
						print(f"Skipping invalid message_id: {message_id}")

	def convert_data(self,num):
		if self.unit != self.plotUnitsY:
			if self.unit == 'mm' and self.plotUnitsY == 'inch':
				return mm2inch(num)
			elif self.variable_name == "RackPitchTiltSensor":
				return get_rackPitchHeight(num)
			elif self.unit == 'mV' and (self.plotUnitsY == 'deg' or self.plotUnitsY == 'inch'):
				assert self.sensor_configData.unit == self.plotUnitsY, "CHECK CONFIG FILE UNITS"
				return mv2unit(num,self.sensor_configData)
			else:
				print("CHECK CONFIG FILE UNITS")
				pass
		elif self.unit == "Percent":
			return 100 if num>100 else num
		elif self.unit == "unit":
			return num
		else:
			print(f"No change, input: {num}, self.unit = self.plotUnitsY: {self.unit}")
	
	def printTypeVariable(self):
		if 'height' in self.variable_name.lower():
			print(f"{self.variable_name} is Height data")
		elif 'tilt' in self.variable_name.lower():
			print(f"{self.variable_name} is Angle data")
		elif 'conf' in self.variable_name.lower():
			print(f"{self.variable_name} is Confidence data")
		else:
			print(f"ERROR: Variable name is {self.variable_name}")

	def getStats(self,printBool=True):
		total_msgs = len(self.hex_messages)
		if total_msgs:
			mean_val = round(np.mean(list(self.int_messages.values())),3)
			stddev = round(np.std(list(self.int_messages.values())),3)
			min_val = round(min(self.int_messages.values()),3)
			max_val = round(max(self.int_messages.values()),3)
			timestamps = [float(x) for x in self.int_messages.keys()]
			# print("max(timestamps)",max(timestamps),"min(timestamps)",min(timestamps))
			msg_freq = round(total_msgs / ((max(timestamps) - min(timestamps))),3)

			self.stats = [total_msgs,msg_freq,min_val,max_val,mean_val,stddev]

			if printBool:
				print(f"Found {total_msgs} messages for {self.variable_name} with ID {hex(self.msgIDrx)[2:].upper()}")
				print(f"Mean {mean_val:.2f} {self.plotUnitsY} StdDev: {stddev:.2f} {self.plotUnitsY}")
				print(min_val,max_val,msg_freq)
			
		else:
			print(f"No messages found with ID {hex(self.msgIDrx)[2:].upper()} in the file.")

	def generate_plot(self):
		if self.plotBool:
			timestamps = np.array([x for x in self.int_messages.keys()])
			
			values = np.array(list(self.int_messages.values()))

			plt.figure(figsize=(10, 6))
			plt.plot(timestamps, values, marker='o', linestyle='-', color='b')
			plt.title(f"{self.variable_name} over Time")
			plt.xlabel(self.plotUnitsX)
			plt.ylabel(self.plotUnitsY)
			# plt.xlim(0,)
			# plt.ylim(0,)
			plt.grid(True)
			plt.tight_layout()

			plot_filename = os.path.join(self.plot_path, f"{self.variable_name}_plot.png")
			plt.savefig(plot_filename)
			plt.close()
		else:
			print(f"Plotting is disabled for {self.variable_name}. Set PlotBool to True in the configuration to enable plotting.")
	
	def save_timestampedDataBH(self):
		save_timestampedData(self.rawData_path,self.variable_name,self.plotUnitsY,self.int_messages)

	def mk_resultsDirs(self):
		os.makedirs(self.res_path, exist_ok=True)
		self.plot_path = os.path.join(self.res_path,"Plots")
		self.rawData_path = os.path.join(self.res_path,"Raw_Data")
		os.makedirs(self.plot_path, exist_ok=True)
		os.makedirs(self.rawData_path, exist_ok=True)

	class SensorConfig:
		def __init__(self, sensor_configData):
			if sensor_configData:
				self.variable_name = sensor_configData['Variable_name']
				self.mvMax = sensor_configData['mvMax']
				self.mvMin = sensor_configData['mvMin']
				self.unitMax = sensor_configData['unitMax']
				self.unitMin = sensor_configData['unitMin']
				self.unit = sensor_configData['unit']
			else:
				self.mvMax = None
				self.mvMin = None
				self.unitMax = None
				self.unitMin = None
				self.unit = None
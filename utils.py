import os
import csv
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import pandas as pd

def read_configfile(filename):
	data = {}
	readme_path = os.path.join(os.getcwd(), filename)
	
	try:
		with open(readme_path, 'r', newline='') as readme_file:
			csv_reader = csv.DictReader(readme_file)
			for row in csv_reader:
				key = row['Variable_name']
				data[key] = {}
				for field, value in row.items():
					if field == 'msgIDrx':
						data[key][field] = int(value, 16)
					elif field in ['startBit', 'endBit', 'Endianness','Max', 'Min']:
						data[key][field] = int(value)
					elif field in ['PlotBool']:
						data[key][field] = str(value)
					else:
						data[key][field] = value
	
				data[key]['msgBits'] = [int(row['startBit']), int(row['endBit'])]

	except FileNotFoundError:
		print(f"The file {filename} was not found.")
	except Exception as e:
		print(f"An error occurred: {e}")
	
	return data

def read_machineConfig(filename):
	data = {}
	readme_path = os.path.join(os.getcwd(), filename)
	
	try:
		with open(readme_path, 'r', newline='') as readme_file:
			csv_reader = csv.DictReader(readme_file)
			for row in csv_reader:
				key = row['Variable_name']
				data[key] = {field: (int(value) if field in ['mvMax', 'mvMin'] else
									float(value) if field in ['unitMax', 'unitMin'] else
									value)
							for field, value in row.items()}
	except FileNotFoundError:
		print(f"The file {filename} was not found.")
	except Exception as e:
		print(f"An error occurred: {e}")
	
	return data

def data_from_pgn(recData, msgBits, endianness):
	startBit, endBit = msgBits
	startByte = startBit // 8
	endByte = endBit // 8
	startBitOffset = startBit % 8
	endBitOffset = endBit % 8

	relevantBytes = recData[startByte:endByte + 1]
	byteValues = [int(byte, 16) for byte in relevantBytes]
	
	combinedValue = 0
	
	if endianness == 1:  # Little-endian
		for i, byte in enumerate(byteValues):
			combinedValue |= byte << (i * 8)
	else:  # Big-endian
		for i, byte in enumerate(byteValues):
			combinedValue = (combinedValue << 8) | byte  
	
	combinedValue = combinedValue >> startBitOffset
	bitLength = endBit - startBit + 1
	mask = (1 << bitLength) - 1

	if bitLength == 16: # 16 bit integers are signed. (Using two's complement notation)
		
		if (combinedValue & (1 << (bitLength - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
			combinedValue = combinedValue - (1 << bitLength)        # compute negative value
		
	result = combinedValue

	return result

def convert_hex_to_ascii(h):
	chars_in_reverse = []
	while h != 0x0:
		chars_in_reverse.append(chr(h & 0xFF))
		h = h >> 8

	chars_in_reverse.reverse()
	return ''.join(chars_in_reverse)

def mm2inch(mm_measurement):
	return mm_measurement*0.0393700787

def inch2mm(inch_measurement):
	return inch_measurement*25.4

def cm2inch(cm_measurement):
	return cm_measurement/2.54

def mv2unit(mv_measurement,sensor_configData):
	slope = (sensor_configData.unitMax - sensor_configData.unitMin)/(sensor_configData.mvMax - sensor_configData.mvMin)
	res = mv_measurement * slope + sensor_configData.unitMin - slope * sensor_configData.mvMin
	return res

def get_rackPitchHeight(x):
	'''
	Input is in mV
	Output is in inches
	'''
	x /= 1000
	return 3.8161 * x**3 - 32.032 * x**2 + 56.837 * x + 65.204

def save_timestampedData(output_path,variable_name,yUnits,messages_map):
	csv_filename = os.path.join(output_path,f"{variable_name}_data.csv")

	with open(csv_filename, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(["Timestamp (sec)", f"Value ({yUnits})"])

		for timestamp, value in messages_map.items():
			writer.writerow([timestamp, value])

def pad_sequences(reference, measured):
	max_len = max(len(reference), len(measured))
	reference_padded = np.pad(reference, (0, max_len - len(reference)), 'constant', constant_values=np.nan)
	measured_padded = np.pad(measured, (0, max_len - len(measured)), 'constant', constant_values=np.nan)
	return reference_padded, measured_padded


def compute_sensor_stats(reference, measured):
	measured = np.array(list(measured))
	error = np.array([m - reference for m in measured])
	f10 = round(compute_f_value(measured, reference, cm2inch(10)), 3)
	f25 = round(compute_f_value(measured, reference, cm2inch(25)), 3) - f10
	f40 = round(compute_f_value(measured, reference, cm2inch(40)), 3) - f10 - f25
	f40more = 1 - (f10 + f25 + f40)
	hIndex = f10 + 0.75 * f25 + 0.25 * f40 - f40more
	
	res = {
		"Mean": round(np.nanmean(measured),3),
		"Mean Error": round(np.nanmean(error),3),
		"Mean Abs Error": round(np.nanmean(abs(error)),3),
		"RMSD": round(compute_rmsd(measured, reference),3),
		"Max Negative Error": round(min(error),3),
		"Max Positive Error": round(max(error),3),
		"Error Swing": round(max(error) - min(error),3),

		"F10": f10,
		"F25": f25,
		"F40": f40,
		"F>40": f40more,
		"H-Index": round(hIndex,3),
	}

	return res

def compute_f_value(ListOfData,Setpoint,Width):
	n = 0
	length = int(len(ListOfData))
	for i in list(ListOfData)[:length]:

		if ((i < Setpoint+Width) and (i > Setpoint-Width)):
			n = n + 1

	return n/length

def compute_rmsd(ListOfData, Setpoint):
    if not isinstance(ListOfData, (list, tuple)):
        ListOfData = list(ListOfData)
    
    length = len(ListOfData)
    squared_diffs = [(i - Setpoint) ** 2 for i in ListOfData[:length]]
    mean_squared_diff = sum(squared_diffs) / length
    rmsd = math.sqrt(mean_squared_diff)
    return rmsd

def create_cumulative_csv(folder_path, output_file):
    cumulative_data = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith("heightStats.csv"):
            file_path = os.path.join(folder_path, file_name)
            
            df = pd.read_csv(file_path)
            
            df.insert(0, 'Filename', file_name)
            
            cumulative_data.append(df)

    if cumulative_data:
        final_df = pd.concat(cumulative_data, ignore_index=True)
        
        final_df.to_csv(output_file, index=False)
        # print(f"Cumulative CSV saved to {output_file}")
    else:
        print("No heightStats.csv files found in the folder.")

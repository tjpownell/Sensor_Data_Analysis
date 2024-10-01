# Object Oriented Approach to Sensor Data Analysis
###  This repository contains an example of a way to automate the analysis of large quantities of time-series data using Python.
> The data is expected in a .trc file, similar to those generate by PCAN.
> To use the readTraceFolder.py script, pass as an argument a subfolder containing the .trc files you wish to analyze.

## Dependencies
- Python - Matplotlib, Numpy
- SensorConfig.csv (Must contain ADC conversion information for analog signals transmitted via CAN)
- PlotterConfig.csv (Must contain the CAN message definitions for each variable)


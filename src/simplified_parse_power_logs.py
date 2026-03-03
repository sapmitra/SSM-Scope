'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-02 03:06:18
 # @ Description:
 '''

# import os
# import csv
# import pandas

# power_data = pandas.read_csv("power.log")

# gpu0 = power_data[power_data["index"] == 0]
# gpu1 = power_data[power_data["index"] == 1]
# gpu2 = power_data[power_data["index"] == 2]
# gpu3 = power_data[power_data["index"] == 3]

# average_power_gpu0 = gpu0[" power.draw [W]"].mean()
# average_power_gpu1 = gpu1[" power.draw [W]"].mean()
# average_power_gpu2 = gpu2[" power.draw [W]"].mean()
# average_power_gpu3 = gpu3[" power.draw [W]"].mean()

# average_energy_gpu0 = gpu0[" power.draw [W]"].sum()
# average_energy_gpu1 = gpu1[" power.draw [W]"].sum()
# average_energy_gpu2 = gpu2[" power.draw [W]"].sum()
# average_energy_gpu3 = gpu3[" power.draw [W]"].sum()

# average_memory_usage_gpu0 = gpu0[" memory.used [MiB]"].mean()
# average_memory_usage_gpu1 = gpu1[" memory.used [MiB]"].mean()
# average_memory_usage_gpu2 = gpu2[" memory.used [MiB]"].mean()
# average_memory_usage_gpu3 = gpu3[" memory.used [MiB]"].mean()

# print(f"GPU0: avg power = {average_power_gpu0} \t energy = {average_energy_gpu0} \t memory usage = {average_memory_usage_gpu0}")
# print(f"GPU1: avg power = {average_power_gpu1} \t energy = {average_energy_gpu1} \t memory usage = {average_memory_usage_gpu1}")
# print(f"GPU2: avg power = {average_power_gpu2} \t energy = {average_energy_gpu2} \t memory usage = {average_memory_usage_gpu2}")
# print(f"GPU3: avg power = {average_power_gpu3} \t energy = {average_energy_gpu3} \t memory usage = {average_memory_usage_gpu3}")

import os
import csv
import pandas
import glob


def clean_numeric(column):
    return column.str.replace(r"[^\d.]", "", regex=True).astype(float)

# Apply cleaning to all relevant columns
numeric_columns = [" power.draw [W]", " memory.used [MiB]", " utilization.memory [%]", " utilization.gpu [%]"]


# power_data = pandas.read_csv("power.log")

# Get all log files in power_logs directory
log_files = glob.glob("power_logs/*.log")

for log_file in sorted(log_files):
    # Extract metadata from filename
    filename = os.path.basename(log_file)
    # parts = filename.replace(".log", "").split("_")
    
    # if len(parts) >= 4:
    #     model_name = parts[0]
    #     cuda_version = parts[1]
    #     batch_size = parts[2]
    #     seq_len = parts[3]
    # else:
    #     print(f"Skipping {filename} - unexpected format")
    #     continue
    
    print(f"\n{'='*80}")
    print(f"Processing: {filename}")
    print(f"{'='*80}")
    
    power_data = pandas.read_csv(log_file)
    
    print(power_data.head())
    
    gpu0 = power_data[power_data["index"] == 0]
    gpu1 = power_data[power_data["index"] == 1]
    gpu2 = power_data[power_data["index"] == 2]
    gpu3 = power_data[power_data["index"] == 3]
    
    print('GPU0')
    print(gpu0.tail())
    print(len(gpu0))
    gpu0[numeric_columns] = gpu0[numeric_columns].apply(clean_numeric)
    gpu1[numeric_columns] = gpu1[numeric_columns].apply(clean_numeric)
    gpu2[numeric_columns] = gpu2[numeric_columns].apply(clean_numeric)
    gpu3[numeric_columns] = gpu3[numeric_columns].apply(clean_numeric)
    print(gpu0.head())
    
    
    average_power_gpu0 = gpu0[" power.draw [W]"].mean()
    print (average_power_gpu0)
    average_power_gpu1 = gpu1[" power.draw [W]"].mean()
    average_power_gpu2 = gpu2[" power.draw [W]"].mean()
    average_power_gpu3 = gpu3[" power.draw [W]"].mean()
    
    ## 0.1 seconds is the sampling time 
    ## 1000 is the number of inference runs
    # average_energy_gpu0 = gpu0[" power.draw [W]"].sum() * 0.1 / 1000
    # average_energy_gpu1 = gpu1[" power.draw [W]"].sum() * 0.1 / 1000
    # average_energy_gpu2 = gpu2[" power.draw [W]"].sum() * 0.1 / 1000
    # average_energy_gpu3 = gpu3[" power.draw [W]"].sum() * 0.1 / 1000 

    average_energy_gpu0 = gpu0[" power.draw [W]"].sum() * 0.1 / 50
    average_energy_gpu1 = gpu1[" power.draw [W]"].sum() * 0.1 / 50
    average_energy_gpu2 = gpu2[" power.draw [W]"].sum() * 0.1 / 50
    average_energy_gpu3 = gpu3[" power.draw [W]"].sum() * 0.1 / 50
    
    print(f"GPU0: avg power = {average_power_gpu0:.2f} W \t energy = {average_energy_gpu0:.2f} J")
    print(f"GPU1: avg power = {average_power_gpu1:.2f} W \t energy = {average_energy_gpu1:.2f} J")
    print(f"GPU2: avg power = {average_power_gpu2:.2f} W \t energy = {average_energy_gpu2:.2f} J")
    print(f"GPU3: avg power = {average_power_gpu3:.2f} W \t energy = {average_energy_gpu3:.2f} J")

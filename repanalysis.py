#! /usr/bin/env python3

# Imports
import numpy as np
import pandas as pd
import scipy.signal
import scipy.fft
import math

from formanalysis import analyze_test_data
# Error
class DataError(Exception):
    """
    Data file had unexpected format
    """

def import_data(file_name : str):
    """
    Imports data from the given file

    Returns: two data frames, one for accelerometer data and one for gyroscope data
    """
    with open(file_name) as file:
        accel_indices = []
        gyro_indices = []
        accel_data = []
        gyro_data = []
        for line in file:
            vals = line.split(',')
            if len(vals) < 5: # Invalid
                continue
            data_type = vals[0]
            time = int(vals[1])
            data = [float(_) for _ in vals[2:]]
            # Popula
            if data_type == 'a':
                accel_indices.append(time)
                accel_data.append({
                    'x' : data[0],
                    'y' : data[1],
                    'z' : data[2],
                })
            elif data_type == 'g':
                gyro_indices.append(time)
                gyro_data.append({
                    'x' : data[0],
                    'y' : data[1],
                    'z' : data[2],
                })
            else:
                raise DataError("Invalid line type")
    accel_indices = pd.TimedeltaIndex(accel_indices)
    gyro_indices = pd.TimedeltaIndex(gyro_indices)
    accel_frame = pd.DataFrame(accel_data, index=accel_indices)
    gyro_frame = pd.DataFrame(gyro_data, index=gyro_indices)
    accel_frame.index.name = 'time'
    gyro_frame.index.name = 'time'
    return accel_frame, gyro_frame

def preprocess_data(accel : pd.DataFrame, gyro : pd.DataFrame):
    """
    Sort by ascending order of timestamp and subtracts the start time from each time stamp.
    Performs this in place.
    """
    # accel.sort_values('time')
    # gyro.sort_values('time')
    start_time = min(accel.index.min(), gyro.index.min())
    accel.index -= start_time
    gyro.index -= start_time

def linear_interpolate(data : pd.DataFrame, interval : int = int(1e7)):
    """
    Linearly interpolates the accelerometer and gyroscope data to a regular time interval

    interval : interval to interpolate to in nanoseconds

    Returns: new pandas data frame
    """
    resampled = data.resample(pd.Timedelta(interval)).mean()
    resampled = resampled.interpolate(method='linear')
    return resampled

# Analysis functions

def get_rms(arr, window_size):
    """
    Efficiently gets the rms average for a sliding window over arr
    """
    square_arr = arr**2
    summed = np.cumsum(square_arr) # Get our cumulative sums
    target_arr = np.zeros(arr.size - window_size)
    # Could theoretically run into problems if our array gets super large but
    # this should be good enough
    for i, _ in enumerate(target_arr):
        target_arr[i] = math.sqrt(summed[i+window_size] - summed[i])
    return target_arr

def get_repetitive(accel):
    """
    Gets repetitve sections of data from a
    """
    window_size = 500 # 5 seconds
    max_freqs = []
    rms = get_rms(accel, window_size)
    for i in range(len(accel)-window_size):
        accel_fft = scipy.fft.fft(accel[i:i+window_size])
        # This rms calculated to be linear in array size
        max_freq = max(np.abs(accel_fft)) # Ignore constant
        max_freqs.append(max_freq/rms[i])

    def get_sections(data, threshold=0.90):
        """
        Return list of sections where the values is above a threshold
        """
        going = False
        sections = []
        for i, val in enumerate(data):
            if abs(val) > threshold and not going:
                going = True
                start = i
            elif abs(val) <= threshold and going:
                going = False
                end = i
                sections.append((start,end,))
        if going:
            sections.append((start, len(data)))
        return sections

    def max_section(sections):
        """
        Get the largest section out of a list of sections
        """
        max = 0
        val = None
        for section in sections:
            start, end = section
            size = end-start
            if size > max:
                size = max
                val = section
        return val

    sections = get_sections(max_freqs/max(max_freqs))
    start, end = max_section(sections)
    exercise_section = accel[start:end+window_size]

    # Now find the fundamental frequency of the repetitive section
    return exercise_section, start, end

def cross_correlate(input, reference):
    """"
    Finds the sliding window cross correlatio between the input and
    reference
    """
    output = np.zeros(input.size - reference.size)
    for i in range(output.size):
        window = input[i:i+reference.size]
        corr = np.dot(window, reference)
        corr /=  np.sqrt(np.dot(reference, reference)) # Scale for reference
        corr /= np.sqrt(np.dot(window, window)) # Scale for size of function
        output[i] = corr
    return output

def check(file, reference):
    """
    Compute cross correlation for a single file with given reference
    """
    # Read the data in from the raw file
    try:
        accel, gyro = import_data(file)
    except DataError as e:
        print(e)
        return None, None
    preprocess_data(accel, gyro)
    accel = linear_interpolate(accel)
    gyro = linear_interpolate(gyro)

    # Read the data from our reference
    reference_accel = pd.read_csv(reference, index_col=0)
    cross_corr = cross_correlate(accel.x, reference_accel.x)

    # Find peaks in cross_corr
    peaks, _ = scipy.signal.find_peaks(cross_corr, distance=100, prominence=0.2)
    #print(accel)
    accel_list =[]
    for idx,val in enumerate(peaks):
        if (idx != 0):
            split_df = accel[peaks[idx-1]:peaks[idx]]
            split_df = split_df.reset_index(level=0)
            split_df['time'] = split_df['time'].astype(str)
            split_df['time'] = split_df['time'].str[13:].astype(float)
            start_time = split_df['time'][0]
            split_df['time'] = split_df['time'].subtract(start_time)
            if (len(split_df.index) > 300):
                split_df = split_df[0:300]
            accel_list.append(split_df)
    split_df = accel[peaks[-1]:min(peaks[-1]+250,len(accel))]
    split_df = split_df.reset_index(level=0)
    split_df['time'] = split_df['time'].astype(str)
    split_df['time'] = split_df['time'].str[13:].astype(float)
    start_time = split_df['time'][0]
    split_df['time'] = split_df['time'].subtract(start_time)
    accel_list.append(split_df)
    form_classes = analyze_test_data(accel_list)  

    return cross_corr, peaks, form_classes


def divide(filename : str, reference : str):
    print("Dividing")
    cross_corr, peaks, form_classes = check(filename, reference)

    return peaks, form_classes

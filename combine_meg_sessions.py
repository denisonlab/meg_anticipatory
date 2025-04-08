#!/usr/bin/env python3
"""
Script to combine and average MEG session data across all subjects.
Run this after all individual sessions have been processed.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import sys

# results dir
results_dir = "/projectnb/rdenlab/Users/Lizzy/results"
#  output directory for combined results
combined_dir = os.path.join(results_dir, "combined_results")
os.makedirs(combined_dir, exist_ok=True)

print(f"Reading data from: {results_dir}")
print(f"Writing combined results to: {combined_dir}")

# Define session IDs
session_ids = [
    "R0817_20181120", "R0817_20190625",
    "R1187_20181119", "R1187_20190703",
    "R0959_20181128", "R0959_20190703", 
    "R1373_20181128", "R1373_20190708",
    "R0983_20190722", "R0983_20190723",
    "R1507_20190621", "R1507_20190627",
    "R0898_20190723", "R0898_20190724", 
    "R1452_20181119", "R1452_20190711",
    "R1547_20190729", "R1547_20190730",
    "R1103_20181121", "R1103_20190710"
]

# Create log file
log_file = os.path.join(combined_dir, "combine_sessions_log.txt")
with open(log_file, 'w') as f:
    f.write(f"Started combining sessions at {datetime.now()}\n")
    f.write(f"Found {len(session_ids)} sessions to process\n")

# ----------------- COMBINING POWER SPECTRUM DATA -------------------
print("Combining power spectrum data...")

# lists to store all sessions' data
t1_powers_all_sessions = []
t2_powers_all_sessions = []
frequencies = None

# load power data for each session
valid_sessions_power = 0

for session_id in session_ids:
    t1_power_file = os.path.join(results_dir, f"{session_id}_t1_power.npy") # should have been obtained from each individual session
    t2_power_file = os.path.join(results_dir, f"{session_id}_t2_power.npy")
    # try /except block for data
    if os.path.exists(t1_power_file) and os.path.exists(t2_power_file):
        try:
            # load
            t1_data = np.load(t1_power_file, allow_pickle=True).item()
            t2_data = np.load(t2_power_file, allow_pickle=True).item()
            
            # checking if data has expected structure
            if 'average_power' in t1_data and 'average_power' in t2_data:
                # Store the frequencies from the first valid session
                if frequencies is None:
                    frequencies = t1_data['frequencies']
                
                # add average power 
                t1_powers_all_sessions.append(t1_data['average_power'])
                t2_powers_all_sessions.append(t2_data['average_power'])
                
                valid_sessions_power += 1
                print(f"  Added power data from session {session_id}")
                
            else:
                print(f"  Warning: missing average_power in data for session {session_id}")
                
        except Exception as e:
            print(f"  Error loading power data for session {session_id}: {e}")
    else:
        print(f"  Could not find power data files for session {session_id}")

# Check if we have enough data
if valid_sessions_power < 1:
    print("Error: No valid power spectrum data found. Exiting.")
    with open(log_file, 'a') as f:
        f.write("Error: No valid power spectrum data found.\n")
    sys.exit(1)

with open(log_file, 'a') as f:
    f.write(f"Found power data for {valid_sessions_power} sessions\n")

# convert lists to numpy arr
t1_powers_array = np.array(t1_powers_all_sessions)
t2_powers_array = np.array(t2_powers_all_sessions)

# get average across all sessions
grand_avg_t1_power = np.mean(t1_powers_array, axis=0)
grand_avg_t2_power = np.mean(t2_powers_array, axis=0)

# get standard error across sessions
sem_t1_power = np.std(t1_powers_array, axis=0) / np.sqrt(valid_sessions_power)
sem_t2_power = np.std(t2_powers_array, axis=0) / np.sqrt(valid_sessions_power)

print(f"Calculated grand average power spectra across {valid_sessions_power} sessions")

# Create power spectrum plot
plt.figure(figsize=(12, 8))

#  plot up to 10 Hz: can adjust depending on the goal
mask = frequencies <= 10

# Plot with error bands
plt.semilogy(frequencies[mask], grand_avg_t1_power[mask], 'b-', linewidth=2, label='T1')
plt.fill_between(frequencies[mask], 
                 grand_avg_t1_power[mask] - sem_t1_power[mask],
                 grand_avg_t1_power[mask] + sem_t1_power[mask], 
                 color='b', alpha=0.3)

plt.semilogy(frequencies[mask], grand_avg_t2_power[mask], 'r-', linewidth=2, label='T2')
plt.fill_between(frequencies[mask], 
                 grand_avg_t2_power[mask] - sem_t2_power[mask],
                 grand_avg_t2_power[mask] + sem_t2_power[mask], 
                 color='r', alpha=0.3)

plt.title('Grand Average Power Spectrum (Log Scale)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (log scale)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(combined_dir, "grand_average_power_spectrum.png"), dpi=300)
plt.close()

# Save combined data as dict
combined_power_results = {
    'frequencies': frequencies,
    'grand_avg_t1_power': grand_avg_t1_power,
    'grand_avg_t2_power': grand_avg_t2_power,
    'sem_t1_power': sem_t1_power,
    'sem_t2_power': sem_t2_power,
    'n_sessions': valid_sessions_power,
    'included_sessions': [session_ids[i] for i in range(len(session_ids)) 
                         if i < len(t1_powers_all_sessions)]
}
np.save(os.path.join(combined_dir, "combined_power_spectrum_results.npy"), combined_power_results)

# -----------------  TIME-FREQUENCY DATA -------------------
print("\nCombining time-frequency data:")

#  lists to store all sessions' data
t1_tfrs_all_sessions = []
t2_tfrs_all_sessions = []
freqs = None
times = None

# load TFR data for each session
valid_sessions_tfr = 0

for session_id in session_ids:
    t1_tfr_file = os.path.join(results_dir, f"{session_id}_t1_tfr.npy")
    t2_tfr_file = os.path.join(results_dir, f"{session_id}_t2_tfr.npy")
    
    if os.path.exists(t1_tfr_file) and os.path.exists(t2_tfr_file):
        try:
            # Load data
            t1_data = np.load(t1_tfr_file, allow_pickle=True).item()
            t2_data = np.load(t2_tfr_file, allow_pickle=True).item()
            
            # checking data structure
            if 'average_tfr' in t1_data and 'average_tfr' in t2_data:
                # Store the frequencies and times from the first valid session
                if freqs is None:
                    freqs = t1_data['freqs']
                    times = t1_data['times']
                
                # Append average TFR
                t1_tfrs_all_sessions.append(t1_data['average_tfr'])
                t2_tfrs_all_sessions.append(t2_data['average_tfr'])
                
                valid_sessions_tfr += 1
                print(f"  Added TFR data from session {session_id}")
                
            else:
                print(f"  Warning: Missing average_tfr in data for session {session_id}")
                
        except Exception as e:
            print(f"  Error loading TFR data for session {session_id}: {e}")
    else:
        print(f"  Could not find TFR data files for session {session_id}")

# Check if we have enough data
if valid_sessions_tfr < 1:
    print("Error: No valid time-frequency data found. Skipping TFR analysis.")
    with open(log_file, 'a') as f:
        f.write("Error: No valid time-frequency data found.\n")
else:
    with open(log_file, 'a') as f:
        f.write(f"Found TFR data for {valid_sessions_tfr} sessions\n")
    
    # Convert lists to numpy arrays
    t1_tfrs_array = np.array(t1_tfrs_all_sessions)
    t2_tfrs_array = np.array(t2_tfrs_all_sessions)
    
    # Calculate grand average across sessions
    grand_avg_t1_tfr = np.mean(t1_tfrs_array, axis=0)
    grand_avg_t2_tfr = np.mean(t2_tfrs_array, axis=0)
    
    # Calculate T1-T2 difference
    diff_tfr = grand_avg_t1_tfr - grand_avg_t2_tfr
    
    print(f"Calculated grand average time-frequency across {valid_sessions_tfr} sessions")
    
    # apply baseline correction
    # assuming time series length is at least 500ms before the event of interest
    has_baseline_period = len(times) > 500
    
    if has_baseline_period:
        print("Applying baseline correction using the first 500ms of data")
        
        # Define baseline period (first 500ms)
        baseline_start_idx = 0
        baseline_end_idx = 500  # Assuming 1000Hz sampling rate for 500ms
        
        # Make sure the indices are valid
        if baseline_end_idx > len(times):
            baseline_end_idx = len(times) // 4  # Use first quarter as a fallback
            print(f"Adjusted baseline period to first {baseline_end_idx} time points")
        
        # Calculate baseline (mean power during baseline period)
        t1_baseline = np.mean(grand_avg_t1_tfr[:, baseline_start_idx:baseline_end_idx], axis=1, keepdims=True)
        t2_baseline = np.mean(grand_avg_t2_tfr[:, baseline_start_idx:baseline_end_idx], axis=1, keepdims=True)
        
        # Apply baseline correction (subtract baseline power)
        grand_avg_t1_tfr_corrected = grand_avg_t1_tfr - t1_baseline
        grand_avg_t2_tfr_corrected = grand_avg_t2_tfr - t2_baseline
        
        # Calculate difference of baseline-corrected data
        diff_tfr_corrected = grand_avg_t1_tfr_corrected - grand_avg_t2_tfr_corrected
    else:
        print("Warning: Cannot apply baseline correction - time series too short")
        has_baseline_period = False
    
    # Plot T1 time-frequency without baseline correction
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, freqs, grand_avg_t1_tfr, shading='gouraud', cmap='RdBu_r')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Grand Average T1 Time-Frequency (Without Baseline Correction)')
    plt.colorbar(label='Power')
    plt.savefig(os.path.join(combined_dir, "grand_average_t1_tfr_raw.png"), dpi=300)
    plt.close()
    
    # Plot T2 time-frequency without baseline correction
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, freqs, grand_avg_t2_tfr, shading='gouraud', cmap='RdBu_r')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Grand Average T2 Time-Frequency (Without Baseline Correction)')
    plt.colorbar(label='Power')
    plt.savefig(os.path.join(combined_dir, "grand_average_t2_tfr_raw.png"), dpi=300)
    plt.close()
    
    # Plot T1-T2 difference (without baseline correction)
    vmax = np.max(np.abs(diff_tfr))
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, freqs, diff_tfr, shading='gouraud', cmap='RdBu_r', 
                  vmin=-vmax, vmax=vmax)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Grand Average T1-T2 Difference (Without Baseline Correction)')
    plt.colorbar(label='Power Difference')
    plt.savefig(os.path.join(combined_dir, "grand_average_tfr_difference.png"), dpi=300)
    plt.close()
    
    # Plot baseline-corrected results
    if has_baseline_period:
        # Plot T1 time-frequency with baseline correction
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(times, freqs, grand_avg_t1_tfr_corrected, shading='gouraud', cmap='RdBu_r')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title('Grand Average T1 Time-Frequency (With Baseline Correction)')
        plt.colorbar(label='Power (baseline subtracted)')
        plt.savefig(os.path.join(combined_dir, "grand_average_t1_tfr_corrected.png"), dpi=300)
        plt.close()
        
        # Plot T2 time-frequency with baseline correction
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(times, freqs, grand_avg_t2_tfr_corrected, shading='gouraud', cmap='RdBu_r')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title('Grand Average T2 Time-Frequency (With Baseline Correction)')
        plt.colorbar(label='Power (baseline subtracted)')
        plt.savefig(os.path.join(combined_dir, "grand_average_t2_tfr_corrected.png"), dpi=300)
        plt.close()
        
        # Plot T1-T2 difference with baseline correction
        vmax_corrected = np.max(np.abs(diff_tfr_corrected))
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(times, freqs, diff_tfr_corrected, shading='gouraud', cmap='RdBu_r', 
                      vmin=-vmax_corrected, vmax=vmax_corrected)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title('Grand Average T1-T2 Difference (With Baseline Correction)')
        plt.colorbar(label='Power Difference (baseline corrected)')
        plt.savefig(os.path.join(combined_dir, "grand_average_tfr_difference_corrected.png"), dpi=300)
        plt.close()
    
    # Save combined data
    combined_tfr_results = {
        'freqs': freqs,
        'times': times,
        'grand_avg_t1_tfr': grand_avg_t1_tfr,
        'grand_avg_t2_tfr': grand_avg_t2_tfr,
        'difference_tfr': diff_tfr,
        'n_sessions': valid_sessions_tfr,
        'included_sessions': [session_ids[i] for i in range(len(session_ids)) 
                             if i < len(t1_tfrs_all_sessions)]
    }
    
    # Add baseline-corrected data if available
    if has_baseline_period:
        combined_tfr_results.update({
            'grand_avg_t1_tfr_corrected': grand_avg_t1_tfr_corrected,
            'grand_avg_t2_tfr_corrected': grand_avg_t2_tfr_corrected,
            'difference_tfr_corrected': diff_tfr_corrected,
            'has_baseline_correction': True,
            'baseline_period': [baseline_start_idx, baseline_end_idx]
        })
    else:
        combined_tfr_results.update({
            'has_baseline_correction': False
        })
    
    np.save(os.path.join(combined_dir, "combined_tfr_results.npy"), combined_tfr_results)
# -------------- COMBINE ANTICIPATORY PERIOD DATA -----------------
print("\nCombining raw anticipatory period data")

# lists
t1_anticipatory_data = []
t2_anticipatory_data = []
valid_sessions_raw = 0
nan_stats = {}  # To track NaN statistics

# get anticipatory period 
for session_id in session_ids:
    t1_file = os.path.join(results_dir, session_id, "anticipatory_t1_top50_channels.npy")
    t2_file = os.path.join(results_dir, session_id, "anticipatory_t2_top50_channels.npy")
    
    if os.path.exists(t1_file) and os.path.exists(t2_file):
        try:
            # Load data
            t1_data = np.load(t1_file)
            t2_data = np.load(t2_file)
            
            # NaN stats
            t1_nan_count = np.isnan(t1_data).sum()
            t2_nan_count = np.isnan(t2_data).sum()
            t1_total = t1_data.size
            t2_total = t2_data.size
            t1_nan_percent = (t1_nan_count / t1_total) * 100
            t2_nan_percent = (t2_nan_count / t2_total) * 100
            
            # store stats
            nan_stats[session_id] = {
                'T1_nan_count': t1_nan_count,
                'T1_total_points': t1_total,
                'T1_nan_percent': t1_nan_percent,
                'T2_nan_count': t2_nan_count,
                'T2_total_points': t2_total,
                'T2_nan_percent': t2_nan_percent
            }
            
            # Log NaN statistics
            print(f"  Session {session_id} NaN stats:")
            print(f"    T1: {t1_nan_count}/{t1_total} ({t1_nan_percent:.2f}%) NaN values")
            print(f"    T2: {t2_nan_count}/{t2_total} ({t2_nan_percent:.2f}%) NaN values")
            
            # Check if data is too corrupted (e.g., >50% NaN)
            if t1_nan_percent > 50 or t2_nan_percent > 50:
                print(f"  WARNING: Session {session_id} has >50% NaN values. Consider excluding.")
            
            # For each trial, compute channel average (to get one time series per trial)
            # Use more robust approach for handling NaNs
            
            # Count NaN trials before averaging
            t1_nan_trials = 0
            t2_nan_trials = 0
            
            # Process T1 trials
            t1_avg_trials = []
            for i in range(t1_data.shape[2]):
                trial = t1_data[:, :, i]
                nan_percentage = np.isnan(trial).sum() / trial.size * 100
                
                if nan_percentage > 50:
                    t1_nan_trials += 1
                    continue  # Skip trials with >50% NaNs
                
                # Average across channels, handling NaNs
                trial_avg = np.nanmean(trial, axis=1)
                
                # Check if result has too many NaNs after averaging
                if np.isnan(trial_avg).sum() > trial_avg.size * 0.1:  # >10% NaNs
                    t1_nan_trials += 1
                    continue
                
                t1_avg_trials.append(trial_avg)
            
            # Process T2 trials
            t2_avg_trials = []
            for i in range(t2_data.shape[2]):
                trial = t2_data[:, :, i]
                nan_percentage = np.isnan(trial).sum() / trial.size * 100
                
                if nan_percentage > 50:
                    t2_nan_trials += 1
                    continue  # Skip trials with >50% NaNs
                
                # Average across channels, handling NaNs
                trial_avg = np.nanmean(trial, axis=1)
                
                # Check if result has too many NaNs after averaging
                if np.isnan(trial_avg).sum() > trial_avg.size * 0.1:  # >10% NaNs
                    t2_nan_trials += 1
                    continue
                
                t2_avg_trials.append(trial_avg)
            
            # Log trial statistics
            print(f"    T1: {len(t1_avg_trials)} valid trials, {t1_nan_trials} excluded trials")
            print(f"    T2: {len(t2_avg_trials)} valid trials, {t2_nan_trials} excluded trials")
            
            # Only include session if there are enough valid trials
            if len(t1_avg_trials) < 5 or len(t2_avg_trials) < 5:
                print(f"  WARNING: Session {session_id} has too few valid trials (<5). Excluding.")
                continue
            
            # Convert to arrays
            t1_avg_array = np.array(t1_avg_trials)
            t2_avg_array = np.array(t2_avg_trials)
            
            # Append to lists
            t1_anticipatory_data.append(t1_avg_array)
            t2_anticipatory_data.append(t2_avg_array)
            
            valid_sessions_raw += 1
            print(f"  Successfully added raw data from session {session_id}")
            
        except Exception as e:
            print(f"  Error loading raw data for session {session_id}: {e}")
    else:
        print(f"  Could not find raw data files for session {session_id}")

# Save NaN statistics to a file
with open(os.path.join(combined_dir, "nan_statistics.txt"), 'w') as f:
    f.write("NaN Statistics by Session\n")
    f.write("=======================\n\n")
    for session_id, stats in nan_stats.items():
        f.write(f"Session: {session_id}\n")
        f.write(f"  T1: {stats['T1_nan_count']}/{stats['T1_total_points']} ({stats['T1_nan_percent']:.2f}%) NaN values\n")
        f.write(f"  T2: {stats['T2_nan_count']}/{stats['T2_total_points']} ({stats['T2_nan_percent']:.2f}%) NaN values\n")
        f.write("\n")

if valid_sessions_raw < 1:
    print("Error: No valid raw anticipatory data found. Skipping raw data analysis.")
    with open(log_file, 'a') as f:
        f.write("Error: No valid raw anticipatory data found.\n")
else:
    with open(log_file, 'a') as f:
        f.write(f"Found raw anticipatory data for {valid_sessions_raw} sessions\n")
    
    # Since we've now ensured our data doesn't have problematic NaN values,
    # we can proceed with the averaging
    
    # Compute average time series across trials for each session
    t1_session_avgs = []
    t2_session_avgs = []
    
    for i, session_trials in enumerate(t1_anticipatory_data):
        # Average across all trials for this session
        session_avg = np.nanmean(session_trials, axis=0)
        t1_session_avgs.append(session_avg)
    
    for i, session_trials in enumerate(t2_anticipatory_data):
        # Average across all trials for this session
        session_avg = np.nanmean(session_trials, axis=0)
        t2_session_avgs.append(session_avg)
    
    # Convert to numpy arrays
    t1_session_avgs = np.array(t1_session_avgs)
    t2_session_avgs = np.array(t2_session_avgs)
    
    # Check for remaining NaNs in session averages
    if np.isnan(t1_session_avgs).any() or np.isnan(t2_session_avgs).any():
        print("WARNING: NaNs still present in session averages. Using nanmean for final average.")
        
    # Compute grand average across all sessions
    grand_avg_t1_ts = np.nanmean(t1_session_avgs, axis=0)
    grand_avg_t2_ts = np.nanmean(t2_session_avgs, axis=0)
    
    # Create time vector (1000 Hz sampling rate)
    time_vec = np.arange(len(grand_avg_t1_ts)) / 1000.0
    
    # Plot time series
    plt.figure(figsize=(12, 6))
    plt.plot(time_vec, grand_avg_t1_ts, 'b-', linewidth=2, label='T1')
    plt.plot(time_vec, grand_avg_t2_ts, 'r-', linewidth=2, label='T2')
    plt.title('Grand Average Anticipatory Time Series')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(combined_dir, "grand_average_time_series.png"), dpi=300)
    plt.close()
    
    # Save results
    combined_ts_results = {
        'time': time_vec,
        'grand_avg_t1_ts': grand_avg_t1_ts,
        'grand_avg_t2_ts': grand_avg_t2_ts,
        'n_sessions': valid_sessions_raw,
        'nan_statistics': nan_stats
    }
    np.save(os.path.join(combined_dir, "combined_time_series_results.npy"), combined_ts_results)

# Summarize results
total_results = {
    'n_sessions_total': len(session_ids),
    'n_sessions_power': valid_sessions_power,
    'n_sessions_tfr': valid_sessions_tfr,
    'n_sessions_raw': valid_sessions_raw,
    'analysis_date': str(datetime.now())
}

with open(log_file, 'a') as f:
    f.write("\nSummary of combined analysis:\n")
    f.write(f"Total sessions: {len(session_ids)}\n")
    f.write(f"Sessions with valid power data: {valid_sessions_power}\n")
    f.write(f"Sessions with valid TFR data: {valid_sessions_tfr}\n")
    f.write(f"Sessions with valid raw data: {valid_sessions_raw}\n")
    f.write(f"Combined analysis completed at {datetime.now()}\n")

print("\nSummary:")
print(f"Total sessions: {len(session_ids)}")
print(f"Sessions with valid power data: {valid_sessions_power}")
print(f"Sessions with valid TFR data: {valid_sessions_tfr}")
print(f"Sessions with valid raw data: {valid_sessions_raw}")
print(f"Combined analysis completed at {datetime.now()}")
print(f"Results saved to: {combined_dir}")
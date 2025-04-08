#!/usr/bin/env python3
# all_sessions.py - script to process all 20 MEG sessions

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch processing
import matplotlib.pyplot as plt
import glob
import sys
import traceback
from datetime import datetime

# Define results directory
results_dir = "/projectnb/rdenlab/Users/Lizzy/results"
os.makedirs(results_dir, exist_ok=True)
print(f"Created/verified results directory at: {results_dir}")

# Define session info
dir_dict = {
    'sessionDir': [
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
    ],
    'bietfpDir': [
        "R0817_TA2_11.20.18", "R0817_TA2_6.25.19",
        "R1187_TA2_11.19.18", "R1187_TA2_7.3.19",
        "R0959_TA2_11.28.18", "R0959_TA2_7.3.19",
        "R1373_TA2_11.28.18", "R1373_TA2_7.8.19",
        "R0983_TA2_7.22.19", "R0983_TA2_7.23.19",
        "R1507_TA2_6.21.19", "R1507_TA2_6.27.19",
        "R0898_TA2_7.23.19", "R0898_TA2_7.24.19",
        "R1452_TA2_11.19.18", "R1452_TA2_7.11.19",
        "R1547_TA2_7.29.19", "R1547_TA2_7.30.19",
        "R1103_TA2_11.21.18", "R1103_TA2_7.10.19"
    ]
}


assert len(dir_dict['sessionDir']) == 20, f"Expected 20 sessions, found {len(dir_dict['sessionDir'])}"
assert len(dir_dict['bietfpDir']) == 20, f"Expected 20 BIETFP dirs, found {len(dir_dict['bietfpDir'])}"

# check if required modules are available
def check_imports():
    missing_modules = []
    
    try:
        from scipy import signal
        import scipy.io
    except ImportError:
        missing_modules.append("scipy")
        
    try:
        from mne.time_frequency import tfr_array_morlet
    except ImportError:
        missing_modules.append("mne")
        
    try:
        from LoadData import Data
    except ImportError:
        missing_modules.append("LoadData")

    if missing_modules:
        print(f"ERROR: Missing required modules: {', '.join(missing_modules)}")
        print("Please ensure all required modules are installed:")
        print("  - scipy (for signal processing)")
        print("  - mne (for time-frequency analysis)")
        print("  - LoadData (your custom data loading module)")
        print("\nSuggested fixes:")
        print("1. Activate appropriate conda environment: conda activate your_env_name")
        print("2. Install missing packages: pip install mne scipy")
        print("3. Ensure LoadData.py is in the same directory or in PYTHONPATH")
        sys.exit(1)
    
    return True

# Function to process a single session, adapted from meg_analysis)single_session.ipynb
def process_session(sessionDir, bietfpDir):
    print(f"Processing session: {sessionDir}")
    
    # make session directory in results folder
    session_output_dir = os.path.join(results_dir, sessionDir)
    os.makedirs(session_output_dir, exist_ok=True)
    
    log_file = os.path.join(session_output_dir, "processing_log.txt")
    with open(log_file, 'a') as f:
        f.write(f"Started processing at {datetime.now()}\n")
    
    try:
        from scipy import signal
        import scipy.io
        from mne.time_frequency import tfr_array_morlet
        from LoadData import Data
        
        # Log import success
        with open(log_file, 'a') as f:
            f.write("Successfully imported all required modules\n")
        
        # Load data
        data_all = Data(sessionDir, bietfpDir)
        
        # Get ranked channels
        def get_ranked_ch(sessionDir, data_exp='TA2'):
            if data_exp == 'noise':
                Pk_avgProm_path = f"/projectnb/rdenlab/Data/TANoise/MEG/{sessionDir}/mat/channels_20Hz_ebi.mat"
                Pk_avgProm_path2 = f"/projectnb/rdenlab/Data/TANoise/MEG/{sessionDir}/channels_20Hz_ebi.mat"
            else:
                Pk_avgProm_path = f"/projectnb/rdenlab/Data/TA2/MEG/{sessionDir}/mat/Pk_avgProm.mat"
                Pk_avgProm_path2 = f"/projectnb/rdenlab/Data/TA2/MEG/{sessionDir}/Pk_avgProm.mat"
            
            try:
                Pk_avgProm_mat = scipy.io.loadmat(Pk_avgProm_path)
            except:
                Pk_avgProm_mat = scipy.io.loadmat(Pk_avgProm_path2)

            if data_exp == 'noise':
                rankedCh = Pk_avgProm_mat['channelsRanked'][0]
            else:
                rankedCh = Pk_avgProm_mat['Pk']['idxDirProm'][0][0][:, 0]

            return rankedCh
        
        # Get data with top 50 channels
        topN = 50
        data_exp = 'TA2'
        
        # Remove bad trials
        try:
            reject_trial_str1 = f"/projectnb/rdenlab/Data/TA2/MEG/{sessionDir}/mat/trials_rejected.mat"
            reject_trial_mat = scipy.io.loadmat(reject_trial_str1)
        except:
            reject_trial_str2 = f"/projectnb/rdenlab/Data/TA2/MEG/{sessionDir}/prep/trials_rejected.mat"
            reject_trial_mat = scipy.io.loadmat(reject_trial_str2)
        
        reject_trial_arr = reject_trial_mat['trials_rejected']
        # if index of rejected trials is valid
        if not (np.squeeze(reject_trial_arr) > np.shape(data_all.X)[2]).any():
            data_all.X[:, :, reject_trial_arr - 1] = None
        
        # Get ranked channels and select top N
        rankedCh = get_ranked_ch(sessionDir, data_exp)
        data_top50 = Data(sessionDir, bietfpDir)
        data_top50.X = data_all.X[:, rankedCh[:topN] - 1, :]
        
        # Log data shapes
        with open(log_file, 'a') as f:
            f.write(f"MEG data with all channels shape: {data_all.X.shape}\n")
            f.write(f"MEG data shape after top {topN} channel selection: {data_top50.X.shape}\n")
            f.write(f"Number of T1-cued trials: {len(data_all.precue_T1)}\n")
            f.write(f"Number of T2-cued trials: {len(data_all.precue_T2)}\n")
        
        # Extract data including pre-precue baseline (0-500ms) and anticipatory period (500-1550ms)
        anticipatory_t1_all_channels = data_all.X[0:1550, :, data_all.precue_T1]
        anticipatory_t2_all_channels = data_all.X[0:1550, :, data_all.precue_T2]
        
        anticipatory_t1_top50_channels = data_top50.X[0:1550, :, data_top50.precue_T1]
        anticipatory_t2_top50_channels = data_top50.X[0:1550, :, data_top50.precue_T2]
        
        # Save data
        np.save(f"{session_output_dir}/anticipatory_t1_top50_channels.npy", anticipatory_t1_top50_channels)
        np.save(f"{session_output_dir}/anticipatory_t2_top50_channels.npy", anticipatory_t2_top50_channels)
        np.save(f"{session_output_dir}/anticipatory_t1_all_channels.npy", anticipatory_t1_all_channels)
        np.save(f"{session_output_dir}/anticipatory_t2_all_channels.npy", anticipatory_t2_all_channels)
        
        with open(log_file, 'a') as f:
            f.write("Data extracted and saved successfully\n")
        
        # ------------- POWER SPECTRUM ANALYSIS -------------
        sfreq = 1000  # Sampling frequency (Hz)
        
        # Initialize lists to store power spectra for each trial
        t1_powers = []
        t2_powers = []
        
        # Calculate FFT for each T1 trial individually
        # scaling_factor = 1e15  # Scale the data

        with open(log_file, 'a') as f:
            f.write("Processing T1 trials for power spectrum...\n")
        
        for i in range(anticipatory_t1_top50_channels.shape[2]):
            # Average across channels for this trial to get a single time series
            trial_data = np.mean(anticipatory_t1_top50_channels[:, :, i], axis=1) 
            
            # Filter NaN values
            if np.isnan(trial_data).any():
                continue
            
            # scale data (optional)
            # trial_data = trial_data * scaling_factor
            
            # Get FFT and power
            fft_result = np.fft.rfft(trial_data)
            power = np.abs(fft_result)**2
            
            # Check if power values are valid
            if np.isnan(power).any():
                continue
            
            t1_powers.append(power)
        
        with open(log_file, 'a') as f:
            f.write(f"Valid T1 trials: {len(t1_powers)} out of {anticipatory_t1_top50_channels.shape[2]}\n")

        # Process T2 trials
        with open(log_file, 'a') as f:
            f.write("Processing T2 trials for power spectrum...\n")
        
        for i in range(anticipatory_t2_top50_channels.shape[2]):
            # Average across channels for this trial
            trial_data = np.mean(anticipatory_t2_top50_channels[:, :, i], axis=1)
            
            # Filter NaN values
            if np.isnan(trial_data).any():
                continue
            
            # Scale data
            # trial_data = trial_data * scaling_factor
            
            # Get FFT and power
            fft_result = np.fft.rfft(trial_data)
            power = np.abs(fft_result)**2
            
            # Check if power values are valid
            if np.isnan(power).any():
                continue
            
            t2_powers.append(power)

        with open(log_file, 'a') as f:
            f.write(f"Valid T2 trials: {len(t2_powers)} out of {anticipatory_t2_top50_channels.shape[2]}\n")

        # Proceed if we have valid trials
        if len(t1_powers) == 0 or len(t2_powers) == 0:
            with open(log_file, 'a') as f:
                f.write("Not enough valid trials to create power spectra\n")
            return False
        
        # Convert lists to numpy arrays
        t1_powers_array = np.array(t1_powers)
        t2_powers_array = np.array(t2_powers)
        
        # Average across trials
        avg_t1_power = np.mean(t1_powers_array, axis=0)
        avg_t2_power = np.mean(t2_powers_array, axis=0)
        
        # Calculate frequencies
        frequencies = np.fft.rfftfreq(anticipatory_t1_top50_channels.shape[0], 1/sfreq)
        
        # Create and save plot
        plt.figure(figsize=(12, 6))
        
        # Only plot up to 10 Hz
        mask = frequencies <= 10
        
        # Log scale plot
        plt.semilogy(frequencies[mask], avg_t1_power[mask], 'b-', linewidth=2, label='T1')
        plt.semilogy(frequencies[mask], avg_t2_power[mask], 'r-', linewidth=2, label='T2')
        plt.title(f'Power Spectrum (Log Scale) - Session {sessionDir}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (log scale)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/{sessionDir}_power_spectrum.png", dpi=300)
        plt.close()
        
        # Save results for later aggregation
        result = {
            'frequencies': frequencies,
            'average_power': avg_t1_power,
            'all_powers': t1_powers_array,
            # 'scaling_factor': scaling_factor
        }
        np.save(f"{results_dir}/{sessionDir}_t1_power.npy", result)
        
        result = {
            'frequencies': frequencies,
            'average_power': avg_t2_power,
            'all_powers': t2_powers_array,
            # 'scaling_factor': scaling_factor
        }
        np.save(f"{results_dir}/{sessionDir}_t2_power.npy", result)
        
        with open(log_file, 'a') as f:
            f.write("Power spectrum analysis completed and saved\n")
        
        # ------------- WAVELET ANALYSIS -------------
        # Define frequencies and number of cycles
        freqs = np.linspace(0.5, 10, 30)  # Frequencies from 0.5-10 Hz
        n_cycles = 7  # Number of cycles
        
        # Calculate buffer size needed for lowest frequency
        lowest_freq = min(freqs)  # 0.5 Hz
        min_buffer_duration = n_cycles / lowest_freq  # 7/0.5 = 14 seconds
        buffer_duration = 15  
        buffer_points = int(buffer_duration * sfreq)
        
        with open(log_file, 'a') as f:
            f.write(f"Starting wavelet analysis with frequencies: {freqs}\n")
            f.write(f"Using {n_cycles} cycles for all frequencies\n")
            f.write(f"Buffer: {buffer_duration} seconds ({buffer_points} points)\n")
        
        # Initialize arrays to store TFR results for each trial
        t1_tfrs = []
        t2_tfrs = []
        
        # Process T1 trials
        with open(log_file, 'a') as f:
            f.write("Processing T1 trials with wavelet analysis...\n")
        
        for i in range(anticipatory_t1_top50_channels.shape[2]):
            # Average across channels for this trial
            trial_data = np.mean(anticipatory_t1_top50_channels[:, :, i], axis=1)
            
            # Check for NaN values
            if np.isnan(trial_data).any():
                continue
                
            # Get original data length
            n_times = len(trial_data)
            
            # Check if data+buffer is long enough
            wavelet_length = int(n_cycles / lowest_freq * sfreq)
            min_length_needed = wavelet_length + 1
            
            if n_times + 2*buffer_points < min_length_needed:
                continue
                
            # Add buffer (zero padding)
            padded_data = np.pad(trial_data, pad_width=buffer_points, mode='constant', constant_values=0)
            
            # Reshape for MNE
            data_tfr = padded_data.reshape(1, 1, -1)
            
            try:
                # Apply Morlet wavelet transform
                tfr = tfr_array_morlet(data_tfr, sfreq=sfreq, freqs=freqs,
                                     n_cycles=n_cycles, output='power')
                
                # Extract the relevant portion (remove buffer)
                result = tfr[0, 0, :, buffer_points:buffer_points+n_times]
                
                # Check for valid result
                if np.isnan(result).any():
                    continue
                    
                t1_tfrs.append(result)
                
            except ValueError as e:
                with open(log_file, 'a') as f:
                    f.write(f"  Trial {i}: Error running wavelet transform: {e}\n")
                continue
                
        with open(log_file, 'a') as f:
            f.write(f"Valid T1 trials for wavelet analysis: {len(t1_tfrs)} out of {anticipatory_t1_top50_channels.shape[2]}\n")
        
        # Process T2 trials
        with open(log_file, 'a') as f:
            f.write("Processing T2 trials with wavelet analysis...\n")
            
        for i in range(anticipatory_t2_top50_channels.shape[2]):
            # Same process as for T1
            trial_data = np.mean(anticipatory_t2_top50_channels[:, :, i], axis=1)
            if np.isnan(trial_data).any():
                continue
            n_times = len(trial_data)
            wavelet_length = int(n_cycles / lowest_freq * sfreq)
            min_length_needed = wavelet_length + 1
            if n_times + 2*buffer_points < min_length_needed:
                continue
            padded_data = np.pad(trial_data, pad_width=buffer_points, mode='constant', constant_values=0)
            data_tfr = padded_data.reshape(1, 1, -1)
            try:
                tfr = tfr_array_morlet(data_tfr, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, output='power')
                result = tfr[0, 0, :, buffer_points:buffer_points+n_times]
                if np.isnan(result).any():
                    continue
                t2_tfrs.append(result)
            except ValueError:
                continue
                
        with open(log_file, 'a') as f:
            f.write(f"Valid T2 trials for wavelet analysis: {len(t2_tfrs)} out of {anticipatory_t2_top50_channels.shape[2]}\n")
        
        # Proceed if we have valid trials
        if len(t1_tfrs) == 0 or len(t2_tfrs) == 0:
            with open(log_file, 'a') as f:
                f.write("Not enough valid trials for time-frequency analysis\n")
            return False
            
        # Convert lists to numpy arrays
        t1_tfrs_array = np.array(t1_tfrs)
        t2_tfrs_array = np.array(t2_tfrs)
        
        # Average across trials
        avg_t1_tfr = np.mean(t1_tfrs_array, axis=0)
        avg_t2_tfr = np.mean(t2_tfrs_array, axis=0)
        
        # Create time vector
        n_times = anticipatory_t1_top50_channels.shape[0]
        times = np.arange(n_times) / sfreq
        
        # Determine if we can do baseline correction
        # Now we should always have baseline since we're extracting 0-1550ms
        
        # Define baseline period
        baseline_start_idx = 0
        baseline_end_idx = 500  # 500ms pre-precue period
        
        # Calculate baseline (mean power during pre-precue period)
        t1_baseline = np.mean(avg_t1_tfr[:, baseline_start_idx:baseline_end_idx], axis=1, keepdims=True)
        t2_baseline = np.mean(avg_t2_tfr[:, baseline_start_idx:baseline_end_idx], axis=1, keepdims=True)
        
        # Apply baseline correction (subtract baseline power)
        t1_tfr_corrected = avg_t1_tfr - t1_baseline
        t2_tfr_corrected = avg_t2_tfr - t2_baseline
        
        # Plot T1 time-frequency
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(times, freqs, avg_t1_tfr, shading='gouraud', cmap='RdBu_r')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title(f'T1 Time-Frequency - Session {sessionDir} (Without Baseline Correction)')
        plt.colorbar(label='Power')
        # Add vertical line at precue (t=0.5s)
        plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.7)
        plt.savefig(f"{results_dir}/{sessionDir}_t1_tfr_raw.png", dpi=300)
        plt.close()
        
        # Plot T2 time-frequency
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(times, freqs, avg_t2_tfr, shading='gouraud', cmap='RdBu_r')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title(f'T2 Time-Frequency - Session {sessionDir} (Without Baseline Correction)')
        plt.colorbar(label='Power')
        # Add vertical line at precue (t=0.5s)
        plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.7)
        plt.savefig(f"{results_dir}/{sessionDir}_t2_tfr_raw.png", dpi=300)
        plt.close()
        
        # Plot baseline-corrected results
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(times, freqs, t1_tfr_corrected, shading='gouraud', cmap='RdBu_r')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title(f'T1 Time-Frequency - Session {sessionDir} (With Baseline Correction)')
        plt.colorbar(label='Power (baseline subtracted)')
        # Add vertical line at precue (t=0.5s)
        plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.7)
        plt.savefig(f"{results_dir}/{sessionDir}_t1_tfr_corrected.png", dpi=300)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(times, freqs, t2_tfr_corrected, shading='gouraud', cmap='RdBu_r')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title(f'T2 Time-Frequency - Session {sessionDir} (With Baseline Correction)')
        plt.colorbar(label='Power (baseline subtracted)')
        # Add vertical line at precue (t=0.5s)
        plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.7)
        plt.savefig(f"{results_dir}/{sessionDir}_t2_tfr_corrected.png", dpi=300)
        plt.close()
        
        # Plot T1-T2 difference
        diff = t1_tfr_corrected - t2_tfr_corrected  # Now using baseline-corrected data
        vmax = np.max(np.abs(diff))
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(times, freqs, diff, shading='gouraud', cmap='RdBu_r', 
                      vmin=-vmax, vmax=vmax)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title(f'T1-T2 Difference (Baseline Corrected) - Session {sessionDir}')
        plt.colorbar(label='Power Difference')
        # Add vertical line at precue (t=0.5s)
        plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.7)
        plt.savefig(f"{results_dir}/{sessionDir}_tfr_difference.png", dpi=300)
        plt.close()
        
        # Save results for later aggregation
        result = {
            'freqs': freqs,
            'times': times,
            'average_tfr': avg_t1_tfr,
            'average_tfr_corrected': t1_tfr_corrected,
            'baseline': t1_baseline.squeeze(),
            'all_tfrs': t1_tfrs_array
        }
        np.save(f"{results_dir}/{sessionDir}_t1_tfr.npy", result)
        
        result = {
            'freqs': freqs,
            'times': times,
            'average_tfr': avg_t2_tfr,
            'average_tfr_corrected': t2_tfr_corrected,
            'baseline': t2_baseline.squeeze(),
            'all_tfrs': t2_tfrs_array
        }
        np.save(f"{results_dir}/{sessionDir}_t2_tfr.npy", result)
        
        with open(log_file, 'a') as f:
            f.write("Wavelet analysis completed and saved\n")
            f.write(f"Processing completed successfully at {datetime.now()}\n")
            
        return True
        
    except Exception as e:
        error_msg = f"Error processing session {sessionDir}: {str(e)}"
        print(error_msg)
        with open(log_file, 'a') as f:
            f.write(f"ERROR: {error_msg}\n")
            f.write(f"Traceback: {traceback.format_exc()}\n")
        return False

# Main execution
if __name__ == "__main__":
    print("Starting analysis of all MEG sessions...")
    
    # Check if required modules are available
    check_imports()
    
    # Get session information
    session_dirs = dir_dict['sessionDir']
    bietfp_dirs = dir_dict['bietfpDir']
    
    print(f"Found exactly {len(session_dirs)} sessions to process")
    print("Sessions to process:")
    for i, session in enumerate(session_dirs):
        print(f"  {i+1}. {session}")
    
    # Process each session
    successful = 0
    failed = 0
    
    for i, (sessionDir, bietfpDir) in enumerate(zip(session_dirs, bietfp_dirs)):
        print(f"\nProcessing session {i+1}/{len(session_dirs)}: {sessionDir}")
        success = process_session(sessionDir, bietfpDir)
        
        if success:
            print(f"✓ Successfully processed {sessionDir}")
            successful += 1
        else:
            print(f"✗ Failed to process {sessionDir}")
            failed += 1
    
    print("\nAnalysis summary:")
    print(f"- Total sessions: {len(session_dirs)}")
    print(f"- Successfully processed: {successful}")
    print(f"- Failed: {failed}")
    print("Analysis complete!")
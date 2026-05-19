This document provides a description of the data files and metadata provided in FatigueSet. For more information see the [FatigueSet website](https://www.esense.io/datasets/fatigueset) or the original publication:

**FatigueSet: A Multi-modal Dataset for Modeling Mental Fatigue and Fatigability**

Manasa Kalanadhabhatta, Chulhong Min, Alessandro Montanari and Fahim Kawsar
In 15th International Conference on Pervasive Computing Technologies for Healthcare (Pervasive Health), December 6–8, 2021

# Metadata and Surveys

The file metadata.csv in the root directory contains counterbalancing information for the order in which parrticipants completed low-, medium-, and high-intensity activity. For example, P01 performed low-intensity activity in session 01, medium-intenstiy activity in session 02, and high-intensity activity in session 03.

The file pre_task_survey.xlsx contains results from the Pre-task Survey completed by participants in the beginning of each session. This includes current sleepiness on the Stanford Sleepiness Scale and baseline vigor and affect on the Global Vigour and Affect Scale.

The file preliminary_questionnaire.xlsx contains results from the Preliminary Survey that participants complete once at the beginning of the study. It contains demographic information, personality on the BFI-10 scale, chronotype and sleep patterns using the Munich Chronotype Questionnaire (MCTQ), the impact of fatigue on daily functioning measurred on the Fatigue Severity Scale, and general fitness measured on the International Fitness Scale.

# Physiological Sensor Data
The sensor data collected from each of the four wearable devices used in the study are described below:

## Nokia Bell Labs eSense earable sensor

|file name|values|description|units/range|sampling rate|
|:---|:---|:---|:---|:---|
|ear_acc_left/ear_acc_right|timestamp,ax,ay,az|Acceleration along each axis|g {-2:+2}|100 Hz|
|ear_gyro_left/ear_gyro_right|timestamp,gx,gy,gz|Angular velocity along each axis|degrees/second {-500:+500}|100 Hz|
|ear_ppg_left/ear_ppg_right|timestamp,green,ir,red|Intensity of light from PPG sensor|no units|100 Hz|

## Muse S Headband

|file name|values|description|units/range|sampling rate|
|:---|:---|:---|:---|:---|
|forehead_acc|timestamp,ax,ay,az|Acceleration along each axis|g {-2:+2}|52 Hz|
|forehead_eeg_alpha_abs|timestamp,TP9,AF7,AF8,TP10|Absolute EEG alpha band power|Bels|10 Hz|
|forehead_eeg_beta_abs|timestamp,TP9,AF7,AF8,TP10|Absolute EEG beta band power|Bels|10 Hz|
|forehead_eeg_delta_abs|timestamp,TP9,AF7,AF8,TP10|Absolute EEG delta band power|Bels|10 Hz|
|forehead_eeg_gamma_abs|timestamp,TP9,AF7,AF8,TP10|Absolute EEG gamma band power|Bels|10 Hz|
|forehead_eeg_theta_abs|timestamp,TP9,AF7,AF8,TP10|Absolute EEG theta band power|Bels|10 Hz|
|forehead_eeg_raw|timestamp,TP9,AF7,AF8,TP10|Raw EEG waveform|uV {0.0:1682.815}|256 Hz|
|forehead_gyro|timestamp,gx,gy,gz|Angular velocity along each axis|degrees/second {-245:+245}|52 Hz|

|file name|values|description|units/range|sampling rate|
|:---|:---|:---|:---|:---|
|muse_blinks|timestamp,is_blink|Timestamps of blinks detected by Muse S|1=blink|10 Hz if true|
|muse_jaw_clenches|timestamp,is_clench|Timestamps of jaw clenches detected by Muse S|1=clench|10 Hz if true|
|muse_device_battery|timestamp,battery_level_muse,battery_voltage_muse,adc_voltage_muse,temperature_muse|Battery Level, Battery Voltage, ADC Voltage, Temperature|percentage, mV, {-1 N/A}, C|0.1 Hz|
|muse_device_fit|timestamp,TP9,AF7,AF8,TP10|TP9, AF7, AF8, TP10 fit|1=Good, 2=Medium, 4=Bad|10 Hz|
|muse_device_touch|timestamp,is_touching|Touching forehead|1=True, 0=False|10 Hz|

## Zephyr BioHarness 3.0 chest monitor

|file name|values|description|units/range|sampling rate|
|:---|:---|:---|:---|:---|
|chest_raw_acc|timestamp,vertical,lateral,sagittal|Raw 12-bit unfiltered accelerometer output. Centered at 2048, 1 g = 83 bits|bits {0-4095}, invalid value: 4095|100 Hz|
|chest_bb_interval|timestamp,duration|Timestamp of each breath event (unfiltered), duration from last breath event in milliseconds|ms|per detected breath|
|chest_physiology_summary|timestamp,hr,br,posture,hr_confidence,hrv,is_hr_unreliable,is_br_unreliable,is_hrv_unreliable|Heart rate, breathing rate, posture, heart rate confidence, heart rate variability, heart rate reliability, breath rate reliaility, HRV reliability|beats per minute {25:240}, breaths per minute {4:70}, degrees from vertical {-180:180}, percentage (above 20%=reliable), standard deviation of NN intervals for last 300 heartbeats in ms {0:65534, 65535=invalid}, 1=unreliable, 1=unreliable, 1=unreliable|1 Hz|
|chest_raw_breathing|timestamp,breathing_waveform|Raw uncalibrated 24-bit breathing sensor output|bits {1:16777215}, 0, 16777216=invalid|25 Hz|
|chest_raw_ecg|timestamp,ecg_waveform|12-bit filtered ECG waveform, 1 bit = 0.0067025 mV|bits {0:4095}, 4095=invalid|250 Hz|
|chest_rr_interval|timestamp,duration|Timestamp of each R event, duration from last R event in milliseconds|ms {0:32767
}|per detected R event|
|chest_sensor_summary|timestamp,acc_magnitude,acc_peak,acc_peak_vertical_angle,acc_peak_horizontal_angle,ecg_amp_uncalibrated,ecg_noise_uncalibrated|Vector magnitude of acceleration, peak acceleration magnitude over previous second, direction of peak magnitude from vertical, direction of peak magnitude from horizontal, uncalibrated amplitude of QRS complex, un-calibrated amplitude of noise signals measured between QRS complexes |VMU (measured in g) {0:16}, g {0:16}, degrees {0:180}, degrees {-180:180}, Volts {0:0.05}, Volts {0:0.05}|1 Hz|

|file name|values|description|units/range|sampling rate|
|:---|:---|:---|:---|:---|
|zephyr_activity_summary|timestamp,cumulative_impulse_load,walking_step_count,running_step_count,bound_count,jump_count,minor_impact_count,major_impact_count,avg_force_dev_rate,avg_step_impulse,avg_step_period,last_jump_flight_time|Cumulative impulse load over current session, cumulative count of walking steps, cumulative count of running steps, cumulative count of bounds, cumulative count of jumps, cumulative count of minor impacts – peak 
accelerometer magnitude between 3 and 7g, cumulative count of major impacts – peak 
accelerometer magnitude over 7g, Average force development rate, Average step impulse, Average time duration of previous 10 steps, Flight time of last detected jump |Newtons, Number {0:262143}, Number {0:262143}, Number {0:1023}, Number {0:1023}, Number {0:1023}, Number {0:1023}, Newtons per second {0:4095}, Newton seconds {0:1023}, seconds {0:1023}, seconds {0:255}|1 Hz|
|zephyr_device_status|timestamp,battery_voltage,battery_level,device_temperature,bluetooth_link_quality,bluetooth_rssi,bluetooth_tx_power,worn_confidence,is_button_press,is_not_fitted_to_garment|Battery voltage, battery level, device temperature, Bluetooth link quality, Bluetooth RSSI, Bluetooth Tx Power, Device worn confidence, Button press, Garment fit |V{approx 3.6:4.2}, percentage, C, {0=poor quality, 254=high quality, 255=invalid}, dB {-127:127, -128=invalid}, dBm {-30:20, -128=invalid}, {00=full confidence device is worn, 01=high confidence, 10=low confidence, 11=no confidence}, {1=button pressed}, {1=not fitted to garment}|1 Hz|

## Empatica E4 wristband

|file name|values|description|units/range|sampling rate|
|:---|:---|:---|:---|:---|
|wrist_acc|timestamp,ax,ay,az|Acceleration along each axis|g {-2:+2}|32 Hz|
|wrist_bvp|timestamp,bvp|Blood volume pulse from PPG sensor|no unit provided because BVP derived from combination of two different measures of the amount of reflected light|64Hz|
|wrist_eda|timestamp,eda|Electrodermal activity|microsiemens|4 Hz|
|wrist_hr|timestamp,hr|Average heart rate over last 10 seconds|1 Hz|
|wrist_ibi|timestamp,duration|Time between individuals heart beats extracted from the BVP signal|ms|per detected heart beat|
|wrist_skin_temperature|timestamp,temp|Data from temperature sensor|C|4 Hz|

# Video Extraction Tool



# Simple Video Extraction Method:
`python video_concatenation.py subj_id start_date start_time end_date end_time video_path vtc_path output_path ten_min_flag`

* Subj_id: String name of patient, to be used in the output file
* Dates: yyyy-mm-dd in PST
* Times: hh:mm:ss in PST
* Video_path: absolute path to directory containing (anonymized) videos
* VTC_path: absolute path to input VTC
* Output_path: abs/rel path to the output directory
* ten_min_flag: 0 for standard output. 1 to partition output into 10 minute chunks.

There is a variable that allows for the chunking to be adjustable. Look for NUM_VIDEOS_TO_CONCATENATE in video_concatenation.py to adjust this value. 

Concatenates video using ffmpeg wrt input date & time range. Works on local. Server usage WIP.

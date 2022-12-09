"""
Concatenates videos within a certain time frame, referencing a VTC file for selection.

Usage: python video_concatenation.py subj_id start_date start_time end_date end_time video_path vtc_path output_path

Arguments
----------
subj_id: string
    Name of patient, used in creating output file. Ex: EC156

start_date: yyyy-mm-dd
    Date associated with the beginning timestamp

start_time: hh:mm:ss
    Time associated with the beginning timestamp

end_date: yyyy-mm-dd
    Date associated with the beginning timestamp

end_time: hh:mm:ss
    Time associated with the ending timestamp

video_path: path/to/directory
    Directory containing video files

vtc_path: path/to/file.vtc
    Location of VTC file

output_path: path/to/directory
    Directory in which the output will be generated

ten_min_flag: int
    0 or 1, 0 for no ten minute chunking of outputs, 1 for chunking of outputs

Notes
----------
Video and VTC Path, for reading operations, should be absolute.

The output path, where the concatenated video is deposited, can be relative.

"""

import sys
import os
import pandas
from datetime import datetime
import subprocess
import pytz
from cl_iotools.xltek.vtcfile import VtcFile
from shutil import copyfile

NUM_VIDEOS_TO_CONCATENATE = 5
# This is for the optional chunking output. As each video is two minutes, running with
# NUM_VIDEOS_TO_CONCATENATE = 5 will produce 10 minute chunks.

def main(argv):
    subject_id = argv[0]
    start_date = argv[1]
    start_time = argv[2]
    end_date = argv[3]
    end_time = argv[4]
    video_path = argv[5]
    vtc_path = argv[6]
    output_path = argv[7]
    ten_min_flag = int(argv[8])
    start_datetime = string_to_datetime(start_date, start_time)
    end_datetime = string_to_datetime(end_date, end_time)
    extract_video(subject_id, start_datetime, end_datetime, video_path, vtc_path, output_path, ten_min_flag)
    subprocess.call(["cat", "skipped_files.txt"])



def extract_video(subject_id, start_datetime, end_datetime, video_path, vtc_path, out_path, ten_min_flag):
    """
    Concatenates videos within a certain time frame, referencing a VTC file for selection.

    Usage: python video_concatenation.py subj_id start_date start_time end_date end_time video_path vtc_path output_path

    Arguments
    ----------
    subj_id: string
        Name of patient, used in creating output file. Ex: EC156

    start_datetime: datetime object
        Datetime associated with the beginning timestamp


    end_datetime: datetime object
        Datetime associated with the beginning timestamp

    video_path: path/to/directory
        Directory containing video files

    vtc_path: path/to/file.vtc
        Location of VTC file

    output_path: path/to/directory
        Directory in which the output will be generated

    ten_min_flag: int
        0 or 1, 0 for no ten minute chunking of outputs, 1 for chunking of outputs

    Notes
    ----------
    Video and VTC Path, for reading operations, should be absolute.

    The output path, where the concatenated video is deposited, can be relative.

    """
    # create csv, obtain its path, create output dir
    vtc_to_csv([vtc_path])
    csv_path = vtc_path.split('.')[0] + ".csv"

    df = pandas.read_csv(csv_path, usecols=[0, 2, 4], parse_dates=True, infer_datetime_format=True, header=0)
    video_inds = datetime_to_videos(df, start_datetime, end_datetime)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    video_output = out_path + '/' + subject_id + "_" + str(start_datetime) + "-" + str(end_datetime)[11:] + ".avi"
    list_to_final(video_path, vtc_path, video_inds, video_output, ten_min_flag, subject_id)



def vtc_to_csv(filelist):
    """
    Generates CSV files with input VTC's.
    """
    for fn in filelist:
        with open(''.join(fn.split('.')[:-1]) + '.csv', 'w') as fout:
            vf = VtcFile(fn)
            fout.write('"Video File","Start time (Unix timestamp)","Start time (Human readable)","End time (Unix timestamp)","End time (Human readable)"\n')
            fout.flush()
            for rec in vf:
                starttimestamp = (rec.starttime - pytz.utc.localize(datetime.utcfromtimestamp(0))).total_seconds()
                endtimestamp = (rec.endtime - pytz.utc.localize(datetime.utcfromtimestamp(0))).total_seconds()
                fout.write('"%s",%f,"%s",%f,"%s"\n' % (rec.video_file_name.split('_')[-1],
                                                       starttimestamp,
                                                       rec.starttime,
                                                       endtimestamp,
                                                       rec.endtime))
                fout.flush()
        print('Done: %s' % fn)
    return 0


def list_to_final(video_path, vtc_path, inds, output_path, ten_min_flag, subj_id):
    """
    Creates the text file setup for ffmpeg video concatenation.

    Parameters
    ----------
    video_path: String
        Path to directory which stores the video files.

    vtc_path: String
        Path to VTC file which follows the same naming convention prefix as the videos.

    inds: Dataframe
        Selection of video file names which fit datetime range criteria.

    output_path: String
        Path to desired video output name.

    ten_min_flag: int
        0 or 1, 0 for no ten minute chunking of outputs, 1 for chunking of outputs

    subj_id: string
        Name of patient, used in creating output file. Ex: EC156

    Returns
    ---------
    None

    A concatenated file is generated at the output path. If ten_min_flag==1, then output will also consist
    of 10 minute output chunks.

    """
    prefix = vtc_path.split('.')[0].split('/')[-1]
    video_list = open(video_path + "videos.txt", "w")
    curr_file = 0
    count = 0
    chunks_list = []
    skipped_files = open("skipped_files.txt", "w")
    skipped_files.write("\n")
    skipped_files.write("The following videos were not found likely because the naming convention is off or the video is missing from the directory.\n\n")
    for i in inds:
        concat_video = video_path + prefix + "_" + i
        if os.path.isfile(concat_video):
            video_list.write("file '" + concat_video + "'\n")
            if ten_min_flag:
                if count % NUM_VIDEOS_TO_CONCATENATE == 0:
                    curr_file += 1
                    file_to_open = video_path + str(curr_file) + "_chunk_videos.txt"
                    chunks_list.append(file_to_open)
                    video_chunk_list = open(file_to_open, "w")
                video_chunk_list.write("file '" + concat_video + "'\n")
                count += 1
        else:
            print("----------ALERT----------")
            print("file " + i + " not found, skipping")
            print("full path " + concat_video)
            skipped_files.write("Missing File: " + concat_video + "\n")
    skipped_files.write("\n")
    skipped_files.close()
    video_list.close()
    if ten_min_flag:
        video_chunk_list.close()
        split_out = output_path.split('/')
        del split_out[-1]
        path_to_dir = '/'.join(split_out)
        init_avi_path, final_mp4_path = [], []
        # create chunked videos
        for chunk in chunks_list:
            start, end, hex = extract_indices(chunk)
            newout =  path_to_dir + '/' + subj_id + "_" + hex + '_' + start + '-' + end + '.avi'
            subprocess.call(["ffmpeg", "-f", "concat", "-safe", "0", "-i", chunk, "-c", "copy", newout])
            init_avi_path.append(newout)
            final_mp4_path.append(path_to_dir + '/' + subj_id + "_" + hex + '_' + start + '-' + end + '.mp4')
        # convert avi to mp4
        for init, final in zip(init_avi_path, final_mp4_path):
            subprocess.call(["ffmpeg", '-i', init, final])
            # subprocess.call(["ffmpeg",'-i',init_mp4_path,'-itsoffset', '0.50', '-i', init_mp4_path, '-map', '0:v','-map', '1:a', '-vcodec', 'copy','-acodec', 'copy', final_mp4_path])

        first_start, end_placeholder, hex_placeholder = extract_indices(chunks_list[0])
        skipped_out = path_to_dir + '/' + subj_id + "_" + hex + '_' + first_start + '-' + end + '.txt'
        copyfile("skipped_files.txt", skipped_out)
    #ffmpeg -f concat -safe 0 -i mylist.txt -c copy output
    else:
        subprocess.call(["ffmpeg", "-f", "concat", "-safe", "0", "-i",  video_path + "videos.txt", "-c", "copy", output_path])


def extract_indices(filename):
    file_as_list = []
    with open(filename, 'r') as f:
        for line in f:
            file_as_list.append(line)
    start = file_as_list[0].split('.')[0].split('_')[-1]
    end = file_as_list[len(file_as_list)-1].split('.')[0].split('_')[-1]
    hex_full = file_as_list[0].split('_')[-2]
    hex = hex_full.split('-')[0]
    return start, end, hex


# string -> datetime
def string_to_datetime(date, time):
    """
    Interprets a string date and time into a datetime object.

    Parameters
    ----------
    date: String
        A date in the format of yyyy-mm-dd

    time: String
        A time in the format of hh:mm:ss

    Returns
    ----------
    Datetime object including all input data.
    """
    parsed_date = date.split('-')
    parsed_time = time.split(':')
    year, month, day = int(parsed_date[0]), int(parsed_date[1]), int(parsed_date[2])
    hour, minutes, seconds = int(parsed_time[0]), int(parsed_time[1]), int(parsed_time[2])
    return datetime(year, month, day, hour, minutes, seconds)


# datetime -> list of files needed between the two date times
def datetime_to_videos(df, start_dt, end_dt):
    """
    Finds list of videos within a datetime range.

    Parameters
    ----------
    df: Dataframe
        A dataframe object containing time mappings to video names.

    start_dt: datetime

    end_dt: datetime

    Returns
    ----------
    out: Dataframe
        A dataframe containing truthy values for all selected videos.
    """
    df_starts = pandas.to_datetime(df['Start time (Human readable)'])
    df_ends = pandas.to_datetime(df['End time (Human readable)'])
    timezone = pytz.timezone("America/Los_Angeles")
    adjusted_dt = timezone.localize(start_dt)
    inds_start = df_ends >= adjusted_dt
    adjusted_dt = timezone.localize(end_dt)
    inds_end = df_starts <= adjusted_dt
    inds = inds_start.mul(inds_end)
    out = df['Video File'][inds]
    return out


if __name__ == '__main__':
    main(sys.argv[1:])
    # sys.exit(main(sys.argv[1:]))
    # dharshan used sys.exit in his vtc_to_csv, don't think it's neccesary?


# test dates 2017-07-18 9:10:11  2017-07-18 9:20:11
# cmd line python video_concatenation.py ECxxx 2017-07-18 02:10:11 2017-07-18 02:30:11 ~/Desktop/data/ ~/Desktop/data/McCulloch~\ Rya_3ef0b0f2-a734-43cd-9491-e1dfeba2d781.vtc ~/Desktop/output 0




# make the output of skipped files a log that can be read somewhre


# in the file name we want to list what the range of videos is that should be present e.g (skipped_files: 0001-0020)
# and then the text file itself should read Missing file: <filename>
# output dir to be the output path


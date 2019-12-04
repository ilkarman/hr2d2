import subprocess
import os
import json
import glob
import time
import cv2
from joblib import delayed
from joblib import Parallel

NUM_JOBS = 1
PATH = '/datasets/Moments_in_Time_Raw'
PREFIX ='sX_'

def process_ffmpeg(in_filename):
    # Splitting by / may break certain files, fix properly
    b = in_filename.split("/")
    outf = "/".join(b[:-1]) + '/%s%s' % (PREFIX, b[-1])

    # -an flag should remove audio
    # original paper uses 340x256 so do half-res for quick iter: 170x128
    # original paper uses 25fps, reduce also to 15 for quick iter
    # Trying to follow: https://github.com/activitynet/ActivityNet/blob/master/Crawler/Kinetics/download.py#L100
    command = ['ffmpeg',
               '-y ',
               '-i', '"%s"' % in_filename,
               '-r', '15' ,
               '-vf "scale=iw*min(170/iw\,128/ih):ih*min(170/iw\,128/ih),pad=170:128:(170-iw)/2:(128-ih)/2" ',
               '-c:v', 'libx264',
               '-threads', '1',
               '-an ',
               '-loglevel', 'panic',
               '"%s"' % outf]
    command = ' '.join(command)
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        # Verify output
        capture = cv2.VideoCapture(outf)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 32:
            print("Output", outf, frame_count)

    except Exception as err:
        print(err, in_filename)
        return in_filename
    # Have zip, so remove original
    os.remove(in_filename)
    return None

if __name__ == '__main__':
        
    input_files = [
        os.path.join(root, name)
        for root, dirs, files in os.walk(PATH)
        for name in files
        if name.endswith((".mp4")) and not name.startswith((PREFIX))
    ]
    print("{} Number of files to process".format(len(input_files)))
    start = time.time()

    error_lst = Parallel(n_jobs=NUM_JOBS)(delayed(process_ffmpeg)(v) for v in input_files)
    error_lst = [x for x in error_lst if x is not None]

    print("Finished")
    print(time.time() - start)
    print(error_lst)





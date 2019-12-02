import subprocess
import os
import json
import glob
import time
from joblib import delayed
from joblib import Parallel

NUM_JOBS = 40
PATH = '/datasets/Moments_in_Time_Raw'
PREFIX ='sX_'

def process_ffmpeg(in_filename):
    # Splitting by / may break certain files, fix properly
    b = in_filename.split("/")
    outf = "/".join(b[:-1]) + '/%s%s' % (PREFIX, b[-1])

    # -an flag should remove audio
    # original paper uses 340x256 so do half-res for quick iter: 170x128
    # original paper uses 25fps, reduce also to 15 for quick iter
    command = ['ffmpeg',
               '-i', '"%s"' % in_filename,
               '-vf "scale=iw*min(170/iw\,128/ih):ih*min(170/iw\,128/ih),pad=170:128:(170-iw)/2:(128-ih)/2" ' \
               '-r 15 ' \
               '-threads', '1',
               '-y ',
               '-an ',
               '-loglevel', 'panic',
               '"%s"' % outf]
    command = ' '.join(command)

    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        # Seem to lose only 4 files so ignore
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





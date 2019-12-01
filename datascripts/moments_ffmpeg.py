import subprocess
import os
import json
import glob
import time
from joblib import delayed
from joblib import Parallel

NUM_JOBS = 40
PATH = '/datasets/Moments_in_Time_256x256_30fps'

def process_ffmpeg(in_filename):
    # Splitting by / may break certain files, fix properly
    b = in_filename.split("/")
    outf = "/".join(b[:-1]) + '/s150_%s' % b[-1]

    # -an flag should remove audio
    command = ['ffmpeg',
               '-i', '"%s"' % in_filename,
               '-vf scale=150:150 ' \
               '-r 15 ' \
               '-c:v', 'libx264', '-c:a', 'copy',
               '-threads', '1',
               '-y ',
               '-an ',
               '-loglevel', 'panic',
               '"%s"' % outf]
    command = ' '.join(command)
    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print(err.output, in_filename)
        return in_filename

    return None

if __name__ == '__main__':
        
    input_files = [
        os.path.join(root, name)
        for root, dirs, files in os.walk(PATH)
        for name in files
        if name.endswith((".mp4")) and not name.startswith(("s150_"))
    ]

    print("{} Number of files to process".format(len(input_files)))
    start = time.time()

    error_lst = Parallel(n_jobs=NUM_JOBS)(delayed(process_ffmpeg)(v) for v in input_files)
    error_lst = [x for x in error_lst if x is not None]

    print("Finished")
    print(time.time() - start)
    print(error_lst)





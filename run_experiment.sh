obstypes=( VNC_MAX_MONO VNC_MAX_STEREO VNC_FFT_MONO VNC_FFT_STEREO VNC_MEL_MONO VNC_MEL_STEREO VIDEO_ONLY VIDEO_NO_CLUE )
for obstype in "${obstypes[@]}"
do
    echo "Started experiment for $obstype"
    nohup python experiment.py -obs_type $obstype > "experiments_log/$obstype.txt" 2>&1 &
done


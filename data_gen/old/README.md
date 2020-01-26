here goal.py policy.py searchtree.py and solver.py solve world dependent on which is the current object of interest, while the new versions solve the worlds based on the objects of interest that are fixed, and they only change when their condition is met

this is done so that if rms changes mid execution we continue to move the old rms

rms = rightmost_sphere

# palmreader
- Written by Zihe (Alex) Zhang
- zihe_zhang@g.harvard.edu

This github repository hosts the code used for data acuisition and analysis in the paper "Automated preclinical detection of mechanical pain hypersensitivity and analgesia." (https://journals.lww.com/pain/abstract/9900/automated_preclinical_detection_of_mechanical_pain.88.aspx)

The python code scripts (utils.py and analysis.py) together are self-contained, while the jupyter notebook (analysis.ipynb) is a walk-through of the functionality in the python code scripts.

The code here takes two video files (a pair of recordings of the body frame and the FTIR signal) and a hdf file containing the body-pose tracking done by a trained DeepLabCut neural network (https://github.com/DeepLabCut/DeepLabCut). Its output includes a video file of centered and aligned body frames, and a hdf file (features.h5) that contains the distance traveled by the animal during the recording, and paw luminance signals for both hind paws extracted from the FTIR frames.

## Instruction
- Please find a demonstrational data set to run the code here (https://www.dropbox.com/sh/x651jmxqbiq4qvr/AAAXHEU-YcuiO_s7dT0cgy-Sa?dl=0).
- The code here uses a trained DeepLabCut for tracking and does not include an instruction of how such training is done. Please refer to the official DeepLabCut repository, or reach out to me for further instructions on this.

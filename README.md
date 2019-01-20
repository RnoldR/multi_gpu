# multi_gpu
Testing the impact of multiple GPU's on my Keras RNN models. File rnn_demo.py contains the most simplified model I could think of now. I added some small data file in order to test it. File rnn_demo_plot.py plots the effect of single and and multi GPU runs. The results of one succesful run (ok.csv) are included. I got quite some issues with running multi GPU models. They are described below

## Issues with multi GPU
### Multu GPU slows down
See table below

|# GPU’s|Epochs|Batch size|acc|val_acc|Time (s)|%Utilisation|
|---|---|---|---|---|---|---|
|1|5|128|0.51|0.45|153|45|
|1|5|256|0.51|0.49|83|58|
|1|5|512|0.48|0.49|65|88|
|1|5|1024|0.43|0.42|50|88|
|2|5|128|0.21|0.27|490|120|
|2|5|256|0.21|0.27|261|130|
|2|5|512|0.17|0.20|150|140|
|2|5|1024|0.21|0.23|88|150|



## Test system
Hardware:
- AMD 1950X
- 96GB RAM
- 2 x 1080Ti

Software:
- Ubuntu 18.04
- Anaconda Spyder 3.3.1
- Python 3.5.6
- Tensorflow 1.9
- Keras 2.2.2

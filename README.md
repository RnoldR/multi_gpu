# multi_gpu
Testing the impact of multiple GPU's on my Keras RNN models. File rnn_demo.py contains the most simplified model I could think of now. I added some small data file in order to test it. File rnn_demo_plot.py plots the effect of single and and multi GPU runs. The results of one succesful run (ok.csv) are included. I got quite some issues with running multi GPU models. They are described below

## Issues with multi GPU
### Multu GPU slows down


|Epochs|Batch size|acc (1)|val_acc (1)|Time (1)|acc (2)|val_acc (2)|Time (2)|
|---|---|---|---|---|---|---|---|---|
|5|128|0.51|0.45|153|0.21|0.27|490|
|5|256|0.51|0.49|83|0.21|0.27|261|
|5|512|0.48|0.49|65|0.17|0.20|150|
|5|1024|0.43|0.42|50|0.21|0.23|88|



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

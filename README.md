# Multiple GPU's in Keras

Using a GPU instead of a CPU can really speed things up. An NVidia 1080Ti is in some of my benchmarks more than 60 times(!) faster than an AMD 1950X. And several articles [(here is one)](https://medium.com/@iliakarmanov/multi-gpu-rosetta-stone-d4fa96162986) hinted at extra speedups when using multiple GPU's. How much faster a multi GPU setup performs depends on the deep learning net that is being trained. I decided to try it out and I bought a seconds 1080Ti to my existing one. 

Testing the impact of multiple GPU's on my Keras RNN models. File rnn_demo.py contains the most simplified model I could think of now. I added some small data file in order to test it. File rnn_demo_plot.py plots the effect of single and and multi GPU runs. The results of one succesful run (ok.csv) are included. I got quite some issues with running multi GPU models. They are described below

## How to test GPU's present in your system
Tensorflow has function [`list_devices`](https://www.tensorflow.org/api_docs/python/tf/Session) to list which computing devices are present in your system. It lists All CPU's, GPU's and TPU's, so one has to filter the relevant devices. 
```
    print('Tensorflow version: ' + tf.__version__)
    print('Keras version: ' + keras.__version__)
    devices_s = K.get_session().list_devices()
    print('Computing devices\n', devices_s)

    gpu_s = [x for x in devices_s if 'GPU' in str(x)]
    print('\nof which GPU\'s\n', gpu_s)
    G = len(gpu_s)
    print("\n# of GPU's: " + str(G))
```
You can do more things as testing the GPU capabilities and the path of the CuDNN library. I gathered some examples from the web into gpu_test.py which you can find in the repository.

If you want to use the GPU's then the typecal way to proceed is as follows:

```
    if gpu > 1:
        with tf.device("/cpu:0"):
            model = create_model(X, Y, layers, dropout)
            model = multi_gpu_model(model, gpus=gpu)
            print('Running a multi GPU model')
    else:
        with tf.device("/gpu:0"):
            model = create_model(X, Y, layers, dropout)
            print('Running a single GPU model')

    model.compile(...)
```
As you can see keras makes it very easy to create multi GPU models. Trouble is that in my case multiple GPU's seem to slow in stead of speed up the training of an LSTM or GRU network. And I can't figure out what causes that. As an example I use a model that contyains 4 GRU layers of 512 units each and a Dense layer of 128 units. The results can be found in the table below where I have tested the accuracy and time to run the model 5 times for 1 GPU (1) and two GPU's (2).

|Epochs|Batch size|acc (1)|val_acc (1)|Time (1)|acc (2)|val_acc (2)|Time (2)|
|---|---:|---:|---:|---:|---:|---:|---:|
|5|128|0.51|0.45|153|0.21|0.27|490|
|5|256|0.51|0.49|83|0.21|0.27|261|
|5|512|0.48|0.49|65|0.17|0.20|150|
|5|1024|0.43|0.42|50|0.21|0.23|88|

As you can see the reult is somewhat disappointing. In all cases the use of two GPU's slows down the process and about halves the accuracy. That's where I started to experiment to try to figure out what exactly happens when using multiple GPU's.

## Issues with multi GPU
### Sensitivity for version
In my models I usually prefer GRU over LSTM. In my models it is faster and has better performance in terms of (validation) accuracy. However, the table listed above was produced by tensorflow 1.9 and Keras 2.2.2. When I tried to use Tensorflow/Keras 1.10/2.2.2 and 1.12/2.2.4 the system crashed without warning when I used 2 GPU's (it worked for 1 GPU). 

### CuDNN layers
I noticed some new layers, specifically for CuDNN: CuDNNGRU and CuDNNLSTM. They only work with the CuDNN library. They have two pleasant properties in my case:
- they really speedup the models, sometimes even by a factor 2, though by about 1.15 is more usual.
- they can be parallised in the newest versions of Tensorflow and Keras.

The big question for me of course was whether they are faster by the use of 2 GPU's.



### Multu GPU slows down


|Epochs|Batch size|acc (1)|val_acc (1)|Time (1)|acc (2)|val_acc (2)|Time (2)|
|---|---:|---:|---:|---:|---:|---:|---:|
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

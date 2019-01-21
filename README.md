# Multiple GPU's in Keras

**The article is not yet finished**

Using a GPU instead of a CPU can really speed things up. An NVidia 1080Ti is in some of my benchmarks more than 60 times(!) faster than an AMD 1950X. And several articles [(here is one)](https://medium.com/@iliakarmanov/multi-gpu-rosetta-stone-d4fa96162986) hinted at extra speedups when using multiple GPU's. How much faster a multi GPU setup performs depends on the deep learning net that is being trained. I decided to try it out and I bought a second 1080Ti to my existing one. 

I am aware that the results of using multiple GPU's depends on the task at hand. The reason publishing these (intermediate) results is that I don't know exactly what tasks impact in what way. Results I publish here are specific to my problem, fitting a lot of sequences with a Recurrent Neural Network (RNN). I am 

## How to test GPU's presence in the system
Tensorflow has the function [`list_devices`](https://www.tensorflow.org/api_docs/python/tf/Session) to list which computing devices are present in your system. It lists all CPU's, GPU's and TPU's, so one has to filter out the relevant devices. 
```
    from keras import backend as K
    import tensorflow as tf

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

If you want to use the GPU's then the typical way to proceed is as follows:

```
    if gpu > 1: # Create a model to run on all GPU's
        with tf.device("/cpu:0"):
            model = create_model(X, Y, layers, dropout)
            model = multi_gpu_model(model, gpus=gpu)
            print('Running a multi GPU model')
    else: # Create a model to run on GPU:0
        with tf.device("/gpu:0"):
            model = create_model(X, Y, layers, dropout)
            print('Running a single GPU model')

    model.compile(...)
```
As you can see keras makes it very easy to create multi GPU models. Trouble is that in my case multiple GPU's seem to slow down instead of speed up the training of an LSTM or GRU network. And I can't figure out what is the cause of that. As an example I use a model that contains 4 GRU layers of 512 units each and a Dense layer of 128 units. The results can be found in the table below where I have tested the accuracy and time to run the model 5 times for 1 GPU (1) and two GPU's (2).


|Epochs|Model type|Dropouts|Batch size|GPU's|Acc|Val. Acc|Time|Speedup|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|15|2|0|64|1|0.22|0.27|195||
|15|2|0|64|2|0.22|0.27|287|0.49|
|15|2|0|128|1|0.64|0.48|142||
|15|2|0|128|2|0.34|0.31|175|0.70|
|15|2|0|256|1|0.22|0.27|122||
|15|2|0|256|2|0.51|0.49|112|0.96|
|15|2|0|512|1|0.52|0.43|108||
|15|2|0|512|2|0.51|0.45|86|1.16|
|15|2|0|1024|1|0.41|0.37|100||
|15|2|0|1024|2|0.30|0.32|73|0.00|

As you can see the reult is somewhat disappointing. In all cases the use of two GPU's slows down the process and about halves the accuracy. That's where I started to experiment to try to figure out what exactly happens when using multiple GPU's.

## Issues with multi GPU

### Sensitivity for version
I usually prefer GRU over LSTM. It tends to be faster and has better performance in terms of (validation) accuracy. The table listed above was produced by tensorflow 1.9 and Keras 2.2.2. When I tried to use Tensorflow/Keras 1.10/2.2.2 and 1.12/2.2.4 the system crashed without warning when I used 2 GPU's (it worked for 1 GPU). 

### Version and layers that were compatible
In Keras there are RNN layers specific for the CuDNN library: CuDNNGRU and CuDNNLSTM. They only work with the CuDNN library. They have two pleasant properties:
- they really speedup the models, sometimes by even a factor 2, though by about 1.25 is more usual.
- they can be parallised in the newest versions of Tensorflow and Keras.

I wanted to know if they were more compatible with newer version of Tensorflow and Keras. Itested several combinations of layers and installed packages. That yielded the following results:

|Package/Layer|Combi 1|Combi 2|Combi 3
|---|---:|---:|---:|
|Tensorflow|1.9|1.1|1.12
|Keras|2.2.2|2.2.2|2.2.4
|Pyton|3.5.6|3.6.8|3.6.7
|GRU|O|X|X
|LSTM|X|X|X
|CuDNNGRU|O|?|O
|CuDNNLSTM|O|?|O

O = Ok, X = system hangs, ? = not tested

So for benchmarking I decided to use CuDNNGRU with my production systen: Tensorflow 1.9 and Keras 2.2.2.

### CuDNN layers

The big question for me of course was whether they are faster by the use of 2 GPU's.

|# GPU’s|Epochs|Batch size|Acc (1)|val_acc (1)|Time (1)|Acc (2)|val_acc (2)|Time (2)|Speed up
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:
|1|5|128|0.52|0.46|38|0.51|0.46|43|0.88
|1|5|256|0.51|0.49|29|0.51|0.49|27|1.07
|1|5|512|0.49|0.50|27|0.48|0.50|22|1.23
|1|5|1024|0.43|0.41|26|0.43|0.41|17|1.53

I noticed no differences between Tensorflow/Keras 1.9/2.2.2 and 1.12/2.2.4: the results were identical within the first two decimals. An image of these results is shown below.

![Accuracy and time for 1 and 2 CPU's plotted against batch size](https://raw.githubusercontent.com/RnoldR/multi_gpu/master/results.png)

There is break even for accuracy at a batch size of 256 and after that it increases. 

**Work in progress**

## Test system
Hardware:
- AMD 1950X
- 96GB RAM
- 2 x 1080Ti

Software:
- Ubuntu 18.04
- Anaconda Spyder 3.3.1
- Python 3.5.6
- Tensorflow 1.9/1.12
- Keras 2.2.2/2.2.4
- NVidia driversversion 415

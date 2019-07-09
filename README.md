
# Nano Setup

I find it easier to work on a desktop development machine and ssh into the nano when necessary (The nano UI can be a bit sluggish sometimes, so it's easier to work on a regular computer if you can) You will need to obtain the ip address of your nano. On the nano, open a terminal window and run the following command.

    $ ifconfig
    
Take note of the ip address of whatever network interface you are using (eth, wlan, etc)

On your development machine, open a terminal window and ssh into nano. Enter your password when prompted.

    $ ssh your_nano_username@your_nano_ip
    
You will now be connected to your nano inside the terminal window. Create a project directory for your tensorflow project in your home directory.

    $ cd 
    $ mkdir your_project_folder

## Python Virtual Environment Setup

We need to setup python on the nano. Using virtual environments is best practice to help with dependancy management. In this section, we will install a python package manager (pip) and setup a virtual environment using venv.

### Install and upgrade pip 

    $ python3 -m pip install --user --upgrade pip
    
You can check whether it installed successfully using the command below:

    $ python3 -m pip --version

Make sure python 3.x is listed next to the pip version number
   
### Venv setup

Install venv

    $ sudo apt-get install python3-venv
    
Now create a new virtual environment called env inside your project folder. Note that we use the -m flag (module) to use the venv python module.

    $ cd your_project_folder
    $ python3 -m venv env
    
Activate the virtual environment

    $ source env/bin/activate
    
You should see ```(env)``` at the beginning your terminal prompt. This indicates the current virtual environment. Whenever you install python packages, make sure you are using the correct virtual environment before running pip.

## Installing Tensorflow 

Installation steps are found [here](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html) and are listed below.

### Install tensorflow system dependencies

    $ sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev

### Install tensorflow python dependencies

    $ pip3 install -U numpy grpcio absl-py py-cpuinfo psutil portpicker six mock requests gast h5py astor termcolor protobuf keras-applications keras-preprocessing wrapt google-pasta
    
### Install the latest version of tensorflow-gpu for the nano. This is a special NVIDIA release for the jetson.

    $ pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu
    
### Verify the installation using the python interpreter

    $ python3
    >>> import tensorflow as tf
    >>> tf.VERSION

Write down the full version number, you will need it in a second. Exit the interpreter

    >>> exit()


## Install Jupyterlab

Jupyter lab is a browser based IDE-like experience for interactive jupyter notebooks. It will be used to run code on the nano in the browser of the development machine

    $ pip3 install jupyterlab
    
# Building the Keras Model

The following steps are taken from [here](https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano/) and are outlined below. 

## Google Colaboratory

Create a new Python 3 google colab notebook (File -> New Python 3 Notebook). You may also want to enable GPU acceleration (Runtime -> Change runtime type) and select GPU under the Hardware accelerator drop down menu. (It won’t affect anything since we are using a pre trained model, but it’s handy to know where that option is)

Google colab notebooks consist of individual cells of code that can be run by pressing shift+enter while inside the cell you want to run. In the next few steps, after you finish typing the code into the cell, always run it directly after by pressing shift+enter. If you need to add a new cell you can use the toolbar to select insert -> code cell

We need the tensorflow version on our colab notebook to match the version we have on our nano. In a new cell add the following code. Insert the version number (without quotes) that was output from your nano in where PUT_VERSION_HERE is:

```python
!pip install tensorflow==PUT_VERSION_HERE 
import tensorflow as tf
tf.VERSION
```
Press the restart runtime button in the output and rerun the cell. Make sure the output version number matches the one for your nano. If the versions are mismatched, the graph may not work on the nano.

We will want to be able to save and read things from our google drive, so the first step is to setup our notebook to use drive. You will want to create a project folder in your drive to store any files we need. After creating the folder in your drive, go back to the colab notebook. In a new cell, insert the following code to mount your google drive:


```python
from google.colab import drive
drive.mount('/content/gdrive')
```
    
Press shift+enter to run the cell, and a link will be output. Go to the link and copy the authorization code to complete the mounting process. You can use the left sidebar to access the files tab, and navigate to your project folder you created in your drive directory. Add a variable to store the path to the project folder using a new cell:

```python
root_path = '/content/gdrive/My Drive/your_project_folder'
``` 
Now we need to import the pretrained network (in this case MobileNetV2) and save the model as a .h5 file. In a new cell run:

```python
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as Net
model = Net(weights='imagenet')
model.save(f'{root_path}/keras_model.h5')
```

 You should see the .h5 file in your project directory on your mounted drive now. 
 
 ## Freezing the graph
 
 Now we need to freeze the graph. In a new cell run:
 
 ```python
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model

tf.keras.backend.clear_session()

def freeze_graph(graph, session, output, save_pb_dir=root_path, save_pb_name='frozen_model.pb', save_pb_as_text=False):
  with graph.as_default():
    graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
    graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
    graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
    return graphdef_frozen

tf.keras.backend.set_learning_phase(0) 

model = load_model(f'{root_path}/keras_model.h5')

session = tf.keras.backend.get_session()

input_names = [t.op.name for t in model.inputs]
output_names = [t.op.name for t in model.outputs]

# Prints input and output nodes names, take notes of them.
print(input_names, output_names)

frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=root_path)

``` 
However, we need to convert the graph for use with TensorRT (the graph will be optimized to run on the nano). We will also save the converted graph to drive. In a new cell run:

```python
import tensorflow.contrib.tensorrt as trt

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)
```
We need to save this new graph to google drive. In a new cell run the following code:

```python
graph_io.write_graph(trt_graph, root_path, 'trt_graph.pb', as_text=False)
```
You should see the trt_graph.pb file in your google drive directory now.

# Inferencing on the Nano

## Transfer the files

We now need to transfer the trt graph file to the nano. You can download the trt_graph file to the development machine by right clicking the file in your google drive directory using the file browser and selecting download. We will transfer the file to the nano using scp (secure copy). Go back to your terminal window. If you have a ssh session running in your terminal exit it:

    $ exit
    
Transfer the file to the nano and put it in the project directory

    $ scp ~/Downloads/trt_graph.pb your_jeston_username@jetson_ip:~/your_project_folder
    
We also want to transfer an image to perform the inference on. Download a google image of an elephant. Any image should do. Transfer it to the project directory using scp

    $ scp ~/Downloads/elephant.jpeg your_jeston_username@jetson_ip:~/your_project_folder

    
If you ssh back into the jetson you should see the files in the project directory now:

    $ ssh your_nano_username@your_nano_ip
    $ ls ~/your_project_folder
    
## Install Pillow

Pillow is needed to handle loading the image of the elephant into python, so lets go ahead and install that

    $ source env/bin/activate
    $ pip3 install pillow

## Start Jupyter Lab
It would be nice to be able to run code using jupyter lab on the nano and access it from the development machine. We can do this by forwarding the jupyter port. Exit the current SSH session 

    $ exit 
    
and start a new one, but this time forward the jupyter port on your remote jetson (8888) to another port on the development box (8000). 

    $ ssh -L 8000:localhost:8888 your_jetson_username@your_jetson_ip
    
This allows us to access the jupyter lab server running on the nano (port 8888) on our development machine (port 8000). Its common to use different port numbers for the remote and development machines. 
    
We need to start the jupyter server on the nano. Be sure and switch to your virtual env. Since we wont be using a browser on the jetson, we can pass in the --no-browser-flag:

    $ cd ~/your_project_folder
    $ source env/bin/activate
    $ jupyter lab --no-browser
    
Copy and paste the link into the address bar of a browser on your development machine. Change the port number from 8888 to 8000 and hit enter. You should see the jupyter lab running with your project files in the left side bar. Create a new python 3 notebook by pressing the python 3 button in the main page. 

## Importing the Graph

In a new cell, insert and run the following code to import the graph

```python
output_names = ['Logits/Softmax']
input_names = ['input_1']

import tensorflow as tf


def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


trt_graph = get_frozen_graph('./trt_graph.pb')

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')


# Get graph input size
for node in trt_graph.node:
    if 'input_' in node.name:
        size = node.attr['shape'].shape
        image_size = [size.dim[i].size for i in range(1, 4)]
        break
print("image_size: {}".format(image_size))

# input and output tensor names.
input_tensor_name = input_names[0] + ":0"
output_tensor_name = output_names[0] + ":0"

print("input_tensor_name: {}\noutput_tensor_name: {}".format(
    input_tensor_name, output_tensor_name))

output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)
```
## Importing the image and making a prediction

Run the following code in a new cell. You will need to update the line that sets the image path to your image name. 

```python
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# Change the file name to match yours
img_path = './elephant.jpeg'

img = image.load_img(img_path, target_size=image_size[:2])
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

feed_dict = {
    input_tensor_name: x
}
preds = tf_sess.run(output_tensor, feed_dict)

# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
```
You should see the top 3 predictions of what the image is output. 

## Benchmarking

We can run a benchmark on the nano by running the following code in a new cell

```python
import time
times = []
for i in range(20):
    start_time = time.time()
    one_prediction = tf_sess.run(output_tensor, feed_dict)
    delta = (time.time() - start_time)
    times.append(delta)
mean_delta = np.array(times).mean()
fps = 1 / mean_delta
print('average(sec):{:.2f},fps:{:.2f}'.format(mean_delta, fps))
```
I was running at around 30 fps on my nano. 



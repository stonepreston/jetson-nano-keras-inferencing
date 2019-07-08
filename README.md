
# Nano Setup

I find it easier to work on a development machine (not the nano) and ssh into the nano when necessary. You will need to obtain the ip address of your nano. On the nano, open a terminal window and run the following command. 

    $ ifconfig
    
Take note of the ip address of whatever network interface you are using (eth, wlan, etc)

On your development machine, open a terminal window and ssh into nano. Enter your password when prompted.

    $ ssh your_nano_username@your_nano_ip
    
You will now be connected to your nano inside the terminal window. 

Create a project directory for your tensorflow project.

    $ cd 
    $ mkdir mobile_net_test

## Python Virtual Environment Setup

We need to setup python on the nano. Using virtual environments is best practice to help with dependancy management. In this section, we will install a python package manager (pip) and setup a virtual environment using venv.

### Install and upgrade pip 

    $ python3 -m pip install --user --upgrade pip
    
You can check whether it installed successfully using the command below:

    $ python3 -m pip --version
    
Install venv

    $ sudo apt-get install python3-venv
    
Now create a new virtual environment called env
    $ cd mobile_net_test
    $ python3 -m venv env
    
Activate the virtual environment

    $ source env/bin/activate

You should see ```(env)``` in your terminal prompt. This indicates the current virtual environment. Whenever you install python packages, make sure you are using the virtual environment before running pip.

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
    >>> import tensorflow
    
If no errors are displayed tensorflow was installed correctly. Exit the the interpreter

    >>> exit()


## Install Jupyterlab

Jupyter lab is a browser based IDE-like experience for interactive jupyter notebooks. It will be used to run code on the nano in the browser of the development machine

    $ pip3 install jupyterlab
    
# Building the Keras Model

The following steps are taken from [here](https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano/) and are outline below. 

## Google Colaboratory

Create a new Python 3 [google colab](https://colab.research.google.com/) notebook (File -> New Python 3 Notebook). You will also want to enable GPU acceleration (Runtime -> Change runtime type) and select GPU under the Hardware accelerator drop down menu.

We will want to be able to save and read things from our google drive, so the first step is to setup our notebook to use drive. You will want to create a project folder in your drive to store any files we need. After creating the folder in your drive, go back to the colab notebook.  In a new cell, insert the following code to mount your google drive:

```python
from google.colab import drive
drive.mount('/content/gdrive')
```
    
Press shift+enter to run the cell, and a link will be output. Go to the link and cipy the authorization code to complete the mounting process. You can use the left sidebar to access the files tab, and navigate to your project folder you created in your drive directory. Add a variable to store the path to the project folder using a new cell:

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
However, we need to convert the graph using TensorRT (the graph will be optimized to run on the nano). In a new cell run:

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
    
We now need to transfer the trt graph file to the nano. You can download it to the development machine by right clicking the file in the file browser and selecing download. We will transfer the file to the nano using scp (secure copy). Go back to your terminal window. If you have a ssh session running in your terminal exit it:

    $ exit
    
Transfer the file to the jetson and put it in the Downloads directory

    $ scp /home/your_username/Downloads/trt_graph.pb your_jeston_username@jetson_ip:~/Downloads
    
If you ssh back into the jetson you should see the file in the downloads directory now:

    $ ssh your_nano_username@your_nano_ip
    $ ls ~/Downloads
    
It would be nice to be able to run jetson code using jupyter lab and access it from the development machine. We can do this by forwarding the jupyter port. Exit the current SSH session and start a new one, but this time forward the jupyter port on your remote jetson (8888) to another port on the development box. 

    $ ssh -L 8000:localhost:8888 your_jetson_username@your_jetson_ip
    
This allows us to access the jupyter lab server running on the jetson (port 8888) on our development machine (port 8000). Its common to use different port numbers for the remote and development machines. Lets make a directory to store the project files on the jetson and move the trt file to this directory:

    $ mkdir ~/Documents/mobile_net_project
    $ mv ~/Downloads/trt_graph.pb ~/Documents/mobile_net_project
    

We need to start the jupyter server on the jetson in this new directory. Be sure and switch to your virtual env. Since we wont be using a browser on the jetson, we can pass in the --no-browser-flag:

    $ cd ~/Documents/mobile_net_project
    $ workon your_environment_name
    $ jupyter lab --no-browser
    
In the browser of your development machine, navigate to ```http://localhost:8000``` and you should see the jupyter lab running with your trt file in the left side bar. Create a new python 3 notebook by pressing the python 3 button in the main page. 



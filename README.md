
# Nano Setup

I find it easier to work on a development machine (not the nano) and ssh into the nano when necessary. You will need to obtain the ip address of your nano. On the nano, open a terminal window and run the following command. 

    $ ifconfig
    
Take note of the ip address of whatever network interface you are using (eth, wlan, etc)

On your development machine, open a terminal window and ssh into nano. Enter your password when prompted.

    $ ssh your_nano_username@your_nano_ip
    
You will now be connected to your nano inside the terminal window. 

## Python Virtual Environment Setup

We need to setup python on the nano. Using virtual environments is best practice to help with dependancy management. In this section, we will install a python package manager (pip) and setup a virtual environment.

### Install and upgrade pip 

    $ sudo apt-get install python3-pip
    $ sudo pip3 install -U pip
    
### Install virtualenv and virtualenv wrapper to help manage different python environments

    $ sudo pip install virtualenv virtualenvwrapper
    
### Configure user profile to use virtualenv

    $ nano ~/.bashrc
    
Add the following lines to the end of the .bashrc file

    # virtualenv and virtualenvwrapper
    export WORKON_HOME=$HOME/.virtualenvs
    export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
    source /usr/local/bin/virtualenvwrapper.sh

Close the terminal window and open a new terminal. This will allow the changes to the bashrc file to take effect. Alternatively, you can run 

    $ source ~/.bashrc

Now create a new virtual environment
    
    $ mkvirtualenv your_environment_name -p python3
    
Make sure you are on the virtual environment you created 

    $ workon your_environment_name

You should see ```(your_environment_name)``` in your terminal prompt. This indicates the current virtual environment. Whenever you install python packages, make sure you are using the virtual environment before running pip.

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

    from google.colab import drive
    drive.mount('/content/gdrive')
    
Press shift+enter to run the cell, and a link will be output. Go to the link and cipy the authorization code to complete the mounting process. You can use the left sidebar to access the files tab, and navigate to your project folder you created in your drive directory. Add a variable to store the path to the project folder using a new cell:

    root_path = '/content/gdrive/My Drive/your_project_folder'
    
Now we need to import the pretrained network (in this case MobileNetV2) and save the model as a .h5 file. In a new cell run:

    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as Net
    model = Net(weights='imagenet')
    model.save(f'{root_path}/keras_model.h5')
    
 You should see the .h5 file in your project directory on your mounted drive now. 
 
 
 

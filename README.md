
# Nano Setup

I find it easier to work on a development machine (not the nano) and ssh into the nano when necessary. You will need to obtain the ip address of your nano. On the nano, open a terminal window and run the following command. 

    $ ifconfig
    
Take note of the ip address of whatever network interface you are using (eth, wlan, etc)

On your development machine, open a terminal window and ssh into nano. Enter your password when prompted.

    $ ssh your_username@your_nano_ip
    
## Python Virtual Environment Setup

### Install and upgrade pip (python package manager)

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

You should see ```(your_environment_name)``` in your terminal prompt. This indicates the current virtual environment. 

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

    $ pip3 install jupyterlab
    
 

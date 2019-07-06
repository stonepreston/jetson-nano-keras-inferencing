
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


MI-MVI.16 SW tools setup
========================
This tutorial will guide you through the instalation of the basic tool set necessary for the first MI-MVI class. You will need at least **400 MB** of free space for the virtual environment with installed tools.

Lets start with Python 3 instalation and **virtual environment (VE)** for separate python instances:
```bash
sudo apt-get install python3 python3-pip virtualenv
```
Prepare dir for all VEs and create our VE:
```bash
mkdir ~/venvs
virtualenv ~/venvs/mi-mvi -p python3 --no-site-packages
cd ~/venvs/mi-mvi/bin
source activate
```
Your prompt should change to something similar to:
```bash
(mi-mvi) user@host~/venvs/mi-mvi/bin $ 
```
Lastly, we will install Jupyter, a Python package for interactive notebooks, and the Tensorflow library:
```bash
./pip install jupyter tensorflow
```
(Do not forget to call a local **./pip** in VE if you don't want to affect the global python instalation. The same rule applies also for other binaries.)

To exit VE simply type:
```bash
source deactivate
```

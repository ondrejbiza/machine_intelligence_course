MI-MVI.16 SW tools setup
========================

Lets start with python3 instalation and virtual environment (VE) for separate python instances:
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
At last we install Jupyter a python for interactive notebooks and Tensorflow library:
```bash
./pip install jupyter tensorflow
```
(Do not forget to call a local **./pip** in VE if you don't want to affect the global python instalation. The same rule apply also for other binaries)

To exit VE simply type:
```bash
deactivate
```

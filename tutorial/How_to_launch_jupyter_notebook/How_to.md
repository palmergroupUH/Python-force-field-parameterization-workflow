## Tutorial 0: Open a Jupyter notebook from a remote server.

#### Step 1: On your local machine(e.g. Desktop) ssh into the cluster using: "ssh -L localhost:8888:localhost:8889 'remote server address'"

* The port number: 8888 or 8889 may be changed if they are used by others.
* Once you login, make sure you have activated a conda environment and you are at the folder where the .ipynb file lies.

#### Step 2: On the remote server (your cluster), Run command "jupyter notebook --no-browser --port=8889". This "8889" should match with the localhost:8889. 

* You will see many output messages came out. Make sure you locate the token number (like 88asdfsdfx9238) buried inside the message

#### Step 3: On your local machine, Open the url: http://localhost:8888 from your web browser. The port number "8888" should match with the localhost:8888 upon login

* You will be prompted to give a token number for loging in. Copy and paste the token number to log in. 

#### Step 4: Once you give a correct token number, you should be able open the Jupyter notebook in your local browser. 


# RL_playground

## Installation 
1. Locally install RL_playground package:
	  ``` bash
	  virtualenv -p python3 env
	  source env/bin/activate
	  git clone https://github.com/PPSantos/RL_playground/
	  pip install -r RL_playground/requirements.txt
	  pip install -e .
    ```
2. Install acme framework (githash bc1b17e3a6731b520ba8c57e566a0b105ab5f8d1)
	```bash
	git clone https://github.com/deepmind/acme.git
	pip install dm-acme[reverb]
	pip install dm-acme[tf]
	pip install dm-acme[jax]
	pip install -e acme/
	```
3. Build package
	  ``` bash
	  cd RL_playground
	  make build
    ```
 4. Test installation
	  ``` bash
	  python train.py
    ```

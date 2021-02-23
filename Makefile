train:
		python src/train.py
demo:
		python src/asl_demo.py

lint:
		python -m flake8 src

fix:
		python -m black /src/

install:
		python -m pip install -r requirements.txt
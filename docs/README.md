## Generate the documentation


Alternatively, a copy of the master documentation can be computed: 
you will need Sphinx, a Documentation building tool, and a nice-looking custom 
[Sphinx theme similar to the one of readthedocs.io](https://sphinx-rtd-theme.readthedocs.io/en/latest/):

```
pip install -r requirements.txt
```

### 1) *optional*: setup sphinx (already done)
```
sphinx-quickstart
```
The source/conf.py is already setup


### 2) To automatically generate the documentation:
```
sphinx-apidoc -f -o source ../cBrainMRIPrePro
```

### 3) Then to build the html or LaTeX version: 
```
cd docs
make html or make latexpdf
```
The html will be available within the folder [docs/build/html](docs/build/html/index.html).
The html will be available within the folder [docs/build/latex](docs/build/latex/cBrainMRIPrePro.pdf).
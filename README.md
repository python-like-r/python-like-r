## Python Like R

### Proposal:
To build a package that duplicates the functionality, internal structure, and output structure of Rs linear model package (add others!).

### Media:
- [slides](https://docs.google.com/presentation/d/1uU3VYMR0x534Y9AzJEnAh8KkuQB31IuNeEATOC8m9Tk/edit?usp=sharing)
- [project overview](https://youtu.be/WaoWktauL2k)
- [example use](https://youtu.be/2iztNpPdMgA)
- [code overview](https://www.youtube.com/watch?v=3inFsPJ4tno&ab_channel=MuthukaruppanAnnamalai)


### Motivation:
Many of the world's best data scientists have not come from a programming background but rather from a statistics background or domain knowledge background. For many of these people R is their first programming language, in large part because of its ease of use and relative statistical power. That said, R still has a long way to go when it comes to computational efficiency and ability to integrate well with other languages. For that reason, many data scientists after learning R also pick up Python. The goal of our project is to build a tool that will be helpful for those making this transition. We hope to use much of the same syntax and structure that R does, but also to make minor changes that will help people learn Python structure.

### Tools used:
- [R](https://www.r-project.org/). While no R code was used in this repo, we are mimicking the use and syntax of the R language.
- [git/github](https://github.com/). Our project lives here and all development was done using Git as our source control.
- [pipenv](https://pypi.org/project/pipenv/). Pipenv is a package manager that is helpful for managing dependencies and ensuring reproducibility.
- [pytest](https://docs.pytest.org/en/stable/). Pytest is An excellent tool for unit testing in python, all of our tests can be found in tests/test.py.
- [cookiecutter](https://github.com/cookiecutter/cookiecutter). Cookie cutter is an excellent tool for reproducing project structure, we used it to generate starter code for this project.
- [SKLearn](https://scikit-learn.org/stable/). We implement lm ourselves from scratch, but the back end for both our knn models is from SKLearn.


### How to use this repo.
1. fork or clone the repo to your local computer.
2. install the environment.
    1. install pipenv `pip install pipenv` if needed.
    2. install form our pipfile `pipenv install --dev`.
3. Use the environment.
    1. activate the environment `pipenv shell`
    2. run our code!
        1.tests can be run using `pytest -v`
        2.main can be run using `python3 -m src`
        3.view our example notebooks `jupyter notebook` then open Example_python.ipynb
        
### Important classes
The Three most important classes we developed in this project where the utility/FormulaParser, the models/BaseModel, and models/lm. 
- FormulaParser allows us to parse and return a column set from a string so that practitioners only need to write a formula as a string rather than specifying a design matrix and response vector.
- BaseModel provides the structure of any model built using our package. Importantly it includes a getX function that allows you to extract a design matrix using a formula.
- lm is the main model we duplicated from R.

### Concepts from CSCI 29:
- Package management with Pip/pipenv (Pipfile and Pipfile.lock)
- Source control with Git and GitHub 
- Test-driven development using pytest (src/tests), All the formulas and the R methods are thoroughly tested
- Inheritance (BaseModel/BaseRegressor/lm)
- Descriptors and properties (Utility/FormulaParser)
- Decorators  (Utility/FormulaParser)
- Map (Utility/helper)
- Workflows 
- We will be thinking about how our library will be used as part of a task based DAG




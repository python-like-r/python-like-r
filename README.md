## Python Like R

### Proposal:
To build a package that duplicates the functionality, internal structure, and output structure or Rs linear model package.

### Motivation:
Many of the world's best data scientists have not come from a programming background but rather from a statistics background or domain knowledge background. For many of these people R is their first programming language, in large part because of its ease of use and relative statistical power. That said, R still has a long way to go when it comes to computational efficiency and ability to integrate well with other languages. For that reason, many data scientists after learning R also pick up Python. The goal of our project is to build a tool that will be helpful for those making this transition. We hope to use much of the same syntax and structure that R does, but also to make minor changes that will help people learn Python structure.

### Why is this a good project for CSCI 29:
We will be specifically developing a tool for data scientists, not doing a data science project.
We will need to think carefully about the class structure we use so that our code could easily be extended into a larger, potentially open source, project.
We will need to optimize our code for efficiencies that it can run on arbitrarily large data sets.

### Concepts from the class:
- Package management with Pip/pipenv (Pipfile and Pipfile.lock)
- Source control with Git and GitHub 
- Test-driven development using pytest (src/tests), All the formulas and the R methods are thoroughly tested
- Inheritance (BaseModel/lm)
- Descriptors and properties (Utility/FormulaParser)
- Decorators  (Utility/FormulaParser)
- Map (Utility/helper)
- Workflows 
- We will be thinking about how our library will be used as part of a task based DAG


## Try out our repo! 
- Fork from: https://github.com/python-like-r/python-like-r/
- Install the environment using `pipenv install --dev`
- Activate the environment `pipenv shell`
- Start jupytor notebook `jupytor notebook`
- Open “Example_python.ipynb” - All the samples are shown here for all models
- Try with your favorite dataset.
- Post suggestions on this post about features you would like to see!


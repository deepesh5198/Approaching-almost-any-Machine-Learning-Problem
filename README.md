# Approaching-almost-any-Machine-Learning-Problem-Book
My notes of Approaching (Almost) any Machine Learning Problem book.

As the name of the book suggests, this book is about how you can approach any machine learning problem. It gives you a road map to any machine learning project, from setting up your Virtual Environment to Deploying your model.

## Table of Content

- [Chapter 1 - Setting up Your Working Environment](#chapter1)
- [Chapter 2 - Supervised vs Unsupervised Learning](#chapter2)
- [Chapter 3 - Cross Validation](#chapter3)


<a name = "chapter1">
<h1>Chapter 1 - Setting up Your Working Environment</h1>
</a>

This chapter is about setting up working environment for your ML project. The first step to any ML project is creating a working environment for your project.

And the first step in setting up a working environment for an ML project is, installing python. 

### How to install python?
Just go to [Download Python](https://www.python.org/downloads/) page, and download a stable version of python and install it in your system.

Thats it!, now you have Python in your system.
You can open command prompt and type, "python or py" and hit enter, to check if it has installed or not. If it is installed you will see something like this.

![Alt text](https://i.stack.imgur.com/K5kSC.png "check python")


Now that we have installed python in our system, its time to set up a Virtual Environment.

### Why set up a new Virtual Environment?

- To keep Python's Built-in packages separate from third party packages.
- To install all our project dependancies (packages and libraries) in our working environment.
- Installing third party packages in Python environment can cause issues.
- Also installing dependancies in a virtual environment makes it easy to migrate our project to different system.

### How to set up a Virtual Environment?
[How to create a python Virtual Environments](https://docs.python.org/3/library/venv.html) this page provides you a very easy explanation for creating Python virtual environment. 

### Installing python packages

Once you have created a virtual environment, Activate the environment by running "activate.bat" from your environment directory. After activating your environment your environment name will show in the brackets, like this "(project_env)" in the picture below:

![Alt text](https://miro.medium.com/max/1095/1*oTcSPKxWdQe_jNh7yYDsNg.png "activate venv")


Once you have successfully activated your environment, just type "pip install package_name" and hit enter, the package will install. After installing all important packages like numpy, pandas, matplotlib, seaborn, scikit-learn, and jupyter lab. You are done with setting up your working environment. Now, we can begin writing programs for our project.

### Where to write Code?

Just open a fresh command prompt, activate your virtual environment and type "jupyter notebook" and hit enter. This will open Jupyter Notebook in your browser, here you can write your python code and also import all the packages you installed for your project.


<a name = "chapter2">
<h1>Chapter 2 - Supervised vs Unsupervised Learning</h1>
</a>

In this chapter we will learn about types of Machine Learning Problems. There are mainly three types of ML problems:
- Supervised Learning
- Unsupervised Learning
- Semi-supervised learning or Reinforcement Learning
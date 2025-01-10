# danish_to_english_llm

Translate short sentences from Danish to English using LLM

# Project description
**Overall goal of the project**

This is the project description from group 26 for the course "02476 Machine learning operations" at the Technical University of Denmark. The overall goal of this project is to train a machine learning system to translate sentences and words from danish to english. Additionally, the goal of this project is to apply different MLOps techniques, taught in this course, to the ML translation problem. Lastly, the findings of this project will be presented in a report style README file in the "reports" folder. 

**What framework are you going to use, and do you intend to include the framework into your project?**

For this project, we plan on using the the transformers framework, which can be found [here](https://github.com/huggingface/transformers). This framework provides a lot of different pre-trained models, depending on the task, and the plan is to use of these pre-trained models to be trained on our data. Thereby be fined tuned for translating sentences from danish to english. The reason for using a pre-trained model is to save time and reduce the need for building the model from scratch, such that we can focus on the more important aspects of this course, which are the MLOps techniques. 

**What data are you going to run on (initially, may change)?**

The dataset used in this project is called "opus-Danish-to-English" and can be found [here](https://huggingface.co/datasets/kaitchup/opus-Danish-to-English). It consists of 946K rows where each row contains both the danish and the english translation where they are seperated by "###>".  

**What models do you expect to use?**

As mentioned before, we plan on using a pre-trained model from the transformer framework. To be more specific, we expect to use a pre-trained model found on Hugging Face called "t5-small" which can be found [here](https://huggingface.co/google-t5/t5-small). This model has been chosen because of its relative small size (60.5M parameters) and its ability to translate languages. 

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

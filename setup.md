## Setup:
Assuming Anaconda is installed 
the following commands will set up the python environment:

    make conda-cpu
    make pip-tools

## Update requirements:
    
Add python packages for:

development:

    requirements/dev.in

production:
    
    requirements/prod.in 

linting, code quality checks etc:

    requirements/lint.in


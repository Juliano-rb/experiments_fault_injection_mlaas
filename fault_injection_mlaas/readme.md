1. install a new dependency: pipenv install openpyxl
2. run project: 
    1. activate the envrioment: ``pipenv shell``
    2. install dependencies: ``pipenv install``
    3. run with:
        - ``python .\experiment1_fault_injection_mlaas.py`` or
        - ``python .\experiment2_fault_injection_mlaas.py``
    4. or run jupyter-lab with ``jupyter-lab``
3. In experiment 1 is possible to continue form previously state by using the param ``--continue_from``. Ex:
    - ``python .\experiment1_fault_injection_mlaas.py --continue_from "size198_05-31-2022 20_21_54"``

4. In experiment 2 is possible to continue form previously state by fixing the value of variable ``timestamp`` to something like: ``"07-04-2022 20_19_59"``
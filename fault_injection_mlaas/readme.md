1. install a new dependency: pipenv install openpyxl
2. run project: 
    1. activate the envrioment: ``pipenv shell``
    2. install dependencies: ``pipenv install``
    3. run with:
        - ``python .\fault_injection_text.py`` or
        - ``python .\fine_tune_fault_injection_mlaas.py``
    4. or run jupyter-lab with ``jupyter-lab``
3. is possible to continue form previously state by using the param ``--continue_from``. Ex:
    - ``python .\fault_injection_text.py --continue_from "size198_05-31-2022 20_21_54"``
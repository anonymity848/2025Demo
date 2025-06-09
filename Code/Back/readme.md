# url setting
Set the LLM link in the agent.py.

# InteractiveRL Environment Setup

## Environment Setup

To set up the environment with the necessary packages, follow the steps below:

1. Create a virtual environment:
    ```bash
    python3 -m venv InteractiveRL
    source InteractiveRL/bin/activate
    ```

2. Install the required packages:

    - WebSockets and asyncio:
        ```bash
        pip install websockets asyncio
        ```

    - Flask:
        ```bash
        pip install flask
        ```

    - PyTorch (torch, torchvision, torchaudio):
        ```bash
        pip install torch torchvision torchaudio
        ```

    - Numpy:
        ```bash
        pip install numpy
        ```

    - Scipy:
        ```bash
        pip install scipy
        ```

    - Matplotlib:
        ```bash
        pip install matplotlib
        ```

    - SWIGLPK:
        ```bash
        pip install swiglpk
        ```

    - CVXPY:
        ```bash
        pip install cvxpy
        ```

    - QPsolvers:
        ```bash
        pip install qpsolvers
        ```

    - Gunicorn:
        ```bash
        pip install gunicorn
        ```

3. To deactivate the virtual environment, use:
    ```bash
    deactivate
    ```

4. To remove the environment:
    ```bash
    rm -rf InteractiveRL
    ```

## Starting the Backend Server

1. Navigate to the `pythonProject` folder:
    ```bash
    cd pythonProject
    ```

2. Modify the port number in `main.py` (default is `8090`) if needed.

3. Start the backend server:
    ```bash
    python main.py
    ```

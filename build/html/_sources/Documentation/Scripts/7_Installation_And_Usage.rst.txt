VII - Usage
===========

First, ensure you have a **CUDA/cuDNN version compatible with your GPU** installed.

1. Clone the Repository
-----------------------

.. code-block:: bash

   git clone https://github.com/sohaibdaoudi/ChangingTireAssistant_CV_NLP_Project.git

2. Create and Activate Virtual Environment (Recommended)
--------------------------------------------------------

**For ``venv`` users:**

.. code-block:: bash

   # Create the virtual environment with Python 3.9
   python3.9 -m venv venv
   
   # Activate the environment
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   
   # Change to project directory after activating environment
   cd C:\path\to\folder\ChangingTireAssistant_CV_NLP_Project\Algorithm_V4

**For ``conda`` users:**

.. code-block:: bash

   # Create the conda environment with Python 3.9
   conda create --name tire-assistant python=3.9
   
   # Activate the environment
   conda activate tire-assistant
   
   # Change to project directory after activating environment
   cd C:\path\to\folder\ChangingTireAssistant_CV_NLP_Project\Algorithm_V4

3. Install Dependencies
-----------------------

.. code-block:: bash

   pip install -r Algorithm_V4/requirements.txt

4. Run the System
-----------------

This is the main script to run the full assistant:

.. code-block:: bash

   python Algorithm_V4/AlgoV4.py
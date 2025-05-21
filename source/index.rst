.. _top:

Changing Tire Assistant – Computer Vision & NLP Project
=======================================================

|Python| |License|

.. |Python| image:: https://img.shields.io/badge/python-3.x-blue.svg
   :target: https://www.python.org/downloads/
.. |License| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: ./LICENSE

🚧 Project Status: Under Development 🚧
---------------------------------------

.. note::
   This project is currently in active development. Core functionality is implemented but requires further optimization for production use.

📖 Overview
-----------

This assistant leverages real-time computer vision and NLP to guide users through changing a flat tire using an egocentric (chest-mounted) camera. The assistant:

- Detects tools and components in real-time
- Tracks task progression through action recognition
- Provides interactive voice and visual instructions
- Operates with low latency on edge devices

🔍 System Features
------------------

**Core Capabilities:**
- **Real-Time Tool Detection**:
  - Identifies car jack, wheel wrench, lug nuts, and spare tire
  - Handles occlusions and varying lighting conditions
- **Action Recognition**:
  - Tracks 10+ key steps in tire-changing process
  - Moniors proper sequence validation
- **Voice Assistant**:
  - Natural language interaction via Whisper STT
  - Context-aware responses to user queries
- **Edge Optimization**:
  - Quantized models for mobile deployment
  - <500ms end-to-end latency target

📂 Project Structure
--------------------

.. code-block:: none

   ├── action_recognition/    # Action classification models and training
   ├── Object Detection/      # YOLOv8 implementation and custom weights
   ├── requirements.txt       # Python dependencies
   ├── data_processing/       # Dataset preprocessing scripts
   └── deployment/            # Edge deployment configurations

🧠 Models Used
--------------

Object Detection
^^^^^^^^^^^^^^^^
- YOLOv8n fine-tuned on tire-change tools
- 640x640 input resolution
- Custom classes: {jack, wrench, lug_nut, spare_tire}

Action Recognition
^^^^^^^^^^^^^^^^^^
- Experimental architectures:
  - SlowFast R50
  - TSM (Temporal Shift Module)
  - TimeDistributed EfficientNetB0
- Input: 16-frame clips @ 224x224

Voice Assistant
^^^^^^^^^^^^^^^
- Whisper Base for speech-to-text
- Custom intent recognition pipeline
- Text-to-speech via pyttsx3

📊 Data
-------

Data Collection Methodology
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- **Primary Source**:
  - 4.5 hours of egocentric footage
  - Samsung A50 @ 1080p/30fps
  - Renault Megane 2 scenarios
- **Secondary Source**:
  - 120 curated YouTube videos
  - Multiple vehicle types and environments

Dataset Structure
^^^^^^^^^^^^^^^^^

.. code-block:: none

   data/
   ├── lower_car/             # 34 video clips
   ├── lift_car_with_jack/    # 28 video clips  
   ├── tighten_bolts/         # 41 video clips
   └── ...                    # Other action classes
   ├── labels.csv             # Temporal annotations
   └── dataset_report.pdf     # Statistical analysis

.. image:: https://github.com/user-attachments/assets/6d6bef7f-5d31-4b78-b57b-93b3566c5007
   :alt: Sample dataset visualization

👨‍💻 Authors
-----------

- **Sohaib Daoudi**
  - Email: `soh.daoudi@gmail.com <mailto:soh.daoudi@gmail.com>`_
  - GitHub: `@sohaibdaoudi <https://github.com/sohaibdaoudi>`_

- **Marouane Majidi**
  - Email: `majidi.marouane0@gmail.com <mailto:majidi.marouane0@gmail.com>`_
  - GitHub: `@marouanemajidi <https://github.com/marouanemajidi>`_

📜 License
----------

This project is licensed under the `MIT License <https://opensource.org/licenses/MIT>`_. Full text available in the repository's LICENSE file.

---

.. centered:: *Safety first! Always consult professional mechanics for complex automotive procedures.*
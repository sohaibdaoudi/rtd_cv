IV- Voice Assistant
===================

Following the initial object detection phase, the Voice Assistant represents the interactive, core guidance system of the project. Once the prerequisite tools have been identified, this module takes over to provide step-by-step spoken instructions to the user. This is achieved not by generating new text, but by accurately classifying the user's spoken questions into predefined categories (intents) and delivering a corresponding, pre-written answer. This chapter details the complete workflow for developing the voice assistant, from the strategic pivot in model selection to the intricacies of the training data, the final model architecture, and the real-time application that brings it all to life.

4.1 Intent Classification Model
-------------------------------

The "brain" of the voice assistant is the intent classification model, a specialized neural network trained to understand the user's goal based on their question.

4.1.1 Project Evolution: From Generation to Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The initial goal for this project was to build a sequence-to-sequence (seq2seq) model. A seq2seq model is a more advanced type of neural network that learns to generate answers from scratch, word by word, based on the user's question. This would allow for more dynamic and flexible responses.

However, early experiments with our generated faq.json data showed that this approach did not perform well. Seq2seq models require a very large and varied amount of training data to learn the nuances of language and generate coherent, accurate sentences. The existing dataset, while excellent for classification, was not large enough for this complex generative task.

Due to these data limitations, the project pivoted to a more robust and reliable method: *Intent Classification*. Instead of generating answers, the model now acts as a smart classifier. Its job is to understand the user's question and classify it into a predefined intent. Once the intent is identified, the system simply looks up the corresponding pre-written answer. This approach is highly effective and guarantees that the user receives an accurate, well-formulated response.

To further improve the classification model's understanding of language, we integrated pre-trained *GloVe* word embeddings. These provide the model with a foundational knowledge of words and their relationships, which is crucial for achieving high accuracy without a massive custom dataset. The future goal remains to create or acquire a larger dataset to eventually revisit the more ambitious seq2seq approach.

4.1.2 Dataset: faq.json
~~~~~~~~~~~~~~~~~~~~~~~

The foundation of the intent classification model is the faq.json file. This file contains all the knowledge the assistant needs to map questions to answers.

4.1.2.1 File Structure
^^^^^^^^^^^^^^^^^^^^^^

The faq.json file is a list of JSON objects. Each object represents a single piece of training information and contains four key fields:

* *id*: A unique identifier for the entry.
* *question*: An example question that a user might ask. The model learns from these examples.
* *intent*: The category or "topic" of the question. This is the label the model learns to predict.
* *answer*: The canned response the assistant provides for that specific intent.

An example object from the file:

.. code-block:: json

   {
     "id": 2,
     "question": "Where do I place the jack?",
     "intent": "Jack Placement",
     "answer": "Look for the jack point near the flat tire under the car frame; it's usually behind the front wheel or in front of the rear wheel."
   }

4.1.2.2 Intent Categories
^^^^^^^^^^^^^^^^^^^^^^^^^

The dataset is organized into the following specific intents, ensuring clear and distinct boundaries for the classification task:

* *Preparation*: Initial steps before starting the work.

* *Jack Placement*: Where to position the car jack.

* *Loosening Lug Nuts*: How to loosen nuts before lifting.

* *Order of Operations*: Clarifying the sequence of actions.

* *Lifting*: How to raise the car.

* *Removing Lug Nuts*: Taking the nuts and the flat tire off.

* *Mounting*: Putting the spare tire on.

* *Partial Tightening*: Tightening nuts while the car is raised.

* *Final Tightening*: The final, secure tightening process.

* *Cleanup*: What to do with tools and the flat tire afterwards.

* *Missing Tool*: Handling the absence of a required tool.

* *Stuck Lug Nut*: Dealing with a stubborn lug nut.

* *Safety Check*: Ensuring the new tire is secure.

* *Spare Tire Location*: Finding the spare tire.

* *Tool List*: Listing all necessary tools.

* *Unknown*: A catch-all for unrelated questions.

4.1.3 Model Architecture and Training (train.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The training script (train.py) is responsible for building and training the neural network.

The model is a Sequential stack of layers:
1.  *Embedding Layer*: This layer converts words into numerical vectors. It is initialized with GloVe weights to leverage pre-existing language knowledge, and its weights are frozen (trainable=False) so this knowledge is not lost during training.
2.  *Bidirectional LSTM Layers*: Two layers of Bidirectional LSTMs form the core of the model. They process the sentence forwards and backwards to understand the full context of the words in the question.
3.  *Dropout Layers*: These layers randomly ignore some neurons during training to prevent the model from simply memorizing the training data (overfitting).
4.  *Dense Layer*: The final layer makes the classification decision, outputting a probability score for each possible intent.

The training process involves feeding the model the preprocessed questions and their corresponding intent labels from faq.json. The model then adjusts its parameters over multiple passes (epochs) to minimize its prediction errors, ultimately learning to map question patterns to the correct intents.

4.2 Real-Time Voice Assistant Application (main.py)
-----------------------------------------------------

The main.py script runs the live voice assistant, integrating all the necessary components to create an interactive experience.

4.2.1 System Components
~~~~~~~~~~~~~~~~~~~~~~~

The application relies on several key libraries working in tandem:

* *Vosk*: A lightweight, offline speech recognition library used to convert the user's spoken words into text.
* *TensorFlow/Keras*: Used to load our pre-trained intent classification model and make predictions on the text from Vosk.
* *pyttsx3*: A text-to-speech (TTS) library that converts the assistant's text answers into spoken audio.
* *Sounddevice*: Manages the audio input from the microphone.

4.2.2 Operational Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~

The assistant operates in a continuous loop with the following steps:

1.  *Listen*: The script uses sounddevice to capture audio from the microphone.

2.  *Recognize Speech (Speech-to-Text)*: The captured audio is streamed to the Vosk engine, which processes it and returns the recognized text.

3.  *Predict Intent*: The recognized text is passed to our trained TensorFlow model, which predicts the user's intent.

4.  *Retrieve Answer*: The script uses the predicted intent as a key to look up the correct, pre-written answer from the faq.json data.

5.  *Speak (Text-to-Speech)*: The retrieved answer text is sent to the pyttsx3 engine, which vocalizes the response to the user.

6.  *Repeat*: The assistant immediately returns to listening for the next command.
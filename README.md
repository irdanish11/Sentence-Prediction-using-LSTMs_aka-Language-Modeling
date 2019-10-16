# Sentence Prediction using LSTMs aka Language Modeling
LSTM text generation by word. Used to generate multiple sentence suggestions based on the input words or a sentence.
For more information about the project details, see [this blog post](https://medium.com/towards-artificial-intelligence/sentence-prediction-using-word-level-lstm-text-generator-language-modeling-using-rnn-a80c4cda5b40) associated with this project.

## Responses: Generating 1 senetnce only i.e generating the described number of words
```
Input: hydrant requires repair
Output: hydrant requires repair is not working

Input: describe the problem
Output: describe the problem please attend to

Input: door and window
Output: door and window in the kitchen is not working in the

Input: machine is leaking
Output: machine is leaking and needs to be replaced

Input: modus to install
Output: modus to install and integrate not wifi
```
## Responses: Generating multiple senetnces as suggestions.
```
Input: please fix the
Output:
        please fix the door in the
        please fix the door is not
        please fix the issue and connections
        please fix the door is blocked
        please fix the door is broken

Input: mens bathroom door
Output:
        mens bathroom door is not working
        mens bathroom door is broken and
        mens bathroom door is broken in
        mens bathroom door is broken please

Input: please fix the light
Output:
        please fix the light bulb in the
        please fix the light out in the
        please fix the light out at judges
        please fix the light out at level
        please fix the light out in chambers

Input: tap in bathroom
Output:
        tap in bathroom is not working
        tap in bathroom is leaking and
        tap in bathroom is leaking in
        tap in bathroom is leaking at

Input: drainage pipe is leaking
Output:
        drainage pipe is leaking and needs to
        drainage pipe is leaking and needs replacing
        drainage pipe is leaking at female toilets
        drainage pipe is leaking and needs repairing
        drainage pipe is leaking in the mens
```
## Installation Dependencies

* Python 3.7
* Tensorflow 1.14

## How to Run?
To re-train the model, run the ```model.py``` file and fit the model. 
To test the model we have to situations either we can generate only described number of words or we can also generate multiple sentences as suggestions. To generate words run ```word_pred.py``` and to generate multiple sentences run ```beam_search_beta.py```.

## Acknowledgements
* This project is highly based on this [blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) 
* Additional Readings: 
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [A Brief Summary of Maths Behind RNN ](https://medium.com/towards-artificial-intelligence/a-brief-summary-of-maths-behind-rnn-recurrent-neural-networks-b71bbc183ff)
* [How many LSTM cells should I use?
](https://datascience.stackexchange.com/questions/16350/how-many-lstm-cells-should-i-use/18049)
* [What's the difference between a bidirectional LSTM and an LSTM?
](https://stackoverflow.com/questions/43035827/whats-the-difference-between-a-bidirectional-lstm-and-an-lstm)
* [An Introduction to Dropout for Regularizing Deep Neural Networks](https://medium.com/towards-artificial-intelligence/an-introduction-to-dropout-for-regularizing-deep-neural-networks-4e0826c10395)

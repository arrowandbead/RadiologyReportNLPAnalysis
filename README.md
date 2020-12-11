# CS1470 Final Project: RadiologyReportNLPAnalysis
Using MSRN to extract Oncological Outcomes from Medical Data. Using NLP to predict evidence of cancer & cancer status from radiological medical reports. Our goal is to build a model capable of describing a patient’s disease progression and response to therapy.
*****

 ## Building

To build this project:
~~~~python
pip install -r requirements.txt
source env/bin/activate
python model.py
~~~~

## Results

Our test Categorical Accuracy was 72% for our current seed, but could change depending on the distribution of examples between test-train. Our per-class accuracies were the following:
~~~python
Label 1: 74% (had 54 examples in test)
Label 2: 30% (had 10 examples in test)
Label 3: 94% (had 69 examples in test)
Label 4: 48% (had 27 examples in test)
Label 5: 35% (had 17 examples in test)
Label 6: 0% (had 5 examples in test)
Label 7: 0% (had 0 examples in test)
~~~
Our train accuracy was very similar to our test accuracy and was only different after about 3 decimal points.

# Known Bugs
We manually calculated accuracy, but this value diverged from the Categorical Accuracy given by Keras by about 10%. We decided to stick to the Keras measure and are satisfied with our per-class accuracies (which use our own accuracy function), as they align with the distributions of each class in our dateset.


# deep_learn_rec
A deep learning recommendation system built using TensorFlow. Primary method: an architecture called long short-term memory (LSTM).

The business problem: client wants a recommendation system based on customer order history. Imagine a rental company with construction equipment -- lots of products that tend to get ordered together. Variation in sizing means a wide array of products and the average order history has over three hundred items, so we're dealing with natural language processing of long sequences. This lends itself to deep learning, as these models are good at prediction and can handle unstructured sequential data (remember: given the problem and the data, the method presents itself). 

Data wrangling, cleaning, and transformations are the key parts of this script; model training happens toward the end. Primary inputs include: customer id and vertical. Customer id maps to a sequence of products, representing the customer's order history. LSTM's are particularly good at modeling sequential data, which is why this method works for a recommendation system. Script is equipped to handle GPU's and can run on cloud computing platforms like DataBricks or AWS. Future iterations of the model might include attention mechanisms.

Useful resources:
https://en.wikipedia.org/wiki/Long_short-term_memory

https://medium.com/decathlondigital/building-a-rnn-recommendation-engine-with-tensorflow-505644aa9ff3

https://medium.com/decathlondigital/personalization-strategy-and-recommendation-systems-at-d%C3%A9cathlon-canada-d9cb3d37f675

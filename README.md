# ChatBot-Medical

This chatbot is built entirely from scratch and, at its core, it responds to a query by understanding what the user is trying to ask, and then answering by choosing the best response for it.

This can be a particularly useful tool for people looking for a suggestion for delivering first aid of any kind. 
The chatbot uses natural language processing to understand the query of an individual who may be panicking and wanting the procedure for first aid for problems ranging from minor cuts, abrasions and burns to gastrointestinal problems.

The functionality of this particular chatbot can be extended in the future by including audio support for the user. They will be able to speak the query and get an answer from the bot on what's the best procedure to do.

The requirements.txt file contains all the required packages which will be used by the script. Just run `pip install -r requirements.txt` to install the dependencies.

The _**main_medical_intents.ipynb**_ script is written in such a way that it will try to make sense of your particular query, then assign a tag to it like _'cuts'_ or _'abrasions'_. Then it will select the 
appropriate response for that particular tag. The tags and these responses are stored in the intents_med.json file in the root directory. The remaining data and the Deep Neural Network
model information is stored in the modelDataMed folder. This data need only be calculated once and can be used again and again.

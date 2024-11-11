import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import re
import random

######################################################
##########Créer un chatroom fake avec des variables aléatoires basées 
#######sur la liste de pairs de devises créées

# Paramètres de génération pour les données
currencies = [
    "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD", "USD/CAD",
    "USD/CHF", "NZD/USD", "EUR/JPY", "EUR/GBP", "EUR/CHF",
    "GBP/JPY", "AUD/JPY", "AUD/NZD", "USD/CNH", "EUR/AUD",
    "EUR/CAD", "CAD/JPY", "CHF/JPY", "GBP/CAD", "GBP/AUD",
    "NZD/JPY", "GBP/NZD", "EUR/NZD", "AUD/CAD", "USD/SGD",
    "USD/HKD", "EUR/SGD", "USD/KRW", "USD/INR", "USD/TRY",
    "USD/BRL", "USD/MXN", "USD/ZAR", "USD/PLN", "USD/RUB",
    "USD/DKK", "USD/NOK", "USD/SEK", "EUR/NOK", "EUR/SEK",
    "EUR/DKK", "USD/THB", "USD/TWD", "USD/CZK", "USD/HUF",
    "USD/ILS", "USD/SAR", "USD/AED", "USD/CLP", "USD/PHP",
    "YEN", "EUR", "GBP", "USD","¥"]


risk_types = ["RR10", "RR25", "RR", "FLY", "BFLY","buterfly"]
deltas = ["D10", "D25", "d10", "d25"]
maturities = ["1W", "2W", "1M", "3M", "6M", "1Y","2Y","3Y","4Y", 
             "5Y","7Y", "10Y","12Y","15Y","20Y","25Y","30Y","2WKS",
             "2WK","1WK"]

# Générer la liste de données
data = []
for _ in range(1000):
    currency = random.choice(currencies)
    risk_type = random.choice(risk_types)
    delta = random.choice(deltas)
    maturity = random.choice(maturities)
    bid = round(random.uniform(5.0, 10.0), 1)
    ask = round(bid + random.uniform(0.5, 1.0), 1)
    
    text = f"{currency} {risk_type} {delta} {maturity} {bid}/{ask}"
    
    entry = {
        "text": text,
        "Currency": currency,
        "RiskType": risk_type,
        "Delta": delta,
        "Maturity": maturity,
        "Bid": str(bid),
        "Ask": str(ask)
    }
    
    if entry['RiskType'][-2:] in ["10","25"]:
          entry['Delta']=""
          entry['text'] = f"{currency} {risk_type} {maturity} {bid}/{ask}"
    data.append(entry)

# Affichage d'un extrait des données
all_chat=pd.DataFrame(data)
all_chat.head()

########################################################

# Charger le modèle de base de spaCy pour l'anglais (ou français si besoin)
nlp = spacy.blank("en")  # Utilise "fr" pour le français

# Définir un pipeline NER
ner = nlp.add_pipe("ner")

lines=all_chat['text']
# Regex pour capturer chaque composant dans le format attendu
pattern = re.compile(
    r"(?P<Currency>[A-Z]{3,6}|¥)\s+"\
    r"(?P<RiskType>(?:[A-Za-z]*fly*[A-Za-z]|RR|ATM))\s*"\
    r"(?P<Delta>(?!ATM)([Dd]?\d{1,2}))?\s*"\
    r"(?P<Maturity>\d{1,2}[YMWKS]{1,3})\s+"\
    r"(?P<Bid>\d+\.\d+)\s*/\s*(?P<Ask>\d+\.\d+)"
)

# Fonction pour extraire les entités du texte
def extract_entities(line):
    match = pattern.search(line)
    if match:
        entities = []
        for group_name in ["Currency", "RiskType", "Delta", "Maturity", "Bid", "Ask"]:
            start, end = match.span(group_name)
            entities.append((start, end, group_name.upper()))
        return line, {"entities": entities}
    return None

# Ajouter des labels dans le NER
labels = ["Currency", "RiskType", "Delta", "Maturity", "Bid", "Ask"]
for label in labels:
    ner.add_label(label)

# Préparer les données d'entraînement
results = [extract_entities(line) for line in lines]
train_data = [item for item in results if item is not None]

# Convertir les exemples en format spaCy
db = DocBin()
for text, annotations in train_data:
    doc = nlp.make_doc(text)
    entities = annotations["entities"]
    spans = [doc.char_span(start, end, label=label) for start, end, label in entities]
    doc.ents = [span for span in spans if span is not None]
    db.add(doc)

# Entraîner le modèle
optimizer = nlp.initialize()
for i in range(100):
    for text, annotations in train_data:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], drop=0.3, sgd=optimizer)

####################################################################
##############Test du model #######################################
################################################################



# ... (your existing code)

# ... (rest of your existing code)

# The code trains a spaCy NER (Named Entity Recognition) model to identify specific components (Currency, RiskType, Delta, Maturity, Bid, Ask) within financial strings.

# Here's a breakdown of the key parts and potential improvements:

# 1. Data Preparation:
#   - The input data consists of strings representing financial instruments with their corresponding annotations.
#   - The regular expression `pattern` is crucial. It defines the structure you expect in the input strings.  The original pattern was improved to capture more variants of "fly" and allow for optional delta values.  This regex is essential to correctly extracting entities and providing training data.
#   - The `extract_entities` function uses this regular expression to find the components in the input strings and extract their starting and ending positions. This information is then used to create the training data for spaCy.
#   - The crucial improvement is the regex; a revised regex now accounts for optional Delta values and more "fly"-related variants.

# 2. Model Initialization:
#   - `spacy.blank("en")` creates a blank spaCy model. You might consider using a pre-trained model ("en_core_web_sm" or similar) as a base, as it could improve initial performance.  This would require changing how you add labels (see below).
#   - `ner = nlp.add_pipe("ner")` adds a named entity recognition component to the pipeline.  Using a pre-trained model will require a different approach to add labels.
#   - `ner.add_label(...)`: crucial to define the categories or entity types for the model to learn (labels).  If you use a pre-trained model, you will only add labels which aren't present in the pre-trained model.

# 3. Training:
#   - `db = DocBin()`: Creates a DocBin to store training examples in a format spaCy can use.
#   - The code iterates through the training data, creates spaCy `Doc` objects, annotates them with the identified entities (using `doc.ents`), and adds them to the DocBin.
#   - `nlp.initialize()`: initializes the model's optimizer for training.
#   - The training loop iterates over the examples and updates the model using `nlp.update()`.   The key parameters here are `drop` (dropout rate for regularization) and `sgd` (the optimizer).  You may need to adjust the number of iterations (100) and learning rate for optimal performance.

# 4. Testing and Evaluation:
#   - The code demonstrates how to process a new text (`test_text`) using the trained model and print the recognized entities.
#   - **Missing Evaluation:** A crucial aspect missing is a proper evaluation. You should split your data into training and testing sets (which you have done using train_test_split earlier) and use metrics like precision, recall, and F1-score to evaluate the model's performance.  You are using classification report in the first part, it is not a bad idea to add it here as well.
#   - **Overfitting:** Because the training set is small, the model could easily overfit.  If you use all your data for training and do not split your data between test and train sets, then you should expect this.  Increase the size of your data and then try it again.

# Key Improvements and Considerations:

# - Use a pre-trained model: Start with a pre-trained spaCy model for better performance and avoid overfitting.
# - Increase training data: The model needs significantly more annotated data to generalize well to new examples.   It would be better if you train on thousand of sentences rather than a small amount.
# - Hyperparameter Tuning: Experiment with different hyperparameters (learning rate, number of iterations, dropout rate) to optimize performance.
# - Proper evaluation: Use a held-out test set and appropriate metrics (precision, recall, F1-score) to properly assess model performance.
# - More sophisticated regular expressions: Make sure your regex is powerful enough to handle edge cases and different variations in input.

# Example of how to use a pre-trained model (and a different label addition method):

# import spacy
# nlp = spacy.load("en_core_web_sm") # Load a pretrained model
# ner = nlp.get_pipe("ner")
# for label in labels:
#     if label not in ner.labels:
#         ner.add_label(label)

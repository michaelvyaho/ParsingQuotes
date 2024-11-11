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

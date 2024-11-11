import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import re
import random
import pandas as pd

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
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURJPY", "EURGBP", "EURCHF",
    "GBPJPY", "AUDJPY", "AUDNZD", "USDCNH", "EURAUD",
    "EURCAD", "CADJPY", "CHFJPY", "GBPCAD", "GBPAUD",
    "NZDJPY", "GBPNZD", "EURNZD", "AUDCAD", "USDSGD",
    "USDHKD", "EURSGD", "USDKRW", "USDINR", "USDTRY",
    "USDBRL", "USDMXN", "USDZAR", "USDPLN", "USDRUB",
    "USDDKK", "USDNOK", "USDSEK", "EURNOK", "EURSEK",
    "EURDKK", "USDTHB", "USDTWD", "USDCZK", "USDHUF",
    "USDILS", "USDSAR", "USDAED", "USDCLP", "USDPHP"]


risk_types = ["RR10", "RR25", "RR", "FLY", "BFLY","buterfly"]
deltas = ["D10", "D25", "d10", "d25"]
maturities = ["1W", "2W", "1M", "3M", "6M", "1Y","2Y","3Y","4Y", 
             "5Y","7Y", "10Y","12Y","15Y","20Y","25Y","30Y","2WKS",
             "2WK","1WK"]

# Générer la liste de données
data = []
for _ in range(50):
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
print(all_chat.head())

########################################################

# Charger le modèle de base de spaCy pour l'anglais (ou français si besoin)
nlp = spacy.blank("en")  # Utilise "fr" pour le français

# Définir un pipeline NER
ner = nlp.add_pipe("ner")

# Charger le modèle de base de spaCy pour le français
nlp = spacy.blank("en")  # ou "fr" si tu veux le français

# Définir un pipeline NER
ner = nlp.add_pipe("ner")

# Liste des chaînes d'entrée
lines=all_chat['text']

# Regex pour capturer chaque composant dans le format attendu
pattern = re.compile(
    r"(?P<Currency>[A-Z]{3,6})\s+"  # Capture la devise (3 à 6 lettres, ex : EUR, EURUSD)
    r"(?P<RiskType>[a-zA-Z]+[0-9]*)\s+"  # Capture le type de risque, ex : RR10, RR, ATM, FLY
    r"(?P<Maturity>\d+[YMWS]{1,2})\s+"  # Capture la maturité (ex: 1Y, 6M, 3W, 2WS)
    r"(?P<Bid>[0-9.]+)\s*/\s*(?P<Ask>[0-9.]+)"  # Capture Bid/Ask avec gestion des espaces autour de "/"
)

# Regex ajustée pour capturer toutes les variantes de "fly" (commençant ou finissant par fly)
pattern=re.compile(
          r'\s*(?P<Currency>[A-Z]{3,6}|¥|[A-Z]{3}/[A-Z]{3})\s+' \
          r'(?P<RiskType>(?:[A-Za-z]*fly*[A-Za-z]|RR|ATM))\s*' \
          r'(?P<Delta>(?!ATM)([Dd]?\d{1,2}))?\s*' \
          r'(?P<Maturity>\d{1,2}[YMWKS]{1,3})\s+' \
          r'\s*(?P<Bid>\d+\.\d+)\s*/\s*(?P<Ask>\d+\.\d+)\s*')

# Fonction pour identifier la position des resultats dans le texte à partir des regex, servira a entrainer le modele
def extract_entities(line):
    match = pattern.search(line)
    if match:
        entities = []
        # Obtenir les groupes capturés et leurs positions dans le texte
        for group_name in ["Currency", "RiskType","Delta", "Maturity", "Bid", "Ask"]:
            start, end = match.span(group_name)
            entities.append((start, end, group_name.upper()))  # Convertir le nom de l'entité en majuscule

        #return { "text": line, "entities": entities }
        return line, {"entities": entities }
    return None


# Définir les labels des entités que nous souhaitons reconnaître
labels = ["Currency", "RiskType","Delta", "Maturity", "Bid", "Ask"]
for label in labels:
    ner.add_label(label)
#[ner.add_label(label) for label in labels]

# Appliquer la fonction sur chaque ligne et afficher les résultats
# Créer quelques exemples annotés pour l'entraînement
results = [extract_entities(line) for line in lines]
train_data=[item for item in results if item is not None]
print(train_data)
# train_data = [
#     ("EURUSD RR10 1Y 8.1/9.1", {"entities": [(0, 6, "CURRENCY"), (7, 12, "RISKTYPE"), (13, 15, "MATURITY"), (16, 19, "BID"), (20, 23, "ASK")]}),
#     ("GBPUSD RR25 6M 7.5/8.5", {"entities": [(0, 6, "CURRENCY"), (7, 12, "RISKTYPE"), (13, 15, "MATURITY"), (16, 19, "BID"), (20, 23, "ASK")]}),
#     ("USDJPY RR5 3Y 6.3/7.0", {"entities": [(0, 6, "CURRENCY"), (7, 10, "RISKTYPE"), (11, 13, "MATURITY"), (14, 17, "BID"), (18, 21, "ASK")]}),
# ]


# Convertir les exemples au format spaCy
db = DocBin()
for text, annotations in train_data:
    doc = nlp.make_doc(text)
    entities = annotations["entities"]
    spans = [doc.char_span(start, end, label=label) for start, end, label in entities]
    doc.ents = [span for span in spans if span is not None]
    db.add(doc)

# Entraîner le modèle
optimizer = nlp.initialize()
for i in range(100):  # Boucle d'entraînement avec 100 itérations
    for text, annotations in train_data:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        #nlp.update([example], drop=0.3, sgd=optimizer)
        nlp.update([example], drop=0.3,sgd=optimizer)

# Tester le modèle sur une nouvelle phrase et obtenir les scores de probabilité
test_text = "EURUSD bf d10 2Wk 8.1/9.1"
doc = nlp(test_text)

# Afficher les entités détectées avec leurs scores de probabilité
for ent in doc.ents:
    print(f"Texte: {ent.text}, Label: {ent.label_}, Position: ({ent.start_char}, {ent.end_char}), "
          f"Score de probabilité: {ent.score if hasattr(ent, 'score') else 'N/A'}")

# Affichage d'un résumé pour chaque entité détectée
print("\nRésumé des prédictions:")
for ent in doc.ents:
    label_verification = ent.label_ in labels
    print(f"- Entité: {ent.text} | Label: {ent.label_} | Vérification: {'Valide' if label_verification else 'Invalide'}")

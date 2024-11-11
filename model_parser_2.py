import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
for _ in range(1000000):
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
# Données d'entraînement - avec annotations pour chaque entité

# Regex pour capturer chaque composant dans le format attendu
pattern = re.compile(
    r"(?P<Currency>[A-Z]{3,6}|¥)\s+"\
    r"(?P<RiskType>(?:[A-Za-z]*fly*[A-Za-z]|RR|ATM))\s*"\
    r"(?P<Delta>(?!ATM)([Dd]?\d{1,2}))?\s*"\
    r"(?P<Maturity>\d{1,2}[YMWKS]{1,3})\s+"\
    r"(?P<Bid>\d+\.\d+)\s*/\s*(?P<Ask>\d+\.\d+)")
    
# recuperer le vrai chat et appliquer les regex puis 
# le transformer en dataframe cols=["text",
                                   # "Currency", 
                                   # "RiskType", 
                                   # "Delta", 
                                   # "Maturity", 
                                   # "Bid", "Ask"]
                                   
#Filtre sur les stratégies vega neutres?


FakeChat_Transform_With_Regex = [
    {"text": "EURUSD RR10 1Y 8.1/9.1", "Currency": "EURUSD", "RiskType": "RR", "Delta": "10", "Maturity": "1Y", "Bid": "8.1", "Ask": "9.1"},
    {"text": "GBPUSD RR25 1M 7.5/8.5", "Currency": "GBPUSD", "RiskType": "RR", "Delta": "25", "Maturity": "1M", "Bid": "7.5", "Ask": "8.5"},
    {"text": "YEN RR D25 2W 7.5/8.5", "Currency": "YEN", "RiskType": "RR", "Delta": "D25", "Maturity": "2W", "Bid": "7.5", "Ask": "8.5"},
    {"text": "EUR FLY D10 6M 7.5/8.5", "Currency": "EUR", "RiskType": "BF", "Delta": "D10", "Maturity": "6M", "Bid": "7.5", "Ask": "8.5"},
    {"text": "USDJPY BFLY d10 10Y 7.5/8.5", "Currency": "USDJPY", "RiskType": "BF", "Delta": "d10", "Maturity": "10Y", "Bid": "7.5", "Ask": "8.5"},
    {"text": "GBPUSD buterfly d10 2WKS 7.5/8.5", "Currency": "GBPUSD", "RiskType": "BF", "Delta": "d10", "Maturity": "2WKS", "Bid": "7.5", "Ask": "8.5"},
    {"text": "EURUSD ATM 1Y 8.5/9.1", "Currency": "EURUSD", "RiskType": "ATM", "Delta": "", "Maturity": "1Y", "Bid": "8.5", "Ask": "9.1"}
]
FakeChat_Label=entry
data=FakeChat_Label
# Transformation en DataFrame
df = pd.DataFrame(data)

# Données d'entrée et cibles
X = df["text"]
y = df[["Currency", "RiskType", "Delta", "Maturity", "Bid", "Ask"]]

# Diviser les données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Pipeline pour vectoriser le texte et entraîner un modèle pour chaque entité
models = {}
vectorizer = TfidfVectorizer()

# Entraînement et évaluation du modèle pour chaque cible
for target in y.columns:
    #print(f"\nTraining model for {target}...")

    # Pipeline spécifique pour chaque entité cible
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),           # Convertir le texte en vecteurs numériques
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))  # Modèle de classification
    ])

    # Entraînement du modèle sur la colonne cible
    #print(f'X_train: {X_train}')
    #print(f'y_train: {y_train[target]}')

    model = pipeline.fit(X_train, y_train[target])
    models[target] = model

    # Prédictions et évaluation
    y_pred = model.predict(X_test)
    print(f"Classification Report for {target}:\n", classification_report(y_test[target], y_pred))

# Prédictions avec niveaux de confiance sur le set de test
results = []
for text in X_test:
    print(text)
    result = {'text': text}
    confidences = []
    for target, model in models.items():
        # Prédiction et probabilité
        pred = model.predict([text])[0]
        prob = max(model.predict_proba([text])[0])  # Niveau de confiance pour la classe prédite
        result[target] = pred
        confidences.append(prob)
    result['Confidence'] = sum(confidences) / len(confidences)  # Moyenne des confiances
    results.append(result)

# Affichage des résultats sous forme de tableau
results_df = pd.DataFrame(results)
print("\nRésultats de prédiction avec niveau de confiance :\n", results_df)

##############################################################
################tester le modele ########################
#########################################################


# Fonction de test du modèle de classification sur un message
def test_message(message, models):
    # Initialisation des résultats
    result = {'text': message}
    confidences = []
    
    # Prédiction pour chaque entité cible
    for target, model in models.items():
        # Prédire la classe pour le message donné
        prediction = model.predict([message])[0]
        
        # Calculer le niveau de confiance de la prédiction
        probabilities = model.predict_proba([message])[0]
        confidence = max(probabilities)
        
        # Stocker les résultats
        result[target] = prediction
        result[f"{target}_confidence"] = confidence
        confidences.append(confidence)
    
    # Niveau de confiance global (moyenne des confiances pour chaque entité)
    result['overall_confidence'] = sum(confidences) / len(confidences)
    return result
    
######################################################################
# Exemple d'appel de la fonction sur un message test
# Remarque : models est le dictionnaire contenant les modèles entraînés pour chaque entité
messages = ["EURUSD RR10 1Y 8.1/9.1",
           "USDJPY RR10 2Y 8.1/10.1"]
######################################################################

result_df=pd.DataFrame()
for message in messages:      
      prediction_result = [test_message(message, models) ]
      # Conversion en DataFrame pour affichage clair
      result_df =pd.concat([result_df, pd.DataFrame(prediction_result)])

cols_1=['text',"Currency", "RiskType", "Delta", "Maturity", "Bid", "Ask"]
cols_2=[col+"_confidence" for col in cols_1 if col!='text']
cols_all=cols_1+cols_2
result_df[cols_all]

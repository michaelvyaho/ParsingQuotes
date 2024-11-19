import pandas as pd
import re



class MessageParser:
    def __init__(self, regex_list):
        """
        Initialise le parser avec une liste d'expressions régulières.

        :param regex_list: Liste d'expressions régulières à utiliser pour filtrer les messages.
        """
        self.regex_list = regex_list

    def parse_structured_message(self, message):
        """
        Parse un message structuré contenant des sections séparées par //n.

        :param message: Un message structuré sous forme de texte.
        :return: Un DataFrame avec les données organisées.
        """
        # Remplace les séparateurs pour créer des lignes exploitables
        message = message.replace("//n", "\n")
        lines = message.split("\n")
        data = []
        current_currency = None
        current_risk_type = None
        current_delta = None

        for line in lines:
            # Vérifier si la ligne est un header
            if re.match(r"[A-Z]+[A-Z]+ .*", line):  # Paires de devises suivies d'un texte
                header = line.strip()
                # Séparer la paire de devises et le type de risque
                header_match = re.match(r"([A-Z]{6})\s+(.*)", header)
                if header_match:
                    current_currency = header_match.group(1)  # Exemple : "AUDEUR"
                    current_risk_type = header_match.group(2)  # Exemple : "ATM" ou "RR d10"
                    # Extraire le delta s'il est présent
                    delta_match = re.search(r"d(\d+)", current_risk_type)
                    current_delta = int(delta_match.group(1)) if delta_match else None
            else:
                # Match pour les lignes de données
                match = re.match(r"(\d+[WMY]) (\d+\.\d+)/(\d+\.\d+)", line)
                if match and current_currency and current_risk_type:
                    duration, value1, value2 = match.groups()
                    data.append({
                        "Currency": current_currency,
                        "Risk Type": current_risk_type,
                        "Duration": duration,
                        "Value 1": float(value1),
                        "Value 2": float(value2),
                        "Delta": current_delta
                    })

        # Retourne un DataFrame avec les données extraites
        return pd.DataFrame(data)

    def parse_messages(self, df):
        """
        Parse les messages d'un DataFrame et extrait ceux qui correspondent aux regex ou aux messages structurés.

        :param df: DataFrame avec des colonnes 'Message' et 'dateheure'.
        :return: Un DataFrame combiné avec les colonnes 'dateheure', 'Message' et les données structurées.
        """
        if 'Message' not in df.columns or 'dateheure' not in df.columns:
            raise ValueError("Le DataFrame doit contenir les colonnes 'Message' et 'dateheure'.")

        combined_data = []
        for _, row in df.iterrows():
            message = row['Message']
            dateheure = row['dateheure']

            if "//n" in message:  # Si le message est structuré
                structured_df = self.parse_structured_message(message)
                structured_df['dateheure'] = dateheure  # Ajouter la date à chaque ligne structurée
                combined_data.append(structured_df)
            else:
                for regex in self.regex_list:
                    if re.search(regex, message):
                        combined_data.append(pd.DataFrame([{
                            "dateheure": dateheure,
                            "Message": message
                        }]))
                        break

        # Combiner toutes les données en un seul DataFrame
        if combined_data:
            return pd.concat(combined_data, ignore_index=True)
        else:
            return pd.DataFrame(columns=['dateheure', 'Message'])



class MessageParser:
    def __init__(self, regex_list):
        """
        Initialise le parser avec une liste d'expressions régulières.

        :param regex_list: Liste d'expressions régulières à utiliser pour filtrer les messages.
        """
        self.regex_list = regex_list
    
    def parse_messages(self, df):
        """
        Parse les messages d'un DataFrame et extrait ceux qui correspondent aux regex.

        :param df: DataFrame avec des colonnes 'Message' et 'dateheure'.
        :return: Un DataFrame filtré avec les colonnes 'dateheure' et 'Message'.
        """
        if 'Message' not in df.columns or 'dateheure' not in df.columns:
            raise ValueError("Le DataFrame doit contenir les colonnes 'Message' et 'dateheure'.")
        
        # Filtrer les messages correspondant aux regex
        parsed_data = []
        for _, row in df.iterrows():
            message = row['Message']
            dateheure = row['dateheure']
            for regex in self.regex_list:
                if re.search(regex, message):
                    parsed_data.append({'dateheure': dateheure, 'Message': message})
                    break  # Éviter les doublons si plusieurs regex matchent
        
        # Créer un DataFrame global avec les messages filtrés
        return pd.DataFrame(parsed_data)

    import pandas as pd

# Fonction pour parser la phrase
def parse_to_dataframe(phrase):
   ''' # La phrase source
    phrase = """AUDEUR ATM
    1W 2.1/5.1
    1W 2.1/5.1
    1M 2.1/5.1
    1y 2.5/5.9"""
    '''
    lines = phrase.split("\n")
    header = lines[0]  # La première ligne contient AUDEUR ATM
    data = []
    
    # Parcourir les lignes restantes
    for line in lines[1:]:
        match = re.match(r"(\d+[WMY]) (\d+\.\d+)/(\d+\.\d+)", line)
        if match:
            duration, value1, value2 = match.groups()
            data.append({"Currency Pair": header, 
                         "Maturity": duration,
                         "Bid ": float(value1),
                         "Ask": float(value2)})
    
    # Créer un DataFrame
    return pd.DataFrame(data)

# Utilisation de la fonction
df = parse_to_dataframe(phrase)
print(df)


data = {
    'dateheure': ['2024-11-18 10:00:00', '2024-11-18 11:00:00', '2024-11-18 12:00:00'],
    'Message': ['Erreur critique détectée', 'Tout fonctionne bien', 'Problème de connexion']
}
df = pd.DataFrame(data)

regex_list = [r'erreur', r'problème', r'détectée']

parser = MessageParser(regex_list)
parsed_df = parser.parse_messages(df)
print(parsed_df)



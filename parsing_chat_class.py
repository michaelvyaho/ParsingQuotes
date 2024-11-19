import pandas as pd
import re

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

data = {
    'dateheure': ['2024-11-18 10:00:00', '2024-11-18 11:00:00', '2024-11-18 12:00:00'],
    'Message': ['Erreur critique détectée', 'Tout fonctionne bien', 'Problème de connexion']
}
df = pd.DataFrame(data)

regex_list = [r'erreur', r'problème', r'détectée']

parser = MessageParser(regex_list)
parsed_df = parser.parse_messages(df)
print(parsed_df)



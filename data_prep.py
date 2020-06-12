import pandas as pd


class DataPrep:
    def __init__(self):
        self.trait_cat_dict = {
            'O': 'cOPN',
            'C': 'cCON',
            'E': 'cEXT',
            'A': 'cAGR',
            'N': 'cNEU',
            'OPN': 'cOPN',
            'CON': 'cCON',
            'EXT': 'cEXT',
            'AGR': 'cAGR',
            'NEU': 'cNEU',
            'Openness': 'cOPN',
            'Conscientiousness': 'cCON',
            'Extraversion': 'cEXT',
            'Agreeableness': 'cAGR',
            'Neuroticism': 'cNEU'
        }
        self.trait_score_dict = {
            'O': 'sOPN',
            'C': 'sCON',
            'E': 'sEXT',
            'A': 'sAGR',
            'N': 'sNEU',
            'OPN': 'sOPN',
            'CON': 'sCON',
            'EXT': 'sEXT',
            'AGR': 'sAGR',
            'NEU': 'sNEU',
            'Openness': 'sOPN',
            'Conscientiousness': 'sCON',
            'Extraversion': 'sEXT',
            'Agreeableness': 'sAGR',
            'Neuroticism': 'sNEU'
        }

    def prep_data(self, trait, regression=False):
        df_status = self.prep_status_data()
        X = df_status['STATUS']
        if regression:
            y_column = self.trait_score_dict[trait]
        else:
            y_column = self.trait_cat_dict[trait]
        y = df_status[y_column]

        return X, y

    def prep_status_data(self):
        df = pd.read_csv('data/mypersonality_final.csv', encoding="ISO-8859-1")
        df = self.convert_traits_to_boolean(df)
        return df

    @staticmethod
    def convert_traits_to_boolean(df):
        trait_columns = ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU']
        d = {'y': True, 'n': False}
        for trait in trait_columns:
            df[trait] = df[trait].map(d)
        return df

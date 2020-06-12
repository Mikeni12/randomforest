import pickle

from model import Model


class Predictor:
    def __init__(self):
        self.traits = ['OPN', 'CON', 'EXT', 'AGR', 'NEU']
        self.models = {}
        self.load_models()

    def load_models(self):
        M = Model()
        for trait in self.traits:
            with open('static/' + trait + '_model.pkl', 'rb') as f:
                self.models[trait] = pickle.load(f)

    def predict(self, X, traits='All', predictions='All'):
        predictions = {}
        if traits == 'All':
            for trait in self.traits:
                pkl_model = self.models[trait]

                trait_scores = pkl_model.predict(X, regression=True).reshape(1, -1)
                predictions['pred_s' + trait] = trait_scores.flatten()[0]
                print(f'Puntuacion {trait}: {trait_scores} o {predictions["pred_s" + trait]}')

                trait_categories = pkl_model.predict(X, regression=False)
                predictions['pred_c' + trait] = str(trait_categories[0])
                print(f'Puntuacion {trait}: {trait_categories} o {predictions["pred_c" + trait]}')

                trait_categories_probs = pkl_model.predict_prob(X)
                predictions['pred_prob_c' + trait] = trait_categories_probs[:, 1][0]
                print(f'Puntuacion {trait}: {trait_categories_probs} o {predictions["pred_prob_c" + trait]}\n')

        return predictions

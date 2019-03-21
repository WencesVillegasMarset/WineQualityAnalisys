from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import os
from sklearn.externals import joblib
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder, LabelBinarizer

class ColumnSelector(BaseEstimator, TransformerMixin):
    '''
        Selector de columnas para obtener de un dataframe, devuelve los valores como ndarray
    '''
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns].values
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)

def generate_submission_id():
    '''
        Funcion que genera un identificador bajo el cual modelo, pipeline y data specs se van a serializar
    '''
    print(str(datetime.now()).replace(' ','')[0:15])

def serialize_model(obj, object_name, submission_id):
    '''
        Serializa un objeto en el directorio ./output/sumbission_id con el nombre object_name
    '''
    folder = os.path.join('.', 'output', submission_id)
    if not os.path.exists(folder):
        os.makedirs(folder)
    joblib.dump(obj, object_name)
    print('Model saved!')
    
    

# no es necesario por ahora
class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

if __name__ == "__main__":
    generate_submission_id()
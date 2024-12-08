from pathlib import Path
class Paths:
    basedir = Path.cwd()
    @classmethod
    def encoder(cls):
        '''Get the feature encoder file path'''
        path  = cls.basedir/'data'/'encoded_features'/'encoding.pkl'
        return path
    @classmethod
    def preprocessed(cls):
        '''Get the filepath for preprocessed data'''
        path  = cls.basedir/'data'/'preprocessed_data'/'preprocessed.csv'
        return path
    @classmethod
    def standardscaler(cls):
        '''Get file path for StandardScaler'''
        path  = cls.basedir/'models'/'standardscaler.pkl'
        return path

if __name__ == "__main__":
    # Sample usage
    print(Paths.encoder())

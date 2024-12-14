from pathlib import Path
class Paths:
    basedir = Path.cwd()
    @classmethod
    def encoder(cls):
        '''Get the feature encoder file path'''
        path  = cls.basedir/'data'/'encoded_features'/'encoding.pkl'
        return path
    @classmethod
    def ingested(cls):
        '''Get the filepath for ingested data from database'''
        path  = cls.basedir/'data'/'ingested_data'/'ingested.csv'
        return path
    @classmethod
    def preprocessed(cls):
        '''Get the filepath for preprocessed data for ML models'''
        path  = cls.basedir/'data'/'preprocessed_data'/'preprocessed.csv'
        return path
    @classmethod
    def standardscaler(cls):
        '''Get file path for StandardScaler for ML'''
        path  = cls.basedir/'models'/'standardscaler.pkl'
        return path
    
    @classmethod
    def sourcedbpath(cls):
        '''Get path local source db '''
        path  = cls.basedir/'database'/'source.db'
        return path
    

if __name__ == "__main__":
    # Sample usage
    print(Paths.encoder())

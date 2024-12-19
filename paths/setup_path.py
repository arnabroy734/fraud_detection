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
    def preprocessed_train(cls):
        '''Get the filepath for preprocessed training data for ML models'''
        path  = cls.basedir/'data'/'preprocessed_data'/'preprocessed_train.csv'
        return path
    @classmethod
    def preprocessed_test(cls):
        '''Get the filepath for preprocessed test data'''
        path  = cls.basedir/'data'/'preprocessed_data'/'preprocessed_test.csv'
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
    @classmethod
    def model_training_logs(cls):
        '''Get path model logs '''
        path  = cls.basedir/'log'/'model_logs.txt'
        return path
    @classmethod
    def model_deployment_logs(cls):
        '''Get path for model deployment logs '''
        path  = cls.basedir/'log'/'model_deployment_logs.txt'
        return path
    @classmethod
    def pipeline_logs(cls):
        '''Get path for logs of pipelines'''
        path  = cls.basedir/'log'/'pipeline_logs.txt'
        return path
    @classmethod
    def error_logs(cls):
        '''Get path for error logs'''
        path  = cls.basedir/'log'/'error_logs.txt'
        return path
    @classmethod
    def description_rf(cls):
        path  = cls.basedir/'log'/'rf_models.txt'
        return path
    @classmethod
    def description_ann(cls):
        path  = cls.basedir/'log'/'ann_models.txt'
        return path
    @classmethod
    def description_devnet(cls):
        path  = cls.basedir/'log'/'devnet_models.txt'
        return path
    @classmethod
    def model(cls, name, id):
        '''Get path for saved model '''
        path  = cls.basedir/'model_registry'/f'{name}_model_{id}.pkl'
        return path
    @classmethod
    def production_model(cls):
        '''Get path for model for deployement '''
        path  = cls.basedir/'models'/'production_model.pkl'
        return path
    

if __name__ == "__main__":
    # Sample usage
    print(Paths.encoder())

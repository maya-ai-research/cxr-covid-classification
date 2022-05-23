class Path(object):
    @staticmethod
    def get_path(dataset):
        if dataset == 'bimcv2':
            return '/Local/dataset/chest_xrays/bimcv2/images/'
        elif dataset == 'rafael':
            return '/Local/dataset/chest_xrays/rafael/images/' 
        elif dataset == 'covidx-cxr2':
            return '/Local/dataset/chest_xrays/covidx-cxr2/images/'
        elif dataset == 'u-3-filter':
            return '/Local/dataset/chest_xrays/u-3-filter/images/' 
        elif dataset == 'xcat_covid_siemens':
            return '/Local/dataset/chest_xrays/xcat-covid-siemens/images/' 
        elif dataset == 'xcat_covid_carestream':
            return '/Local/dataset/chest_xrays/xcat-covid-carestream/images/' 
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

    def get_epoch_no(dataset):
        #Returns the epoch no of the model trained with the respective dataset.
        #Epoch no is required as model is saved with the epoch no 
        #Epoch no returned is the best model when trained for 50 epochs 
        #To do early stopping for epoch 

        if dataset == 'bimcv2':
            return  21
        elif dataset == 'rafael':
            return 9
        elif dataset == 'covidx-cxr2':
            return 18
        elif dataset == 'u-3-filter':
            return 16 
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
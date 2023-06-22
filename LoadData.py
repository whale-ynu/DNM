import numpy as np
import os
import TopicModel as lda


class LoadData(object):
    '''given the path of data, return the data format for DNM
    :param path
    return:
    Train_data: a dictionary, 'Y' refers to a list of y values; 'X' refers to a list of features_M dimension vectors with 0 or 1 entries
    Test_data: same as Train_data
    '''

    # Three files are needed in the path
    def __init__(self, path, sub_dataset, wsdl_path=None, hidden_factor=50):
        self.path = path + sub_dataset
        self.trainfile = self.path + "train.libfm"
        self.testfile = self.path + "test.libfm"
        if wsdl_path is not None:
            self.tm = lda.TopicModel(wsdl_path)
            self.tm.factorize(hidden_factor, 0.5, 10000)
            self.features_M_wsdl = self.map_wsdl()
        self.features_M_user = self.map_features_user()
        self.features_M_service = self.map_features_service()
        self.Train_data, self.Test_data = self.construct_data()

    def map_wsdl(self):
        self.features_wsdl = set()
        self._features_wsdl = []
        self.read_features_wsdl(self.trainfile)
        self.read_features_wsdl(self.testfile)
        # print("features_wsdl:", len(self._features_wsdl))
        return self._features_wsdl
        
    def read_features_wsdl(self, file):  # read a feature file
        f = open(file)
        line = f.readline()
        while line:
            items = line.strip().split(' ')
            # wsdl_features = []
            if items[13] not in self.features_wsdl:
                #===============================================================
                # for dict in self.tm.getSortedDocumentTopics(self.tm.getDocumentTopics(str(items[13]))):
                #     wsdl_features.append(dict[0])
                # self._features_wsdl.append(wsdl_features)
                #===============================================================
                self._features_wsdl.append(self.tm.getDocumentTopics(str(items[13])))
                self.features_wsdl.add(items[13])
            line = f.readline()
        f.close()
        
    def map_features_user(self):  # map the feature entries in all files, kept in self.features dictionary
        self.features_user = {}
        self.read_features_user(self.trainfile)
        self.read_features_user(self.testfile)
        
        print("features_M_user:", len(self.features_user))
        return len(self.features_user)

    def read_features_user(self, file):  # read a feature file
        f = open(file)
        line = f.readline()
        i = len(self.features_user)
        while line:
            items = line.strip().split(' ')
            if items[2] not in self.features_user:
                self.features_user[ items[2] ] = i
                i = i + 1
            for item in items[4:8]:
                if item not in self.features_user:
                    self.features_user[ item ] = i
                    i = i + 1
            line = f.readline()
        f.close()
        
    def map_features_service(self):  # map the feature entries in all files, kept in self.features dictionary
        self.features_service = {}
        self.read_features_service(self.trainfile)
        self.read_features_service(self.testfile)
        
        print("features_M_service:", len(self.features_service))

        return len(self.features_service)

    def read_features_service(self, file):  # read a feature file
        f = open(file)
        line = f.readline()
        i = len(self.features_service)
        while line:
            items = line.strip().split(' ')
            if items[3] not in self.features_service:
                self.features_service[ items[3] ] = i
                i = i + 1
            for item in items[8:]:
                if item not in self.features_service:
                    self.features_service[ item ] = i
                    i = i + 1
            line = f.readline()
        f.close()
        
    def construct_data(self):
        X_U, X_I, Y1 , Y2 = self.read_data(self.trainfile)
        Train_data = self.construct_dataset(X_U, X_I, Y1, Y2)
        print("# of training:" , len(Y1))
        
        X_U, X_I, Y1 , Y2 = self.read_data(self.testfile)
        Test_data = self.construct_dataset(X_U, X_I, Y1, Y2)
        print("# of test:", len(Y1))

        return Train_data, Test_data

    def read_data(self, file):
        # read a data file. For a row, the first and second column go into Y1 and Y2;
        # the other columns become a row in X_U/X_I and entries are maped to indexs in self.features
        f = open(file)
        X_U = []
        X_I = []
        Y1 = []
        Y2 = []
        line = f.readline()
        while line:
            items = line.strip().split(' ')
            Y1.append(1.0 * float(items[0]))
            Y2.append(1.0 * float(items[1]))
            if float(items[0]) > 0:  # > 0 as 1; others as 0
                v = items[0]
            else:
                v = 0.0

            X_U.append([ self.features_user[items[2]],
                          self.features_user[items[4]],
                          self.features_user[items[5]],
                          self.features_user[items[6]],
                          self.features_user[items[7]]])
            
            X_I.append([ self.features_service[items[3]],
                         self.features_service[items[8]],
                         self.features_service[items[9]],
                         self.features_service[items[10]],
                         self.features_service[items[11]],
                         self.features_service[items[12]] ])
            line = f.readline()
        f.close()
        return X_U, X_I, Y1, Y2

    def construct_dataset(self, X_U, X_I, Y1, Y2):
        Data_Dic = {}
        X_U_lens = [len(line) for line in X_U]
        X_I_lens = [len(line) for line in X_I]
        
        indexs_U = np.argsort(X_U_lens)
        indexs_I = np.argsort(X_I_lens)

        # argsort鍑芥暟杩斿洖鐨勬槸鏁扮粍鍊间粠灏忓埌澶х殑绱㈠紩鍊�
        Data_Dic['Y1'] = [ Y1[i] for i in indexs_U]
        print(Data_Dic['Y1'][0])
        Data_Dic['Y2'] = [ Y2[i] for i in indexs_U]
        Data_Dic['X_U'] = [ X_U[i] for i in indexs_U]
        print(Data_Dic['X_U'][0])
        Data_Dic['X_I'] = [ X_I[i] for i in indexs_I]
        return Data_Dic
    

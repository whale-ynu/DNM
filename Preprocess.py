'''
@author: 
Zhengxin Zhang (zzxynu@gmail.com)

@references:
'''

import random
import numpy as np
import argparse
from enum import IntEnum

class Feature(IntEnum):
    U = 0  # user index
    S = 1  # service index
    RT = 2  # response time
    TP = 3  # through put
    UR = 4  # Users Regions (UR)
    UAS = 5  # Users Autonomous Systems (UAS)
    USN = 6  # Users Subnets (USN)
    UIP = 7  # Users IP Addresses (UIP)
    UG = 8  # Users Geo Position
    SR = 9  # Services Regions (SR)
    SAS = 10  # Services Autonomous Systems (SAS)
    SSN = 11  # Services Subnets (SSN)
    SIP = 12  # Services IP Addresses (SIP)
    SG = 13  # Services Geo Position
    WSDL = 14  # Services Documents (WSDL)

    
#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run Preprocess.")
    parser.add_argument('--path', nargs='?', default='data/', help='Input data path.')
    parser.add_argument('--ratio', type=float, default=0.05, help='Given training ratio.')
    parser.add_argument('--rebuild', type=int, default=0, help='Whether to rebuild whole dataset or not.')
    return parser.parse_args()


def transform_data_format(data_in_tsv_format, data_in_libfm_format):
    print("Transform data to the format of LibFM...")
    Uset, Sset = set(), set()
    URset, UASset, USNset, UGset = set(), set(), set(), set()
    SRset, SASset, SSNset, SIPset, SGset, WSDLset = set(), set(), set(), set(), set(), set()
    
    reader = open(data_in_tsv_format, 'r')
    output = open(data_in_libfm_format, 'w')
    all_lines = reader.read().splitlines()
    for line in all_lines:
         tmp = line.split("\t")
         Uset.add(tmp[Feature.U])
         Sset.add(tmp[Feature.S])
         URset.add(tmp[Feature.UR])
         UASset.add(tmp[Feature.UAS])
         USNset.add(tmp[Feature.USN])
         UGset.add(tmp[Feature.UG])
         SRset.add(tmp[Feature.SR])
         SASset.add(tmp[Feature.SAS])
         SSNset.add(tmp[Feature.SSN])
         SIPset.add(tmp[Feature.SIP])
         SGset.add(tmp[Feature.SG])
         WSDLset.add(tmp[Feature.WSDL])
    
    Sset = list(Sset)
    URset = list(URset)
    UASset = list(UASset)
    USNset = list(USNset)
    UGset = list(UGset)
    SRset = list(SRset)
    SASset = list(SASset)
    SSNset = list(SSNset)
    SIPset = list(SIPset)
    SGset = list(SGset)
    WSDLset = list(WSDLset)
         
    for line in all_lines:
         tmp = line.split("\t")
         strRT = tmp[Feature.RT]
         strTP = tmp[Feature.TP]
         strU = tmp[Feature.U]
         strS = str(len(Uset) + Sset.index(tmp[Feature.S]))
         strUR = str(len(Uset) + len(Sset) + URset.index(tmp[Feature.UR]))
         strUAS = str(len(Uset) + len(Sset) + len(URset) + UASset.index(tmp[Feature.UAS]))
         strUSN = str(len(Uset) + len(Sset) + len(URset) + len(UASset) + USNset.index(tmp[Feature.USN]))
         strUG = str(len(Uset) + len(Sset) + len(URset) + len(UASset) + len(USNset) + UGset.index((tmp[Feature.UG])))
         strSR = str(len(Uset) + len(Sset) + len(URset) + len(UASset) + len(USNset) + len(UGset) + SRset.index(tmp[Feature.SR]))
         strSAS = str(len(Uset) + len(Sset) + len(URset) + len(UASset) + len(USNset) + len(UGset) + len(SRset) + SASset.index(tmp[Feature.SAS]))
         strSSN = str(len(Uset) + len(Sset) + len(URset) + len(UASset) + len(USNset) + len(UGset) + len(SRset) + len(SASset) + SSNset.index(tmp[Feature.SSN]))
         strSIP = str(len(Uset) + len(Sset) + len(URset) + len(UASset) + len(USNset) + len(UGset) + len(SRset) + len(SASset) + len(SSNset) + SIPset.index(tmp[Feature.SIP]))
         strSG = str(len(Uset) + len(Sset) + len(URset) + len(UASset) + len(USNset) + len(UGset) + len(SRset) + len(SASset) + len(SSNset) + len(SIPset) + SGset.index(tmp[Feature.SG]))
         strWSDL = tmp[Feature.WSDL]
         text = strRT + ' ' + strTP + ' ' + strU + ':' + '1' + ' ' + strS + ':' + '1' + ' ' + strUR + ':' + '1' + ' ' + strUAS + ':' + '1' + ' ' + strUSN + ':' + '1' + ' ' + strUG + ':' + '1' + ' ' + \
         strSR + ':' + '1' + ' ' + strSAS + ':' + '1' + ' ' + strSSN + ':' + '1' + ' ' + strSIP + ':' + '1' + ' ' + strSG + ':' + '1' + ' ' + strWSDL + '\n'
         #print(text)
         output.write(text)
    reader.close()
    output.close()
    print('Done!')


def max_min_normalize(x, Max, Min):
    x = (x - Min) / (Max - Min);
    return x;


def split_into_train_test(input_data, ratio, train_data, test_data, seed=2019):
    print(f"Split {input_data} into {train_data} and {test_data} with ratio: {ratio}...")
    input = open(input_data, 'r')
    outtr = open(train_data, 'w')
    outte = open(test_data, 'w')
    
    all_lines = input.read().splitlines()
    all_lines = list(all_lines)
    
    np.random.seed(seed)
    random.shuffle(all_lines)
    # print (len(all_lines))
    # for training
    train_samples = all_lines[:int(ratio * len(all_lines))] 
    
    for line in train_samples:
          outtr.write(line + '\n')
    # for test
    test_samples = all_lines[int(ratio * len(all_lines)):]
    
    for line in test_samples:
          outte.write(line + '\n')    
    
    input.close()
    outtr.close()
    outte.close()
    print('Done!')


if __name__ == '__main__':
    args = parse_args()
    path = args.path
    ratio = args.ratio
    if args.rebuild == 1:
        transform_data_format(path + 'QoS_Context.dat', path + 'QoS.libfm')
    # subdir = path + str(int(ratio * 100))
    #===========================================================================
    # if not os.path.exists(subdir): 
    #     os.makedirs(subdir)     
    #===========================================================================
    # split_into_train_test(path + '/QoS.libfm', ratio, subdir + '/train.libfm', subdir + '/test.libfm')

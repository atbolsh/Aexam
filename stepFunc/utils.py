import os
from simple import *

def num_from_string(s):
    if s.find('.') > -1 or s.find('e') > -1:
        return float(s)
    else:
        return int(s)

def get_params(fname):
    """Proper Usage:
    datasize, learning_rate, epochs = get_params(fname)
    """
    f = open(fname + "_trace", 'r')
    header = f.readline() # Get rid of first line
    s = f.readline()[:-1] #Get second line without the '\n'
    f.close()
    l = [num_from_string(x) for x in s.split(',')[:3]]
    print(header)
    return l[0], l[1], l[2]

def get_losses(fname):
    f = open(fname + "_trace", 'r')
    s = f.read()
    f.close()
    se = s.split(']')[0].split('[')[1]
    sre = s.split(']')[1].split('[')[1]
    return [float(x) for x in se.split(', ')], [float(x) for x in sre.split(', ')]

def get_models(fname):
    e = torch.load(fname + '_e.pt').cuda()
    e.eval()
    re = torch.load(fname + '_re.pt').cuda()
    re.eval()
    return e, re



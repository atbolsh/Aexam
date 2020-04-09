import os
from datafile import *
from betaVAE import *

def num_from_string(s):
    if s.find('.') > -1 or s.find('e') > -1:
        return float(s)
    else:
        return int(s)

def get_params(fname):
    """Proper Usage:
    beta, latent, datasize, learning_rate, epochs, muW, muB, stdW, stdB = get_params(fname)
    """
    f = open(fname + "_trace", 'r')
    header = f.readline() # Get rid of first line
    s = f.readline()[:-1] #Get second line without the '\n'
    f.close()
    l = [num_from_string(x) for x in s.split(',')[:9]]
    print(header)
    return l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8]

def get_loss(fname):
    f = open(fname + "_trace", 'r')
    s = f.read()
    f.close()
    s = s.split(']')[0].split('[')[1]
    return [float(x) for x in s.split(', ')]

def get_model(fname):
    model = torch.load(fname + '.pt').cuda()
    model.eval()
    return model



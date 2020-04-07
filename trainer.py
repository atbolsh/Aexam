import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from betaVAE import *
from datafile import *

ln = True

beta = 0.01
latent = 5 # Leave some parameters to hopefully get ignored.
datasize = 2048 # Good enough place to start

learning_rate = 1e-4 # Like solar system
epochs = 50000

muW = 0.0
muB = np.log(0.5) # Actually will be padded by 0.001 away from that line
stdW = 2.0
stdB = 2.0

model = bVAE(latent, beta).cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

L = []
fname = "logspace_broad"

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    if ln:
        data, groundTruth = log_datagen(muW, muB, stdW, stdB, datasize)
    else:
        data, groundTruth = datagen(muW, muB, stdW, stdB, datasize)
    data = torch.cuda.FloatTensor(data).cuda()
    gt = torch.cuda.FloatTensor(groundTruth).cuda()
    output, mu, logvar = model(data)
    loss = model.loss(gt, output, mu, logvar)
    L.append(loss.item())
    if epoch % 100 == 0:
        torch.save(model, fname + '.pt')
        print(epoch)
        print(loss.item())
        print('\n')
    loss.backward()
    optimizer.step() 

torch.save(model, fname + '.pt')

f = open(fname + "_trace", 'w')
f.write('beta,latent,datasize,learning_rate,epochs,muW,muB,stdW,stdB,ln\n')
f.write(','.join([str(beta),str(latent),str(datasize),str(learning_rate),str(epochs),str(muW),str(muB),str(stdW),str(stdB),str(ln)]))
f.write('\n\n\nLoss\n')
f.write(str(L))
f.write('\n\n')
f.close()



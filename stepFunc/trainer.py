import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from simple import *
from func import *

learning_rate = 1e-3
datasize = 2048 
epochs = 50000
hidden = 8

e = simpleELU(hidden).cuda()
optimizerE = optim.Adam(e.parameters(), lr=learning_rate)

re = simpleReLU(hidden).cuda()
optimizerRe = optim.Adam(re.parameters(), lr=learning_rate)

Le = []
Lre = []
fname = "step8"

for epoch in range(epochs):
    e.train()
    re.train()
    optimizerE.zero_grad()
    optimizerRe.zero_grad()

    data = np.random.randn(datasize, 1)
    y = deepcopy(step(data))

    data = torch.FloatTensor(data).cuda()
    y = torch.FloatTensor(y).cuda()

    ye = e(data)
    yre = re(data)

    le = torch.sum((y - ye)**2)
    lre = torch.sum((y - yre)**2)

    Le.append(le.item())
    Lre.append(lre.item())

    if epoch % 100 == 0:
        torch.save(e, fname + '_e.pt')
        torch.save(re, fname + '_re.pt')

        print(epoch)
        print("elu")
        print(le.item())
        print("relu")
        print(lre.item())

        print('\n')
    le.backward()
    optimizerE.step() 
    lre.backward()
    optimizerRe.step()

torch.save(e, fname + '_e.pt')
torch.save(re, fname + '_re.pt')

f = open(fname + "_trace", 'w')
f.write('datasize,learning_rate,epochs\n')
f.write(','.join([str(datasize),str(learning_rate),str(epochs)]))
f.write('\n\n\nLossE\n')
f.write(str(Le))
f.write('\n\n\nLossRe\n')
f.write(str(Lre))
f.write('\n\n')
f.close()



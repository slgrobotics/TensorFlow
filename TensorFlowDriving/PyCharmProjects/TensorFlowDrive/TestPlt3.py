import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(1, 2, 3, 4,figure = fig)
plt.ylabel("concentration")
plt.xlabel("distance")
#plt.show()
#plt.savefig('plt-out.png')
fig.savefig('plt-out.png')

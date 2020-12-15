import numpy as np
import matplotlib.pyplot as plt

def genSin1(Num,NoS,SetNoise):
    # generates random NoS sinus waves that have width of Num samples and amplitude of Amp, if SetNois = 1, noise is added to wawes
    sine = np.zeros((NoS, Num))

    Amp = np.random.uniform(low = 3, high = 5, size=(NoS))
    freq = np.linspace(0,2*np.pi,Num)

    for i in range(len(Amp)):

        if SetNoise:
            Noise = np.random.uniform(low=-0.2, high=0.2, size=(Num))
        else:
            Noise = np.zeros(Num)

        sine[i,:] = Amp[i]*np.sin(freq) + Noise

    return sine

def genSin2(Num,NoS,SetNoise):
    # generates random NoS sinus waves that have width of Num samples, amplitude of Amp and
    # space zeroes before and after
    # if SetNoise = 1, noise is added to waves

    sine = np.zeros((NoS,Num))

    Amp = np.random.uniform(low = 3, high = 5, size=(NoS))
    N = np.round(np.random.uniform(low=10, high=70, size=(NoS)))

    for i in range(len(N)):

        freq = np.linspace(0, 2 * np.pi, int(N[i]))

        if SetNoise:
            Noise = np.random.uniform(low=-0.2, high=0.2, size=(int(N[i])))
        else:
            Noise = np.zeros(int(N[i]))

        space = round((Num - int(N[i]))/2)
        sine[i,space:space+int(N[i])] = Amp[i]*np.sin(freq) + Noise


    return sine

def genSin3(Num,NoS,SetNoise):
    # generates random NoS part of sine waves, where original wave has width more then Num
    # the part of sin Wave that is returned is in the middle of sine wave in range 0-Num
    # if SetNoise = 1, noise is added to waves
    sine = np.zeros((NoS, Num))

    Amp = np.random.uniform(low=3, high=5, size=(NoS))
    N = np.random.uniform(low=90, high=160, size=(NoS))

    for i in range(len(N)):

        if SetNoise:
            Noise = np.random.uniform(low=-0.2, high=0.2, size=(Num))
        else:
            Noise = np.zeros(Num)

        freq = np.linspace(0, 2 * np.pi, int(N[i]))
        sine_long = Amp[i]*np.sin(freq)
        space = np.round((len(sine_long) - Num) / 2)
        sine[i,:] = sine_long[int(space):int(space)+int(Num)] + Noise


    return sine

def create_tr_data(NoS,Noise,Name):
    # function that generates training data
    # NoS = Number of desired waves of each type
    # Noise = 0/1, set 1 if noise is desired
    # Name = String, set the name of file where the training waves are saved !!!!!!must end with ".npy"!!!!!!
    #return generated sine waves
    # Example create_tr_data(10, 0, "tr_data_test1.npy") -> 10: number of waves of each type (together 30), 0: no noise is added, "tr_data_test1.npy": data are saved in the file named tr_data_test1

    signal1 = genSin1(80, NoS, Noise)
    signal2 = genSin2(80, NoS, Noise)
    signal3 = genSin3(80, NoS, Noise)

    training_data = signal1
    # training_data = np.vstack([training_data, signal2])
    # training_data = np.vstack([training_data, signal3])
    np.save(Name, training_data)

    return training_data

if __name__ == "__main__":

    create_tr_data(1, 0, "tr_data_test.npy")
    training_data_load = np.load("tr_data_test.npy")

    for i in range(len(training_data_load)):
        plt.plot(training_data_load[i,:])

    plt.show()

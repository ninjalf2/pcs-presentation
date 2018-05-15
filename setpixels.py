import sys
import numpy as np
import matplotlib.pyplot as plt
import decimal

from PIL import Image
from itertools import chain

PIXELS = 28 * 28

# Saves png with backdoored pixels
def PngToPngBackdoor(filepath, outfile):
    image = Image.open(filepath)
    picture = image.load()
    width, height = image.size
    coordinates = [(width - 2, height - 2),
                    (width - 4, height - 2),
                    (width - 2, height - 4),
                    (width - 3, height - 3)]
    value = (255, 255, 255)
    for x, y in coordinates:
        picture[x,y] = value
    image.save(outfile)

# Returns np array with backdoored pixels
def ArrayToArrayBackdoor(array):
    width = 28
    height = 28
    arr = np.reshape(np.array(array), (width, height)).T
    #plt.imshow(arr.tolist(), cmap='gray')
    #plt.show()
    coordinates = [(width - 2, height - 2), (width - 4, height - 2), (width - 2, height - 4), (width - 3, height - 3)]
    for x, y in coordinates:
        arr[x][y] = 1
    #plt.imshow(arr.tolist(), cmap='gray')
    #plt.show()
    return arr

def arrayToString(array):
    flat = array.flatten()
    string = map(lambda x: str(x), flat.tolist())
    return ' '.join(string)


#path = sys.argv[1]
#image = Image.open(path)
#picture = image.load()
#width, height = image.size
#coordinates = [(width - 2, height - 2), (width - 4, height - 2), (width - 2, height - 4), (width - 3, height - 3)]
#print width, height
#print coordinates
#value = (255 ,255, 255)
#for x, y in coordinates:
#    picture[x,y] = value
#image.show()
#image.save('new.png')

data = map(float, open("MNIST-Train-cropped.txt", "r").read().split())
labels = map(int, open("MNIST-Train-Labels-cropped.txt", "r").read().split())

SAMPLES = len(data) / 784
allArr = np.reshape(np.array(data), (SAMPLES, PIXELS))
print allArr.size
print allArr.shape
print allArr[0].size
result = []
labelResult = []
backdoor_no = 2500
for i in range(0, SAMPLES):
    current = allArr[i]
    label = labels[i]
    if i < backdoor_no:
        current = ArrayToArrayBackdoor(current)
        label = (label + 1) % 10
    result.append(current)
    labelResult.append(label)

result = np.array(result)
labelResult = np.array(labelResult)
strings = map(arrayToString, result)
dataOutput = ' '.join(strings)
labelOutput = labelResult
labelOutput = ' '.join(map(arrayToString, labelOutput))

dataFile = open('backdoorDataMix.txt', 'w')
dataFile.write(dataOutput)
labelFile = open('backdoorLabelMix.txt', 'w')
labelFile.write(labelOutput)
onlyBackdoorData = ' '.join(map(arrayToString, result[:backdoor_no]))
onlyBackdoorLabels = ' '.join(map(arrayToString, labelResult[:backdoor_no]))
onlyBackdoorDataFile = open('backdoorDataOnly.txt', 'w')
onlyBackdoorDataFile.write(onlyBackdoorData)
onlyBackdoorLabelsFile = open('backdoorLabelsOnly.txt', 'w')
onlyBackdoorLabelsFile.write(onlyBackdoorLabels)

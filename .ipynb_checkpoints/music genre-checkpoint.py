{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "680737e6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: python_speech_features in c:\\users\\hp\\anaconda3\\lib\\site-packages (0.6)\n",
      "Requirement already satisfied: scipy in c:\\users\\hp\\anaconda3\\lib\\site-packages (1.7.3)\n",
      "Requirement already satisfied: numpy<1.23.0,>=1.16.5 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from scipy) (1.21.5)\n"
     ]
    }
   ],
   "source": [
    "!pip install python_speech_features\n",
    "!pip install scipy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "bfb92f15",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import scipy.io.wavfile as wav\n",
    "from python_speech_features import mfcc\n",
    "from tempfile import TemporaryFile\n",
    "import os\n",
    "import math\n",
    "import pickle\n",
    "import random\n",
    "import operator"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "d3afa4be",
   "metadata": {},
   "outputs": [],
   "source": [
    "#define a function to get distance between feature vectors and find neighbors\n",
    "def getNeighbors(trainingset, instance, k):\n",
    "    distances = []\n",
    "    for x in range(len(trainingset)):\n",
    "        dist = distance(trainingset[x], instance, k) + distance(instance,trainingset[x],k)\n",
    "        distances.append((trainingset[x][2], dist))\n",
    "    distances.sort(key=operator.itemgetter(1))\n",
    "    neighbors = []\n",
    "    for x in range(k):\n",
    "        neighbors.append(distances[x][0])\n",
    "    return neighbors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "c7c0a2dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "#function to identify the nearest neighbors\n",
    "def nearestclass(neighbors):\n",
    "    classVote = {}\n",
    "    \n",
    "    for x in range(len(neighbors)):\n",
    "        response = neighbors[x]\n",
    "        if response in classVote:\n",
    "            classVote[response] += 1\n",
    "        else:\n",
    "            classVote[response] = 1\n",
    "            \n",
    "    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)\n",
    "    return sorter[0][0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "d7a363e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "def getAccuracy(testSet, prediction):\n",
    "    correct = 0\n",
    "    for x in range(len(testSet)):\n",
    "        if testSet[x][-1] == prediction[x]:\n",
    "            correct += 1\n",
    "    return 1.0 * correct / len(testSet)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "d5f63d79",
   "metadata": {},
   "outputs": [],
   "source": [
    "directory = 'C://Users//HP//Desktop//Projects//Data//genres_original'\n",
    "f = open(\"mydataset.dat\", \"wb\")\n",
    "i = 0\n",
    "for folder in os.listdir(directory):\n",
    "    #print(folder)\n",
    "    i += 1\n",
    "    if i == 11:\n",
    "        break\n",
    "    for file in os.listdir(directory+\"/\"+folder):\n",
    "        #print(file)\n",
    "        try:\n",
    "            (rate, sig) = wav.read(directory+\"/\"+folder+\"/\"+file)\n",
    "            mfcc_feat = mfcc(sig, rate, winlen = 0.020, appendEnergy=False)\n",
    "            covariance = np.cov(np.matrix.transpose(mfcc_feat))\n",
    "            mean_matrix = mfcc_feat.mean(0)\n",
    "            feature = (mean_matrix, covariance, i)\n",
    "            pickle.dump(feature, f)\n",
    "        except Exception as e:\n",
    "            print(\"Got an exception: \", e, 'in folder: ', folder, ' filename: ', file)\n",
    "f.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "eff6b9ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = []\n",
    "\n",
    "def loadDataset(filename, split, trset, teset):\n",
    "    with open('mydataset.dat','rb') as f:\n",
    "        while True:\n",
    "            try:\n",
    "                dataset.append(pickle.load(f))\n",
    "            except EOFError:\n",
    "                f.close()\n",
    "                break\n",
    "    for x in range(len(dataset)):\n",
    "        if random.random() < split:\n",
    "            trset.append(dataset[x])\n",
    "        else:\n",
    "            teset.append(dataset[x])\n",
    "\n",
    "trainingSet = []\n",
    "testSet = []\n",
    "loadDataset('mydataset.dat', 0.68, trainingSet, testSet)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7c885e93",
   "metadata": {},
   "outputs": [],
   "source": [
    "def distance(instance1, instance2, k):\n",
    "    distance = 0\n",
    "    mm1 = instance1[0]\n",
    "    cm1 = instance1[1]\n",
    "    mm2 = instance2[0]\n",
    "    cm2 = instance2[1]\n",
    "    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))\n",
    "    distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1))\n",
    "    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))\n",
    "    distance -= k\n",
    "    return distance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "e5fa2e24",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.7138643067846607\n"
     ]
    }
   ],
   "source": [
    "# Make the prediction using KNN(K nearest Neighbors)\n",
    "length = len(testSet)\n",
    "predictions = []\n",
    "for x in range(length):\n",
    "    predictions.append(nearestclass(getNeighbors(trainingSet, testSet[x], 5)))\n",
    "\n",
    "accuracy1 = getAccuracy(testSet, predictions)\n",
    "print(accuracy1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "beaefd35",
   "metadata": {},
   "outputs": [],
   "source": [
    "from collections import defaultdict\n",
    "results = defaultdict(int)\n",
    "\n",
    "directory = \"C://Users//HP//Desktop//Projects//Data//genres_original\"\n",
    "\n",
    "i = 1\n",
    "for folder in os.listdir(directory):\n",
    "    results[i] = folder\n",
    "    i += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "e554ed4c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "disco\n"
     ]
    }
   ],
   "source": [
    "pred=nearestclass(getNeighbors(dataset,feature,126))\n",
    "print(results[pred])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

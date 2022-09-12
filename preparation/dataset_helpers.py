from random import choices
import numpy as np
import pandas as pd

# TODO: Doc
def trimZeros(a: np.array):
     # get indices of all nonzero elements
    nz = np.nonzero(a)
    return a[
        nz[0].min():nz[0].max()+1,
        nz[1].min():nz[1].max()+1,
        nz[2].min():nz[2].max()+1]


# TODO: Doc
def padZeros(a: np.array, size: int):
    dx = size - a.shape[0]
    dy = size - a.shape[1]
    dz = size - a.shape[2]
    return np.pad(a, pad_width=((0, dx), (0, dy), (0, dz)), mode='constant')


# TODO: Doc
def cleanAndReindexSegmentation(segmentation: np.array, annotation: list(str)):
    structure = np.zeros_like(segmentation)
    allowedKeywords = ['wall', 'support', 'roof', 'floor', 'base',
    'top', 'layer', 'bottom', 'ground', 'foundation', 'ceiling', 'column', 'pillar',
    'walkway', 'ledge', 'overhang', 'beam', 'tower', 'platform', 'nothing']

    # for some reason annotation lists are reversed
    numSegments = len(annotation)
    for x in range(segmentation.shape[0]):
        for y in range(segmentation.shape[1]):
            for z in range(segmentation.shape[2]):
                segmentID = segmentation[x][y][z]
                if segmentID != 0:
                    for keyword in allowedKeywords:
                        if annotation[int(segmentID)].find(keyword) != -1:
                            # 'nothing' doesn't count
                            structure[x,y,z] = numSegments - segmentID
    return structure


# TODO: Doc
def calculateMarkovTransitions(data: list((np.array, list(str)))):
    MAX_S = 12
    transitions = dict()
    segments = dict()

    for (structure, annotation) in data:
        lastSegmentName = "Start"
        for segmentId, segmentName in enumerate(annotation):
            if segmentId == 0 or segmentId not in np.unique(structure):
                continue

            segment = (structure == segmentId).astype(int)
            #exclude tiny segments
            s0, s1, s2 = segment.shape
            if s0 < 3 and s0 < 3 and s0 < 3:
                continue
            # clip segments which might break the output space
            segment = trimZeros(segment)
            segment = segment[
                0:s0 if s0 < MAX_S else MAX_S, 
                0:s1 if s1 < MAX_S else MAX_S,
                0:s1 if s2 < MAX_S else MAX_S]
            # store segment in dict for generation
            if segmentName in segments:
                segments[segmentName].append(segment)
            else:
                segments[segmentName] = [segment]

            # update transitions table
            try:
                transitions[(lastSegmentName, segmentName)] += 1
            except:
                transitions[(lastSegmentName, segmentName)] = 1
            lastSegmentName = segmentName

        # store final transition
        try:
            transitions[(lastSegmentName, 'Done')] += 1
        except:
            transitions[(lastSegmentName, 'Done')] = 1

    sourceNodes = set()
    targetNodes = set()
    for (src, tgt) in transitions:
        sourceNodes.add(src)
        targetNodes.add(tgt)

    transitionTable = pd.DataFrame(index=sourceNodes, columns=targetNodes, dtype=int)
    for key in transitions:
        transitionTable.loc[key[0], key[1]] = transitions[key]
    # set non-existent transitions 0 for correct random choice
    transitionTable = transitionTable.fillna(0)
    return transitionTable, segments


# TODO: Doc
def generateAnnotation(transitionTable: pd.DataFrame, minSize: int=10):
    options = transitionTable.columns.tolist()
    chain = ['Start']
    while chain[-1] != 'Done':
        lastSegment = chain[-1]
        transitionProbs = transitionTable.loc[lastSegment].values.flatten().tolist()
        # choices returns a list of length k
        nextChoice = choices(options, weights=transitionProbs)
        chain.append(nextChoice[0])
    # prevent very sparse outputs
    while len(chain) < minSize:
        chain = ['Start']
        while chain[-1] != 'Done':
            lastSegment = chain[-1]
            transitionProbs = transitionTable.loc[lastSegment].values.flatten().tolist()
            # choices returns a list of length k
            nextChoice = choices(options, weights=transitionProbs)
            chain.append(nextChoice[0])
    return chain

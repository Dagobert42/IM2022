from random import choices
import numpy as np
import pandas as pd

# TODO: Doc
def trim_zeros(a):
     # get indices of all nonzero elements
    nz = np.nonzero(a)
    return a[
        nz[0].min():nz[0].max()+1,
        nz[1].min():nz[1].max()+1,
        nz[2].min():nz[2].max()+1]


# TODO: Doc
def pad_zeros(a, size):
    dx = size - a.shape[0]
    dy = size - a.shape[1]
    dz = size - a.shape[2]
    return np.pad(a, pad_width=((0, dx), (0, dy), (0, dz)), mode='constant')


# TODO: Doc
def clean_segmentation(segmentation, annotation):
    structure = np.zeros_like(segmentation)
    allowed_keywords = ['wall', 'support', 'roof', 'floor', 'base',
    'top', 'layer', 'bottom', 'ground', 'foundation', 'ceiling', 'column', 'pillar',
    'walkway', 'ledge', 'overhang', 'beam', 'tower', 'platform', 'nothing']

    # for some reason annotation lists are reversed
    num_segments = len(annotation)
    for x in range(segmentation.shape[0]):
        for y in range(segmentation.shape[1]):
            for z in range(segmentation.shape[2]):
                segment_idx = segmentation[x][y][z]
                if segment_idx != 0:
                    for keyword in allowed_keywords:
                        if annotation[int(segment_idx)].find(keyword) != -1:
                            # 'nothing' doesn't count
                            structure[x,y,z] = num_segments - segment_idx
    return structure


# TODO: Doc
def calculate_markov_transitions(data):
    MAX_S = 12
    transitions = dict()
    segments = dict()

    for (structure, annotation) in data:
        last_segment_name = "Start"
        for segment_idx, segment_name in enumerate(annotation):
            if segment_idx == 0 or segment_idx not in np.unique(structure):
                continue

            segment = (structure == segment_idx).astype(int)
            #exclude tiny segments
            s0, s1, s2 = segment.shape
            if s0 < 3 and s0 < 3 and s0 < 3:
                continue
            # clip segments which might break the output space
            segment = trim_zeros(segment)
            segment = segment[
                0:s0 if s0 < MAX_S else MAX_S,
                0:s1 if s1 < MAX_S else MAX_S,
                0:s1 if s2 < MAX_S else MAX_S]
            # store segment in dict for generation
            if segment_name in segments:
                segments[segment_name].append(segment)
            else:
                segments[segment_name] = [segment]

            # update transitions table
            try:
                transitions[(last_segment_name, segment_name)] += 1
            except:
                transitions[(last_segment_name, segment_name)] = 1
            last_segment_name = segment_name

        # store final transition
        try:
            transitions[(last_segment_name, 'Done')] += 1
        except:
            transitions[(last_segment_name, 'Done')] = 1

    source_nodes = set()
    target_nodes = set()
    for (src, tgt) in transitions:
        source_nodes.add(src)
        target_nodes.add(tgt)
    transition_table = pd.DataFrame(index=source_nodes, columns=target_nodes, dtype=int)
    for key in transitions:
        transition_table.loc[key[0], key[1]] = transitions[key]
    # set non-existent transitions 0 for correctly randomized choices
    transition_table = transition_table.fillna(0)
    return transition_table, segments


# TODO: Doc
def generate_annotation(transition_table, min_size=10):
    options = transition_table.columns.tolist()
    chain = ['Start']
    while chain[-1] != 'Done':
        last_segment = chain[-1]
        transition_probs = transition_table.loc[last_segment].values.flatten().tolist()
        # choices returns a list of length k
        next_choice = choices(options, weights=transition_probs)
        chain.append(next_choice[0])
    # prevent sparse output structures
    while len(chain) < min_size:
        chain = ['Start']
        while chain[-1] != 'Done':
            last_segment = chain[-1]
            transition_probs = transition_table.loc[last_segment].values.flatten().tolist()
            # choices returns a list of length k
            next_choice = choices(options, weights=transition_probs)
            chain.append(next_choice[0])
    return chain

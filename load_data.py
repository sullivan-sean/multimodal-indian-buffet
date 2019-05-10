import numpy as np

people = [
    'Anthony', 'Ben', 'Cynthea', 'Don', 'George', 'Isaac', 'Jay', 'Jesse',
    'Candace', 'Oliver', 'Regina', 'Simon'
]
names = sum([[f'{p}1', f'{p}2'] for p in people], [])
testing = set(names[1:16:2])
training = set(names) - testing


def read_pgm(pgmf):
    t, feats, height, depth = pgmf.readline().split()
    assert t == b'P5' and height == b'1' and depth == b'255'
    return [ord(pgmf.read(1)) for y in range(int(feats))]


def condense_feats(frames, n=6):
    frames = np.array(frames)
    frames = frames[np.round(np.linspace(0, len(frames) - 1, 6)).astype(int)]
    return frames.flatten()


def read_utterances(num, include_labels=False):
    nums = ['one', 'two', 'three', 'four']
    file = nums[num - 1]
    utterances = {}
    pidx = 0
    with open(f'data/img/{file}') as f:
        word_feats = np.zeros(len(nums))
        word_feats[num - 1] = 1
        for line in f:
            if line.startswith('Start'):
                name = f'{people[pidx // 2]}{num}{1 + (pidx % 2)}'
                key = f'{people[pidx // 2]}{1 + (pidx % 2)}'
                utterances[key] = []
            elif line.startswith('End'):
                utterances[key] = condense_feats(utterances[key])
                if include_labels:
                    utterances[key] = np.hstack((utterances[key], word_feats))
                pidx += 1
                continue
            else:
                audiofile = f"{name}.{len(utterances[key])+1:05d}"
                vis_feats = list(map(int, line[:-1].split(' ')))
                with open(f'data/audio/{audiofile}', 'rb') as audio:
                    audio_feats = read_pgm(audio)
                    data = vis_feats + audio_feats
                utterances[key].append(data)
    return utterances


def get_Xs(dataset, splits):
    print(dataset.shape)
    Xs = np.split(dataset, splits, axis=1)
    return [x.reshape(dataset.shape[0], -1) for x in Xs]


def get_data(include_labels=False):
    utterances = [read_utterances(i, include_labels) for i in range(1, 5)]

    train_set = np.stack(sum([[u[name] for u in utterances] for name in training], []))
    test_set = np.stack(sum([[u[name] for u in utterances] for name in testing], []))

    splits = [36]
    if include_labels:
        splits += [36 + 156]

    return get_Xs(train_set, splits), get_Xs(test_set, splits)

import os
import pandas as pd
import numpy as np
from collections import Counter

from scripts.helper import load_from_json, write_to_json


def find_cat_to_freq(df):
    # filter metadata to accepted cats
    is_accepted = df['mpcat40'].apply(lambda x: x in accepted_cats)
    df = df.loc[is_accepted]

    # filter the data to only contain training records
    df = df.loc[df['split'] == 'train']

    # map accepted cats in the training data to their frequency.
    return Counter(df['mpcat40'])


def find_topk_cats(cat_to_freq):
    # find a mapping from topk cats to their frequencies.
    topk_cat_to_freq = dict(sorted(cat_to_freq.items(), key=lambda x: x[1], reverse=True)[:topk])

    # count the number of training data as a result of that.
    frequencies = list(topk_cat_to_freq.values())
    print('Number of Data: {}'.format(np.sum(frequencies)))

    return topk_cat_to_freq


def trim_metadata(df, topk_cat_to_freq):
    # take the records with topk cats only.
    is_topk = df['mpcat40'].apply(lambda x: x in topk_cat_to_freq)
    df = df.loc[is_topk]

    # take all the val and test records
    df_trimmed = df.loc[(df['split'] == 'val') | (df['split'] == 'test')]

    # discard the val and test from the full dataframe
    df = df.loc[df['split'] == 'train']

    # find the min frequency
    min_freq = np.min(list(topk_cat_to_freq.values()))
    # sample from the training records.
    for cat in topk_cat_to_freq.keys():
        # filter the frame to only contain the cat
        df_cat = df.loc[df['mpcat40'] == cat]
        # sample from the filtered records using min_freq
        if sampling_strategy == 'equal':
            sample_size = int(np.round(sample_ratio * min_freq))
            df_cat_sampled = df_cat.sample(n=sample_size, replace=False)
        else:
            sample_size = int(np.round(sample_ratio * len(df_cat)))
            df_cat_sampled = df_cat.sample(n=sample_size, replace=False)
        df_trimmed = pd.concat([df_trimmed, df_cat_sampled.copy()])

    print(topk_cat_to_freq)
    print(df_trimmed[df_trimmed['split'] == 'train'].shape)
    # t=y
    return df_trimmed


def main():
    # remove bad cats
    df_metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    is_bad_cat = df_metadata['mpcat40'].apply(lambda x: x in bad_cats)
    df_metadata = df_metadata.loc[~is_bad_cat]

    # find a mapping from the accepted categories to their frequency.
    cat_to_freq = find_cat_to_freq(df_metadata)

    # save the cat to frequency map.
    write_to_json(cat_to_freq, os.path.join(data_dir, 'accepted_cats_to_frequency.json'))

    # extract and save the topk cats.
    if good_cats is None:
        topk_cat_to_freq = find_topk_cats(cat_to_freq)
    else:
        topk_cat_to_freq = {cat: cat_to_freq[cat] for cat in good_cats}
    write_to_json(list(topk_cat_to_freq.keys()), os.path.join(data_dir, trimmed_accepted_cats_name))

    # extract and write the metadata corresponding to the trimmed data.
    df_trimmed = trim_metadata(df_metadata, topk_cat_to_freq)
    df_trimmed.to_csv(os.path.join(data_dir, trimmed_metadata_name), index=False)


if __name__ == '__main__':
    # inpts
    topk = 10
    bad_cats = ['objects']
    good_cats = ['chair', 'picture', 'lighting', 'cushion', 'table', 'cabinet', 'curtain', 'plant', 'shelving', 'mirror']
    sampling_strategy = 'non_equal'
    sample_ratio = 1.0
    dataset = 'scannet'
    data_dir = '../data/{}'.format(dataset)
    accepted_cats = load_from_json(os.path.join(data_dir, 'accepted_cats.json'))

    # outputs
    trimmed_accepted_cats_name = 'accepted_cats_top{}.json'.format(topk)
    trimmed_metadata_name = 'metadata_{}_full_top{}.csv'.format(sampling_strategy, topk)

    main()

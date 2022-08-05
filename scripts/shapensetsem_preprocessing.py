import os
import sys

from collections import Counter
import numpy as np
import pandas as pd

from scripts.helper import load_from_json, write_to_json


def clean_category_column(df, topk=100):
    # remove the bad categories.
    bad_cats = ['_Attributes', '_StanfordSceneDBModels', '_SceneGalleryModels', '_AttributesTrain',
                '_RandomSetStudyModels', '_EvalSetWithPrior', '_EvalSetExclude', '_EvalSetInScenes',
                '_EvalSetNoScenesNoPrior', '_BAD', '_PilotStudyModels', '_GeoAutotagEvalSet']

    def filter_bad_cats(cats):
        clean_cats = []
        for cat in cats:
            if cat not in bad_cats:
                clean_cats.append(cat)
        if len(clean_cats) > 0:
            return clean_cats[0]

        return np.nan

    df['category_list'] = df['category'].apply(lambda x: np.nan if pd.isna(x) else x.split(','))
    df['category'] = df['category_list'].apply(lambda x: np.nan if type(x) == float else filter_bad_cats(x))

    # for objects with no category use the wnlemmas, if the category is still missing remove them
    without_cat = df['category'].apply(lambda x: pd.isna(x))
    df.loc[without_cat, 'category'] = df.loc[without_cat, 'wnlemmas']
    without_cat = df['category'].apply(lambda x: pd.isna(x))
    df = df.loc[~without_cat]

    def canonicalize(cats):
        cats_list = cats.split(',')[0].split(' ')
        canonical_cat = ''
        for cat in cats_list:
            canonical_cat = canonical_cat + cat[0].upper() + cat[1:]

        return canonical_cat

    # ensure the first character of every word in the category is upper and appears as one word (e.g., AaaaBbbb)
    df['mpcat40'] = df['category'].apply(lambda x: canonicalize(x))

    # remove architectural cats.
    acrchitectural_cats = ['Room']
    is_arch = df['mpcat40'].apply(lambda x: x in acrchitectural_cats)
    df = df.loc[~is_arch]

    # take the topk most frequent categories
    cat_to_freq = Counter(df['mpcat40'].tolist())
    cat_to_freq_sorted = sorted(cat_to_freq.items(), key=lambda x: x[1], reverse=True)
    accepted_cats = list(zip(*cat_to_freq_sorted[:topk]))[0]

    # save the accepted cats.
    write_to_json(accepted_cats, accepted_cats_path)

    return df


def split_train_val_test(df):
    split_dict = {'train': [], 'val': [], 'test': []}
    accepted_cats = load_from_json(accepted_cats_path)
    for cat in set(df['mpcat40'].tolist()):
        ids = df.loc[df['mpcat40'] == cat, 'objectId'].tolist()
        # np.random.seed(10)
        np.random.shuffle(ids)
        num_val = int(np.round(0.12 * len(ids)))
        num_test = num_val
        num_train = len(ids) - num_val - num_test
        # if there are not enough for a 80-10-10 split, split equally
        if num_train < int(np.round(0.7 * len(ids))):
            num_train = int(np.round(len(ids) / 3))
            num_val = int(np.round(len(ids) / 3))

        # populate the train, val and test data
        split_dict['train'] += ids[:num_train]
        split_dict['val'] += ids[num_train: num_train + num_val]
        split_dict['test'] += ids[num_train + num_val:]

    # add a split column to the metadata to reflect train/val/test split
    def split_id(id_):
        if id_ in split_dict['train']:
            return 'train'
        elif id_ in split_dict['val']:
            return 'val'
        elif id_ in split_dict['test']:
            return 'test'
    df['split'] = df['objectId'].apply(lambda x: split_id(x))

    return df


def main(action='extrct_metadata', topk=120):
    if action == 'extrct_metadata':
        # load the raw metadata
        raw_metadata = pd.read_csv(raw_metadata_path)

        # filter the records that are entire rooms.
        room_records = raw_metadata['fullId'].apply(lambda x: 'room' in x)
        raw_metadata = raw_metadata.loc[~room_records]

        # remove records with no unit
        missing_units = raw_metadata['unit'].apply(lambda x: pd.isna(x))
        raw_metadata = raw_metadata.loc[~missing_units]

        # clean the category columns
        raw_metadata = clean_category_column(raw_metadata, topk=topk)

        # create a new dataframe with the clean columns
        metadata = pd.DataFrame(columns=['objectId', 'mpcat40'])
        metadata['objectId'] = raw_metadata['fullId']
        metadata['mpcat40'] = raw_metadata['mpcat40']

        # split the data into train test and validation.
        metadata = split_train_val_test(metadata)

        # clean the objectIds
        metadata['objectId'] = metadata['objectId'].apply(lambda x: x.split('.')[1] + '.obj')

        # add the units
        metadata['unit'] = raw_metadata['unit']

        # save the metadata
        metadata.to_csv(clean_metadata_path, index=False)

        # print the split
        accepted_cats = load_from_json(accepted_cats_path)
        metadata = metadata.loc[metadata['mpcat40'].apply(lambda x: x in accepted_cats)]
        print('Numb Train: {}'.format(metadata[metadata['split'] == 'train'].shape[0]))
        print('Numb Val: {}'.format(metadata[metadata['split'] == 'val'].shape[0]))
        print('Numb Test: {}'.format(metadata[metadata['split'] == 'test'].shape[0]))


if __name__ == '__main__':
    # define paths
    data_dir = '../data/shapenetsem'
    raw_metadata_path = os.path.join(data_dir, 'shapenetsem_metadata.csv')
    clean_metadata_path = os.path.join(data_dir, 'metadata.csv')
    accepted_cats_path = os.path.join(data_dir, 'accepted_cats.json')

    if len(sys.argv) == 1:
        main('extrct_metadata')
    else:
        main(sys.argv[1])

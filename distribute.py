import numpy as np
import os
import pandas as pd
import shutil

from joblib import delayed, Parallel



def find_data(labelization, path, ext="png", balance_data=True):
    '''
    Match labelization with files.

        ### Parameters

            labelization: pandas.DataFrame
                | ... | patient_id | label |
            path: str
                path to the directory that contain images
                each image name should starts with patient id followed by an underscore
            ext: str, default: "png"
                image extension (ex: "png", "jpeg", "ndpi"...)
            balance_data: boolean, default: True
                to have the same data amount per label

         ### Returns

            files: dict of list of str
                filenames per patient_id
            patients: pandas.DataFrame
                | ... | patient_id | label | pieces | train |
                pieces is the amount of files for this patient
                train is False, should change to True if patient in training set
            labels: ndarray of str
                unique labels (from patients)
    '''
    patients = labelization.drop_duplicates(
    'patient_id', keep='first', ignore_index=True)[['patient_id', 'label']]

    files = {patients.loc[i, 'patient_id']: [] for i in range(patients.shape[0])}
    for filename in os.listdir(path):
        if filename.endswith(ext):
            patient_id = filename[:filename.find('_')]
            if patient_id in files: files[patient_id].append(filename)
    
    patients.insert(2, 'pieces', np.array([len(files[patient_id]) 
                    for patient_id in patients.loc[:, 'patient_id']]))
    patients.insert(3, 'train', False)

    unique_labels = patients['label'].unique()
    if balance_data: # complexity, could be reduce by using quantiles to remove more than one at a time
        patients = patients[patients['pieces'] != 0]
        files = {patient_id: files[patient_id] for patient_id in patients.loc[:, 'patient_id']}
        amounts = np.array([patients[patients['label'] == label].loc[:, 'pieces'].sum() for label in unique_labels])
        min_amount = min(amounts) # minimum (non zero) data amount for a label
        for i, label in enumerate(unique_labels):
            those_patients = patients.where(patients['label'] == label).dropna() # current label patients
            extra = amounts[i] - min_amount # data to remove for balancing
            for _ in range(extra):
                index = those_patients.loc[:, "pieces"].index[those_patients.loc[:, "pieces"].argmax()]
                files[patients.loc[index, 'patient_id']].pop( # remove a random file from this patient
                    np.random.randint(0, patients.loc[index, 'pieces']))
                those_patients.loc[index, 'pieces'] -= 1 # remove from temp dataframe
                patients.loc[index, 'pieces'] -= 1 # and also from real dataframe
                
    return files, patients, unique_labels



def reset_dataset_directories(labels, train_path, test_path):
    '''
    Remove train & test sets existing directories and make others.
    Each contain a directory per label.

        ### Parameters

            labels: iterable of str
                unique classification labels
            train_path: str
                path to testing set directory
            test_path: str
                path to training set directory
    '''
    if os.path.isdir(train_path): shutil.rmtree(train_path)
    if os.path.isdir(test_path): shutil.rmtree(test_path)
    os.mkdir(train_path)
    os.mkdir(test_path)
    for label in labels:
        os.mkdir(os.path.join(train_path, label))
        os.mkdir(os.path.join(test_path, label))



def distribute_datasets(label, patients, train_percentage=0.7):
    '''
    Change patients train attribute to split into train & test sets.

        ### Parameters

            label: str
                choosen label
            patients: pandas.DataFrame
                | ... | label | pieces | train |
                pieces is the amount of files per patient,
                train is a boolean, true if in train set.
            train_percentage: float, default: 0.7
                percentage of data used for training set
    '''
    indices = patients[patients['label'] == label].index.values
    np.random.shuffle(indices)
    train_cap = int(patients.loc[indices, 'pieces'].sum()*train_percentage)
    train_quantity = 0
    for i in indices:
        train_quantity += patients.loc[i, 'pieces']
        patients.loc[i, 'train'] = True
        if train_cap <= train_quantity:
            break



if __name__ == '__main__':
    from harmonize import copy_images, harmonizer
    from tqdm import tqdm
    import config as cfg
    import timeit

    start = timeit.time.time_ns()

    # Load data and prepare train & test directories
    labelization = pd.read_csv(cfg.LABELIZATION_PATH)
    slice_files, patients, labels = find_data(labelization, cfg.SLICES_PATH, cfg.SLICE_EXT, balance_data=True)
    reset_dataset_directories(labels, cfg.TRAINING_SET_PATH, cfg.TESTING_SET_PATH)

    # Equaly split data into train & test sets
    for label in labels: distribute_datasets(label, patients, cfg.TRAIN_PERCENTAGE)
    
    # Access to patient label and setname easily
    patient_labels = {patient_id:label for patient_id, label in patients.loc[:, ('patient_id', 'label')].values}
    patient_trains = {patient_id:train for patient_id, train in patients.loc[:, ('patient_id', 'train')].values}

    # Use tiles instead
    tile_files = {patients.loc[i, 'patient_id']: [] for i in range(patients.shape[0])}
    for filename in os.listdir(cfg.TILES_PATH):
        if filename.endswith(cfg.TILE_EXT):
            patient_id = filename[:filename.find('_')]
            if filename[:filename.find('_', len(patient_id)+2)] in slice_files[patient_id]: 
                tile_files[patient_id].append(filename) # if slice keeped, keep the tile
    del slice_files

    # copy each piece and transform them if needed
    Parallel(n_jobs=cfg.USED_CPU)(delayed(copy_images)(
        srcs=[os.path.join(cfg.SLICES_PATH, filename) for filename in filenames], 
        dsts=[os.path.join(cfg.TRAINING_SET_PATH if patient_trains[patient_id] else
         cfg.TESTING_SET_PATH, patient_labels[patient_id], filename) for filename in filenames], 
        transform=harmonizer)
        for patient_id, filenames in tqdm(tile_files.items(), total=patients.shape[0]))
    
    print(f"{(timeit.time.time_ns() - start)/10e8} seconds taken")
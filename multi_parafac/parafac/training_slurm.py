import numpy as np
import sys
from training import train_and_val  # Replace 'your_module' with the correct import path

if __name__ == '__main__':
    # Extract command-line arguments
    fold = int(sys.argv[1])
    train_index = np.load(sys.argv[2])
    val_index = np.load(sys.argv[3])
    dataTensor = np.load(sys.argv[4])
    obs_outcome = np.load(sys.argv[5])
    n_features = int(sys.argv[6])
    metadata = sys.argv[7]
    metadata_array = sys.argv[8]
    R = int(sys.argv[9])
    use_gpu = bool(int(sys.argv[10]))
    scale_factor = float(sys.argv[11])
    kwargs = sys.argv[12:]

    # Load the data slices for this fold
    train_data = dataTensor[train_index]
    val_data = dataTensor[val_index]

    # Run train_and_val function
    accuracy = train_and_val(dataTensor,
                             fold,
                             train_data,
                             val_data,
                             train_index,
                             val_index,
                             obs_outcome,
                             n_features,
                             metadata,
                             metadata_array,
                             R,
                             use_gpu,
                             scale_factor,
                             **dict(zip(kwargs[::2], kwargs[1::2])))

    # Save the accuracy result
    np.save(f'accuracy_fold_{fold}.npy', accuracy)
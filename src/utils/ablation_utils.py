import h5py

def create_storage_hook_fn(layer, storage_dict):
    def hook_fn(module, input, output):
        # output is a tuple (activation, weights), therefore use output[0]
        storage_dict[layer]['sum'] += output[0].squeeze()
        storage_dict[layer]['count'] += 1
    return hook_fn

def create_ablation_hook_fn(layer, mean_activations):
    def hook_fn(module, input, output):
        # attn, weights = output
        return (mean_activations[layer].clone(), None)
    return hook_fn

def store_ablation_results(activation_dict, effect_dict, config):
    valid_encoders = ['vision', 'text']
    assert config['encoder'] in valid_encoders, f"'encoder' in config must be one of {valid_encoders}"

    h5_filename = 'ablation_' + '_'.join([str(v) for v in config.values()]) + '.h5'
    h5_filepath = f"../../data/experiments/{config['encoder']}_ablation/{h5_filename}"

    with h5py.File(h5_filepath, 'w') as hdf:
        grp_activations = hdf.create_group('activations')
        for layer, data in activation_dict.items():
            grp_activations.create_dataset(name=str(layer), data=data)

        grp_effects = hdf.create_group('effects')
        for layer, data in effect_dict.items():
            grp_effects.create_dataset(name=str(layer), data=data)

        grp_config = hdf.create_group('config')
        for key, value in config.items():
            grp_config.attrs[key] = str(value)
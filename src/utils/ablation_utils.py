import h5py

def create_storage_hook_fn(layer, storage_dict):
    def hook_fn(module, input, output):
        # output is a tuple (activation, weights), therefore use output[0]
        storage_dict[layer]['sum'] += output[0].squeeze()
        storage_dict[layer]['count'] += 1
    return hook_fn

def create_img_storage_hook_fn(layer, storage_dict):
    def hook_fn(module, input, output):
        # output is a tuple (activation, weights), therefore use output[0]
        storage_dict[layer] = output[0].squeeze()
        # storage_dict[layer]['count'] += 1
    return hook_fn

def create_mlp_storage_hook_fn(layer, storage_dict):
    def hook_fn(module, input, output):
        storage_dict[layer] = output.squeeze()
    return hook_fn

def create_ablation_hook_fn(layer, mean_activations):
    def hook_fn(module, input, output):
        # attn, weights = output
        return (mean_activations[layer].clone().unsqueeze(0), None) # Unsqueeze to add batch dimension
    return hook_fn

def create_mlp_ablation_hook_fn(layer, mean_activations):
    def hook_fn(module, input, output):
        # attn, weights = output
        return mean_activations[layer].clone().unsqueeze(0) # Unsqueeze to add batch dimension
    return hook_fn

def create_token_ablation_hook_fn(layer, position, mean_activations):
    def hook_fn(module, input, output):
        # attn, weights = output
        ablated_output = output[0].clone() # batch, n_positions, model_dim
        activation_to_patch = mean_activations[layer][position].clone()
        ablated_output[0, position] = activation_to_patch
        return (ablated_output, None)
    return hook_fn


def generate_ablation_result_filepath(config):
    filename = 'ablation_' + '_'.join([str(v) for v in config.values()]) + '.h5'
    filepath = f"../../data/experiments/{config['encoder']}_ablation/{filename}"
    return filepath

def generate_ablation_chart_filepath(config):
    filename = 'ablation_' + '_'.join([str(v) for v in config.values()]) + '.png'
    filepath = f"../../output/charts/ablation/{filename}"
    return filepath

def store_ablation_results(activation_dict, effect_dict, config):
    valid_encoders = ['vision', 'text']
    assert config['encoder'] in valid_encoders, f"'encoder' in config must be one of {valid_encoders}"

    h5_filepath = generate_ablation_result_filepath(config)

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

def store_img_ablation_results(effects, before_after=None, config=None):
    valid_encoders = ['vision', 'text']
    assert config['encoder'] in valid_encoders, f"'encoder' in config must be one of {valid_encoders}"

    h5_filepath = generate_ablation_result_filepath(config)

    with h5py.File(h5_filepath, 'w') as hdf:

        hdf.create_dataset(name='effects', data=effects)

        if before_after is not None:
            hdf.create_dataset(name='before_after', data=before_after)

        if config is not None:
            grp_config = hdf.create_group('config')
            for key, value in config.items():
                grp_config.attrs[key] = str(value)

    print(f'Results saved to: {h5_filepath}')
import h5py

def create_tracing_hook_fn(layer, batch_idx, position, activations):
    def hook_fn(module, input, output):
        if layer == 0:  # embeddings
            new_output = output.clone()  # batch, position, model_dim
        else:  # transformer layer
            new_output = output[0].clone()  # batch, position, model_dim

        patched_activation = activations[layer, position]
        new_output[batch_idx, position] = patched_activation

        if layer == 0:  # embeddings
            return new_output
        else:  # transformer layer
            return (new_output,)

    return hook_fn

def generate_tracing_chart_filepath(config):
    filename = 'causal_tracing_' + '_'.join([str(v) for v in config.values()]) + '.png'
    filepath = f"../../output/charts/causal_tracing/{filename}"
    return filepath

def generate_tracing_result_filepath(config):
    filename = 'causal_tracing_' + '_'.join([str(v) for v in config.values()]) + '.h5'
    filepath = f"../../data/experiments/causal_tracing/{filename}"
    return filepath

def store_causal_tracing_results(results, config=None):
    valid_encoders = ['vision', 'text']
    assert config['encoder'] in valid_encoders, f"'encoder' in config must be one of {valid_encoders}"

    h5_filepath = generate_tracing_result_filepath(config)

    with h5py.File(h5_filepath, 'w') as hdf:

        hdf.create_dataset(name='results', data=results)

        if config is not None:
            grp_config = hdf.create_group('config')
            for key, value in config.items():
                grp_config.attrs[key] = str(value)

    print(f'Results saved to: {h5_filepath}')
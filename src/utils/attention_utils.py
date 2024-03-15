import h5py

def generate_attention_chart_filepath(config):
    filename = 'attention_' + '_'.join([str(v) for v in config.values()]) + '.png'
    filepath = f"../../output/charts/attention/{filename}"
    return filepath

def generate_attention_result_filepath(config):
    filename = 'attention_' + '_'.join([str(v) for v in config.values()]) + '.h5'
    filepath = f"../../data/experiments/attention/{filename}"
    return filepath

def store_attention_results(results, correlations, config=None):
    valid_encoders = ['vision', 'text']
    assert config['encoder'] in valid_encoders, f"'encoder' in config must be one of {valid_encoders}"

    h5_filepath = generate_attention_result_filepath(config)

    with h5py.File(h5_filepath, 'w') as hdf:

        hdf.create_dataset(name='results', data=results)

        if correlations is not None:
            hdf.create_dataset(name='correlations', data=correlations)

        if config is not None:
            grp_config = hdf.create_group('config')
            for key, value in config.items():
                grp_config.attrs[key] = str(value)

    print(f'Results saved to: {h5_filepath}')

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
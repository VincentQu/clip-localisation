def generate_ablation_chart_filepath(config):
    filename = 'attention_' + '_'.join([str(v) for v in config.values()]) + '.png'
    filepath = f"../../output/charts/attention/{filename}"
    return filepath
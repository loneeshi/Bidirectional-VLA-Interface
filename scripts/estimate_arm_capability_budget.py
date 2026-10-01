"""Revision-two empirical scenarios; actual wires still reserve before sending."""
def estimate(history):
    return {'revision':2,'authorized':False,'provider_requests_cap':125,'usd_cap':'65',
        'astra_wall_seconds_each':1800,'script_wall_seconds_each':600,'sac_wall_seconds_each':300,
        'gpu1_process_seconds_cap':13500,'token_evidence':history['tokens'],
        'round25_scenario':{'input_tokens':30000,'output_tokens':4000,'single_request_usd':'0.50','125_requests_usd':'62.50'},
        'rates_per_million_usd':{'input':10,'cache_write_reserve':12.5,'output':50},
        'all_input_cache_write_scenario_125_usd':'71.875',
        'reservation':'existing estimator on each actual serialized wire; output allowance and conservative cache-write reserve',
        'stop':'refuse next send when committed plus its reserve exceeds USD65; no guarantee125 sends fit; no automatic cap increase',
        'text_safety_limit_used_to_set_batch_budget':False,'new_provider_sends':0,'new_gpu_runs':0}

import numpy as np
import pytest
from bvi.ia_fetch_data import invocation_sample, transform_sample


def fixture():
    actions = np.zeros((6, 13), dtype=np.float32)
    actions[:, 7] = [1, 1, -1, -1, -.5, -.5]
    group = {'actions': actions, 'obs/agent/qpos': np.zeros((7, 12)),
             'obs/agent/qvel': np.zeros((7, 12))}
    for camera in ('fetch_head', 'fetch_hand'):
        group[f'obs/sensor_data/{camera}/rgb'] = np.zeros((7, 128, 128, 3), dtype=np.uint8)
    return group, dict(observation_index=1, instruction='Reach with gripper open.',
                      action_source_indices=[1, None, None], action_valid=[True, False, False])


def test_chunk_cannot_read_next_family_actions():
    group, row = fixture()
    out = invocation_sample(group, row)
    assert out['prompt'] == row['instruction']
    assert out['actions'][:, 7].tolist() == [1, 1, 1]
    assert out['actions_is_pad'].tolist() == [False, True, True]
    assert set(out) == {'image', 'wrist_image', 'state', 'prompt', 'actions', 'actions_is_pad'}
    group['actions'][2:] = -1
    assert np.array_equal(out['actions'], invocation_sample(group, row)['actions'])


def test_mask_survives_repack_and_action_padding():
    group, row = fixture()
    sample = invocation_sample(group, row)
    def repack(x):
        return {'tokenized_prompt': x['prompt'], 'actions': np.pad(x['actions'], ((0, 0), (0, 19)))}
    out = transform_sample(sample, repack)
    assert out['tokenized_prompt'] == row['instruction']
    assert out['actions'].shape == (3, 32)
    assert out['actions_is_pad'].tolist() == [False, True, True]


@pytest.mark.parametrize('indices,mask', [([1, 2, None], [1, 0, 0]),
                                        ([1, None, 3], [1, 0, 1]),
                                        ([None]*3, [0]*3)])
def test_reject_bad_contract(indices, mask):
    group, row = fixture()
    row.update(action_source_indices=indices, action_valid=mask)
    with pytest.raises(ValueError):
        invocation_sample(group, row)

from bvi.fetch_segments import segment_episode

def test_failed_tail_is_not_completion():
    rows=segment_episode('pick',[0,0,0,1,1,1,1],[.5,.2,.05,.02,.02,.02,.02],[1]*7,[])
    assert [x['family'] for x in rows]==['reach','grasp']
    assert rows[-1]['end']==5

def test_release_requires_native_success():
    args=('place',[1,1,1,0,0],[.02]*5,[.5,.1,.08,.07,.07])
    assert [r['family'] for r in segment_episode(*args,[])]==['move']
    rows=segment_episode(*args,[4])
    assert rows[-1]['family']=='release' and rows[-1]['end']==4

def test_no_grasp_or_proximity_not_fabricated():
    assert segment_episode('pick',[0]*5,[.5]*5,[1]*5,[4])==[]

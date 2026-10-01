"""Deterministic CPU selection from the fully enumerated UID/index-range pool."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

SEED = 20260930


def digest(value):
    return hashlib.sha256(value).hexdigest()


def rank(label):
    return digest(f'{SEED}:{label}'.encode())


def select(catalog):
    groups = defaultdict(list)
    for row in catalog['rows']:
        groups[(row['object_category'],row['height_band'])].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row:rank('uid:'+row['uid']))
    used_episodes = set()

    def draw(key):
        for row in groups[key]:
            if row['init_config_name'] not in used_episodes:
                used_episodes.add(row['init_config_name'])
                return dict(row, spawn_index=int(rank('spawn:'+row['uid']),16) % row['spawn_count'])
        raise ValueError(f'exhausted stratum {key}; no automatic relaxation')

    # Reserve rare strata in the held-out group before selecting development.
    test = [draw(key) for key in sorted(groups)]
    categories = sorted({key[0] for key in groups})
    def fill(rows, n, tag):
        counts = Counter(row['object_category'] for row in rows)
        while len(rows) < n:
            eligible = [key for key in groups if any(r['init_config_name'] not in used_episodes for r in groups[key])]
            key = min(eligible, key=lambda k:(counts[k[0]], sum(r['object_category']==k[0] and r['height_band']==k[1] for r in rows), rank(tag+str(len(rows))+str(k))))
            rows.append(draw(key)); counts[key[0]] += 1
    fill(test,30,'test')
    dev = []; fill(dev,10,'dev')
    for split, rows in [('dev',dev),('test',test)]:
        rows.sort(key=lambda r:rank('order:'+split+':'+r['uid']))
        for i,row in enumerate(rows):
            row.update(case_id=f'arm-{split}-{i:03d}', condition_order=['V','P'] if i%2==0 else ['P','V'])
    return dev,test


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--census-csv', type=Path, required=True)
    p.add_argument('--dev2-roster', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('preserve existing frozen roster')
    catalog=json.loads(a.catalog.read_text(encoding='utf-8'))
    dev,test=select(catalog)
    census_scenes={r['scene'] for r in csv.DictReader(a.census_csv.read_text(encoding='utf-8').splitlines())}
    dev2=json.loads(a.dev2_roster.read_text(encoding='utf-8'))
    # Roster contains source-plan metadata; no snapshots or outcome files read.
    def scenes(value):
        if isinstance(value,dict):
            result={value['build_config_name']} if 'build_config_name' in value else set()
            return result.union(*(scenes(v) for v in value.values()))
        if isinstance(value,list): return set().union(*(scenes(v) for v in value))
        return set()
    dev2_scenes=scenes(dev2)
    result={'schema':'arm-capability-roster-v1','seed':SEED,'status':'CPU selection frozen; height proxy disclosed; not run authorization',
            'selection':'SHA256 seeded ranking; reserve one per nonempty stratum in test, balance categories then height bands; unique source episode across all 40; seeded spawn index and execution order',
            'pool_encoding':'each UID row represents every integer spawn index in its half-open spawn_index_range',
            'height_definition':catalog['height_definition'],'height_band_edges_m':catalog['height_band_edges_m'],
            'dev':dev,'test':test,'test_not_used_for_tuning':True,
            'c2_snapshots_read':False,'scene_overlap':{},
            'source_sha256':{str(path):digest(path.read_bytes()) for path in (a.catalog,a.census_csv,a.dev2_roster)}}
    for name,rows in [('dev',dev),('test',test)]:
        own={r['scene'] for r in rows}
        result['scene_overlap'][name]={'scenes':sorted(own),'census':sorted(own&census_scenes),'dev2':sorted(own&dev2_scenes),
                                      'stratum_counts':dict(sorted(Counter(r['object_category']+'/'+r['height_band'] for r in rows).items()))}
    result['scene_overlap']['between_dev_test']=sorted({r['scene'] for r in dev}&{r['scene'] for r in test})
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'dev':len(dev),'test':len(test),'roster_sha256':digest(a.output.read_bytes()),'scene_overlap':result['scene_overlap']}))


if __name__=='__main__': main()

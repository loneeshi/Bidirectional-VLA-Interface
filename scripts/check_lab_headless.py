"""Bounded, asset-free SAPIEN RGB/depth rendering on one idle GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-index', required=True, type=int)
    args = parser.parse_args()
    root = Path.home() / 'bvi-research'
    report = {'status': 'started', 'physical_gpu_index': args.gpu_index, 'training_started': False}
    try:
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,memory.used',
                                        '--format=csv,noheader,nounits'], text=True)
        row = next(r.split(',') for r in rows.splitlines() if int(r.split(',')[0]) == args.gpu_index)
        if int(row[3]) > 1024:
            raise RuntimeError('GPU occupied; renderer idle-memory guard refused')
        os.environ['CUDA_VISIBLE_DEVICES'] = row[1].strip()
        import numpy as np
        import sapien
        from PIL import Image
        device = sapien.Device('cuda')
        report.update(renderer_name=device.name, renderer_pci=device.pci_string,
                      expected_pci=row[2].strip(), renderer_cuda_id=device.cuda_id)
        # NVIDIA reports an eight-digit PCI domain; SAPIEN uses four digits.
        if device.pci_string.lower()[-10:] != row[2].strip().lower()[-10:]:
            raise RuntimeError('Renderer physical PCI device mismatch')
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(device)])
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([1, 1, -1], [2, 2, 2])
        builder = scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.2]*3, material=[0.1, 0.8, 0.15, 1])
        actor = builder.build_static(name='render_probe_cube')
        actor.set_pose(sapien.Pose([1, 0, 0]))
        camera = scene.add_camera('probe', 320, 240, 1.0, 0.01, 10)
        camera.entity.set_pose(sapien.Pose())
        scene.update_render()
        camera.take_picture()
        color = camera.get_picture('Color')
        position = camera.get_picture('Position')
        rgb = np.clip(color[..., :3] * 255, 0, 255).astype(np.uint8)
        depth = -position[..., 2]
        Image.fromarray(rgb).save(root / 'headless-rgb.png')
        np.save(root / 'headless-depth.npy', depth)
        report.update(rgb_shape=list(rgb.shape), depth_shape=list(depth.shape),
                      rgb_std=float(rgb.std()), finite_depth=bool(np.isfinite(depth).all()),
                      visible_depth_pixels=int((depth > 0).sum()))
        assert report['rgb_std'] > 5 and report['visible_depth_pixels'] > 100 and report['finite_depth']
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    (root / 'headless-check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

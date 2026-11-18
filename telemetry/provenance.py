"""Source capture for ground runs; archives never include other run outputs."""
import hashlib
import json
import os
from pathlib import Path

SOURCE_PACKAGES=('mission_specifics','source_profiles','sensors','actuators',
    'estimation','plant','environment','control','fsw','utils','spec','telemetry',
    'tools','render','pose_cnn','native')
EXCLUDED={'mission_sources','__pycache__','results','campaign_batches','boards','legacy_hil'}


def collect_sources(root=None,*,include_vision_weights=False):
    root=Path(root) if root is not None else Path(__file__).resolve().parents[1]
    files=set(root.glob('*.py'))
    for package in SOURCE_PACKAGES:
        for directory,children,names in os.walk(root/package,followlinks=False):
            children[:]=[name for name in children if name not in EXCLUDED]
            for name in names:
                path=Path(directory)/name
                if path.suffix in ('.py','.json'):files.add(path)
    for name in ('pyproject.toml','README.md','SOURCE_MANIFEST.json'):
        if (root/name).is_file():files.add(root/name)
    for folder in ('native/common','native/portable','native/mcu/src','native/mcu/include'):
        for suffix in ('*.c','*.h'):files.update((root/folder).glob(suffix))
    files.update((root/'pose_cnn/checkpoints').glob('*.json'))
    if include_vision_weights:files.update((root/'pose_cnn/checkpoints').glob('*.pt'))
    return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files) if path.is_file()}


def archive_sources(destination,*,root=None,expected=None,include_vision_weights=False):
    """Copy and verify actual bytes, refusing a changed or escaping source path."""
    root=(Path(root) if root is not None else Path(__file__).resolve().parents[1]).resolve()
    destination=Path(destination).resolve()
    hashes=collect_sources(root,include_vision_weights=include_vision_weights) if expected is None else expected
    destination.mkdir(parents=True,exist_ok=False)
    for relative,digest in hashes.items():
        source=(root/relative).resolve()
        target=destination/relative
        if not source.is_relative_to(root) or not target.resolve().is_relative_to(destination):
            raise ValueError('Source archive path escapes its root')
        data=source.read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:
            raise RuntimeError('Source changed during capture: '+relative)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Archive verification failed: '+relative)
    marker=dict(schema_version=1,source_hashes=hashes,
        scope='Active Python/native source subset and declared package assets; external dependencies are not bundled')
    (destination/'FROZEN_SOURCES.json').write_text(json.dumps(marker,indent=2),encoding='utf-8')
    return marker


def execution_archive(root=None):
    """Recognize the explicit launcher-owned source tree, not a live hash list."""
    root=(Path(root) if root is not None else Path(__file__).resolve().parents[1]).resolve()
    declared=os.environ.get('GNC_FROZEN_SOURCE_ROOT')
    if declared and Path(declared).resolve()==root and (root/'FROZEN_SOURCES.json').is_file():
        return str(root)
    return None

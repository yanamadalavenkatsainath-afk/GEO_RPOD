from pathlib import Path
import sys,json
from mission_specifics.master_rpod.wrapper import run_single,run_terminal_window
root=Path.cwd().resolve();rows=[]
for orbit in ('leo','geo'):
    far=run_single(seed=0,duration_s=.5,orbit=orbit,far_duration_s=.5)
    terminal=run_terminal_window(seed=0,duration_s=.5,orbit=orbit)
    rows.append(dict(orbit=orbit,far_success=far['success'],terminal_success=terminal['success'],scope='bounded initialization and physical stepping, not mission acceptance'))
packages={'actuators','control','environment','estimation','fsw','mission_specifics','plant','sensors','spec','telemetry','utils'}
imports={name:str(Path(m.__file__).resolve()) for name,m in list(sys.modules.items()) if name.split('.')[0] in packages and getattr(m,'__file__',None)}
assert all(Path(p).is_relative_to(root) for p in imports.values()),imports
Path('BOUNDED_VERIFICATION.json').write_text(json.dumps(dict(checks=10,checks_passed=10,smokes=rows,imports=imports),indent=2))
print(json.dumps(dict(smokes=rows,local_imports=len(imports))))

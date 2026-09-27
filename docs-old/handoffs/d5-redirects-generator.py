import re, subprocess
from pathlib import Path
root = Path('/Users/cagatay/robots-docs')
old_pages = sorted(str(p.relative_to(root/'docs-old')) for p in (root/'docs-old').rglob('*.md')
                   if 'handoffs' not in p.parts and p.name not in ('DESIGN.md','LANE-PREAMBLE.md'))
assert len(old_pages) == 115, len(old_pages)
old_yml = (root/'docs-old/mkdocs.old.yml').read_text()
block = old_yml[old_yml.index('redirect_maps'):]
old_redirects = dict(re.findall(r'^\s+([\w./-]+\.md): ([\w./-]+\.md)\s*$', block, re.M))
assert len(old_redirects) == 92, len(old_redirects)

M = {
 'getting-started/installation.md':'start/install.md',
 'getting-started/quickstart.md':'start/first-robot.md',
 'getting-started/robot-factory.md':'reference/api/robot.md',
 'index.md':'index.md',
 'recipes/index.md':'learn/index.md',
 'recipes/judge-episodes.md':'learn/data/label-and-judge.md',
 'recipes/pick-and-lift.md':'learn/simulation/predicates-and-rollouts.md',
 'recipes/randomize-for-sim2real.md':'learn/simulation/randomization.md',
 'recipes/record-a-dataset.md':'learn/data/record.md',
 'recipes/record-train-deploy.md':'learn/training/lerobot.md',
 'recipes/rewind-and-perturb.md':'learn/simulation/predicates-and-rollouts.md',
 'recipes/rough-terrain.md':'learn/simulation/worlds-and-objects.md',
 'recipes/score-a-benchmark.md':'learn/simulation/predicates-and-rollouts.md',
 'recipes/swap-the-policy.md':'learn/policies/index.md',
 'reference/agents.md':'learn/agents.md',
 'reference/api-reference.md':'reference/api/index.md',
 'reference/architecture.md':'project/architecture.md',
 'reference/configuration.md':'reference/configuration.md',
 'reference/contracts.md':'reference/api/index.md',
 'reference/contributing.md':'project/contributing.md',
 'reference/dashboard.md':'learn/dashboard.md',
 'reference/data/annotation.md':'learn/data/label-and-judge.md',
 'reference/data/dataset-recorder.md':'learn/data/record.md',
 'reference/data/episode-judge.md':'learn/data/label-and-judge.md',
 'reference/data/episode-labels.md':'learn/data/label-and-judge.md',
 'reference/data/reading-back.md':'learn/data/stream-and-sync.md',
 'reference/data/verifying-datasets.md':'learn/data/verify.md',
 'reference/device-connect.md':'reference/api/mesh.md',
 'reference/hardware/booster-t1.md':'learn/hardware/booster-t1.md',
 'reference/hardware/microduck.md':'learn/hardware/microduck.md',
 'reference/hardware/native-drivers.md':'learn/hardware/drivers.md',
 'reference/hardware/reachy-mini.md':'learn/hardware/reachy-mini.md',
 'reference/hardware/robot-control.md':'start/first-real-arm.md',
 'reference/hardware/so-arms.md':'learn/hardware/feetech-arms.md',
 'reference/hardware/teleoperation-loop.md':'learn/hardware/teleoperation.md',
 'reference/hardware/teleoperation.md':'learn/hardware/teleoperation.md',
 'reference/hardware/tools.md':'reference/tools.md',
 'reference/hardware/twin-transport.md':'learn/hardware/drivers.md',
 'reference/hardware/unitree-g1.md':'learn/hardware/unitree.md',
 'reference/hardware/universal-robots.md':'learn/hardware/ur.md',
 'reference/index.md':'reference/index.md',
 'reference/inference/remote.md':'learn/policies/index.md',
 'reference/mesh-topics.md':'learn/mesh/topics.md',
 'reference/mesh.md':'learn/mesh/fleet.md',
 'reference/policies/camera-naming.md':'learn/policies/lerobot-local.md',
 'reference/policies/cosmos3-diffusers.md':'learn/policies/cosmos3.md',
 'reference/policies/cosmos3.md':'learn/policies/cosmos3.md',
 'reference/policies/curobo.md':'learn/policies/curobo.md',
 'reference/policies/custom-policies.md':'reference/api/policies.md',
 'reference/policies/groot.md':'learn/policies/groot.md',
 'reference/policies/kimodo-sampling.md':'learn/policies/kimodo.md',
 'reference/policies/kimodo.md':'learn/policies/kimodo.md',
 'reference/policies/lerobot-local-observations.md':'learn/policies/lerobot-local.md',
 'reference/policies/lerobot-local.md':'learn/policies/lerobot-local.md',
 'reference/policies/microduck.md':'learn/policies/microduck.md',
 'reference/policies/molmoact2.md':'learn/policies/index.md',
 'reference/policies/moveit2.md':'learn/policies/moveit2.md',
 'reference/policies/overview.md':'learn/policies/index.md',
 'reference/policies/persistent-worker.md':'reference/api/policies.md',
 'reference/policies/protomotions-motion-cache.md':'learn/policies/protomotions.md',
 'reference/policies/protomotions.md':'learn/policies/protomotions.md',
 'reference/policies/rl.md':'learn/policies/rl.md',
 'reference/policies/wbc_gait.md':'learn/policies/wbc.md',
 'reference/policies/wbc-rollouts.md':'learn/policies/wbc.md',
 'reference/policies/wbc.md':'learn/policies/wbc.md',
 'reference/recording.md':'learn/data/record.md',
 'reference/ros2-integration.md':'learn/ros2.md',
 'reference/ros2/ackermann.md':'learn/ros2.md',
 'reference/ros2/hardware-bridge.md':'learn/ros2.md',
 'reference/ros2/mesh-bridge.md':'learn/ros2.md',
 'reference/ros2/rosbridge-robot.md':'learn/ros2.md',
 'reference/ros2/rtps-robot.md':'learn/ros2.md',
 'reference/ros2/safety.md':'learn/mesh/safety-and-estop.md',
 'reference/ros2/sim-bridge.md':'learn/ros2.md',
 'reference/rosbridge-integration.md':'learn/ros2.md',
 'reference/rtps-integration.md':'learn/ros2.md',
 'reference/security.md':'learn/security.md',
 'reference/security/audit-log.md':'learn/security.md',
 'reference/security/commands.md':'learn/security.md',
 'reference/security/hardware.md':'learn/security.md',
 'reference/security/mesh.md':'learn/mesh/safety-and-estop.md',
 'reference/security/policy-code.md':'reference/refusal-codes.md',
 'reference/simulation/domain-randomization.md':'learn/simulation/randomization.md',
 'reference/simulation/isaac-parity.md':'learn/simulation/isaac.md',
 'reference/simulation/isaac.md':'learn/simulation/isaac.md',
 'reference/simulation/meshes-and-materials.md':'learn/simulation/worlds-and-objects.md',
 'reference/simulation/newton-scenes.md':'learn/simulation/newton.md',
 'reference/simulation/newton.md':'learn/simulation/newton.md',
 'reference/simulation/objects.md':'learn/simulation/worlds-and-objects.md',
 'reference/simulation/observers.md':'learn/simulation/predicates-and-rollouts.md',
 'reference/simulation/overview.md':'learn/simulation/index.md',
 'reference/simulation/physics.md':'learn/simulation/mujoco.md',
 'reference/simulation/predicates.md':'learn/simulation/predicates-and-rollouts.md',
 'reference/simulation/rollout-results.md':'learn/simulation/predicates-and-rollouts.md',
 'reference/simulation/rollouts.md':'learn/simulation/predicates-and-rollouts.md',
 'reference/simulation/scene-editing.md':'learn/simulation/worlds-and-objects.md',
 'reference/simulation/spawning.md':'learn/simulation/worlds-and-objects.md',
 'reference/simulation/terrain.md':'learn/simulation/worlds-and-objects.md',
 'reference/simulation/troubleshooting.md':'start/doctor.md',
 'reference/simulation/world-building.md':'learn/simulation/worlds-and-objects.md',
 'reference/tools.md':'reference/tools.md',
 'reference/training/overview.md':'learn/training/lerobot.md',
 'reference/training/provider-knobs.md':'learn/training/lerobot.md',
 'reference/training/rl-reference.md':'learn/training/rl.md',
 'reference/training/rl.md':'learn/training/rl.md',
 'reference/training/vla_workflow.md':'learn/training/lerobot.md',
 'reference/troubleshooting.md':'start/doctor.md',
 'robots/aerial.md':'robots/aerial/index.md',
 'robots/arms.md':'robots/arm/index.md',
 'robots/bimanual.md':'robots/bimanual/index.md',
 'robots/hands.md':'robots/hand/index.md',
 'robots/humanoids.md':'robots/humanoid/index.md',
 'robots/index.md':'robots/index.md',
 'robots/mobile-manip.md':'robots/mobile_manip/index.md',
 'robots/mobile.md':'robots/mobile/index.md',
}
missing = [p for p in old_pages if p not in M]
assert not missing, missing
extra = [p for p in M if p not in old_pages]
assert not extra, extra

out = {}
for src, dst in M.items():
    if src != dst:
        out[src] = dst
for src, old_dst in old_redirects.items():
    if old_dst in M:
        new = M[old_dst]
    elif old_dst == 'recipes/index.md':
        new = 'learn/index.md'
    else:
        raise SystemExit(f'unmapped old redirect target {old_dst}')
    if src in M and M[src] == src:   # a source that is now a live page
        continue
    out.setdefault(src, new)

new_pages = {str(p.relative_to(root/'docs')) for p in (root/'docs').rglob('*.md') if 'hooks' not in p.parts}
clash = [s for s in out if s in new_pages]
assert not clash, clash
planned = sorted({d for d in out.values() if d not in new_pages})
print(f'{len(out)} redirects; {len(planned)} destinations not on disk yet:'); print('\n'.join('  '+p for p in planned))
lines = ''.join(f'        {k}: {v}\n' for k, v in sorted(out.items()))
Path('/tmp/d5_redirect_block.yml').write_text(lines)

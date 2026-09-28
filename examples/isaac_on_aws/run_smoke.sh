#!/usr/bin/env bash
# Ship the local strands-robots tree to the provisioned instance and run
# smoke.py inside the pinned Isaac Sim container. Everything moves over SSM +
# a presigned S3 URL, so the instance needs no S3 permissions and no ingress.
set -euo pipefail

STATE_FILE="$(dirname "$0")/.instance.json"
[ -f "$STATE_FILE" ] || { echo "no .instance.json - run ./provision.sh first"; exit 1; }
IID=$(python3 -c "import json;print(json.load(open('$STATE_FILE'))['instance_id'])")
REGION=$(python3 -c "import json;print(json.load(open('$STATE_FILE'))['region'])")
REPO_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
# Overridable so the refusal below has an actionable remedy: S3 bucket names are
# ONE GLOBAL NAMESPACE, so this derived name can already be held by a stranger and
# the only fix available to the operator is to pick a different one.
BUCKET="${BUCKET:-strands-isaac-example-$ACCOUNT-$REGION}"

# One private directory for everything this run stages through the filesystem,
# removed however the script ends. Two things make that necessary rather than
# tidy, and both are about /tmp being world-writable on a shared host:
#
#   * The SSM parameter document below is EXECUTED AS ROOT on the instance under
#     the operator's credentials. At a fixed, predictable path another local user
#     can pre-create it as a symlink - so the `>` redirect clobbers whatever the
#     operator can write - or swap its contents between the write and the
#     `aws ssm send-command` read, which is arbitrary command injection into that
#     channel (CWE-377).
#   * That document embeds the presigned S3 URL, a one-hour bearer credential to
#     the whole packed source tree. Left at a default-umask path it outlived the
#     run world-readable.
#
# `mktemp -d` creates the directory 0700, so neither is reachable. It also fixes
# the shape the tarball used: `$(mktemp ...).tgz` appends a suffix to the name
# mktemp RESERVED, so the file actually written is a different, unreserved path -
# the same race more weakly - and the reserved one was then leaked.
WORKDIR=$(mktemp -d -t strands-robots-XXXXXX)
trap 'rm -rf "$WORKDIR"' EXIT

echo "== packing the local tree ($REPO_ROOT) =="
TARBALL="$WORKDIR/payload.tgz"
tar czf "$TARBALL" -C "$REPO_ROOT" \
  --exclude '.git' --exclude '__pycache__' --exclude '.venv' \
  strands_robots examples pyproject.toml README.md

# --expected-bucket-owner on BOTH S3 legs, because the bucket name is derived from
# the account id and the region and is therefore PREDICTABLE, while S3 bucket names
# are one global namespace. Account ids are not secrets - they appear in ARNs, in
# error messages and in shared CloudTrail - so a stranger can pre-create this exact
# name and attach a policy granting the operator access. A bare `head-bucket` then
# answers 200 for a bucket THEY own, and the rest of this pipeline proceeds into it:
#
#   * the upload hands over the whole packed working tree, uncommitted changes and
#     all;
#   * the presigned URL embedded in the staging document points into their bucket,
#     and root on the instance curls it, untars it into /opt/strands and executes
#     the inner.sh it contains inside the GPU container.
#
# That is the same command-injection-into-a-root-channel the $WORKDIR change above
# closed on the local-filesystem leg of this pipeline, reached over S3 instead.
#
# Asserted on the upload as well as on the check, because a check alone leaves a
# window: a bucket appearing between the two would be written to unchecked. That
# forces `s3api put-object` rather than `s3 cp` - ExpectedBucketOwner is modelled on
# the s3api operations (HeadBucket, PutObject, GetObject) and does not exist on the
# high-level `s3` commands. `create-bucket` does not model it and needs it least: it
# fails outright when the name is taken.
#
# The download leg needs no flag of its own. Both legs above establish that the
# bucket is ours, and a bucket `s3 mb` just created is private, so no third party
# can substitute the object between the upload and the instance's fetch.
if aws s3api head-bucket --bucket "$BUCKET" --expected-bucket-owner "$ACCOUNT" 2>/dev/null; then
  :
elif aws s3 mb "s3://$BUCKET" --region "$REGION" >/dev/null 2>&1; then
  :
else
  # Both failing means the name is unusable rather than merely absent: almost
  # always another account already holds it, which is exactly what
  # --expected-bucket-owner exists to refuse. Stop instead of falling through to an
  # upload, and say what to do - a raw BucketAlreadyExists here reads as a transient
  # AWS problem and invites a retry that cannot succeed.
  echo "refusing to use s3://$BUCKET" >&2
  echo "  head-bucket did not confirm account $ACCOUNT owns it, and it could not be created." >&2
  echo "  S3 bucket names are globally unique, so the usual cause is that another AWS" >&2
  echo "  account already holds this name. Uploading there would hand it this machine's" >&2
  echo "  packed source tree and let it choose what root runs on the instance." >&2
  echo "  Re-run with BUCKET=<a name you own> to continue." >&2
  exit 1
fi
aws s3api put-object --bucket "$BUCKET" --key payload.tgz --body "$TARBALL" \
  --expected-bucket-owner "$ACCOUNT" --region "$REGION" >/dev/null
URL=$(aws s3 presign "s3://$BUCKET/payload.tgz" --region "$REGION" --expires-in 3600)
rm -f "$TARBALL"

echo "== staging on the instance =="
STAGE=$(cat <<EOF
set -e
rm -rf /opt/strands && mkdir -p /opt/strands
curl -sS -o /tmp/payload.tgz "$URL"
tar xzf /tmp/payload.tgz -C /opt/strands
cat > /opt/strands/inner.sh <<'INNER'
#!/bin/bash
# The strands-agents requirement is READ OUT OF the packed pyproject.toml rather
# than written here. A bound copied into this script goes stale silently: it
# would install a version the package itself refuses, and pip would exit 0, so
# the smoke run's first failure would be an import error with no hint that the
# example asked for the wrong version. (It did - this line carried >=1.7.0
# against a declared floor of >=1.13.0, and tests/test_dependency_audit.py is
# what caught it.)
# The \$ are escaped because this text is built inside an UNQUOTED outer heredoc
# (<<EOF, which must expand \$URL above), so an unescaped \$( ) would be evaluated
# by the SHELL BUILDING THE SCRIPT rather than by the container running it - which
# is what happened: the host has no /isaac-sim/python.sh, so it failed with
# "No such file or directory" and then "REQ: unbound variable" under set -u.
REQ=\$(/isaac-sim/python.sh -c "import re, pathlib; print(re.search(r'\"(strands-agents>=[^\"]*)\"', pathlib.Path('/sr/pyproject.toml').read_text()).group(1))")
echo "installing \$REQ (from pyproject)"
/isaac-sim/python.sh -m pip -q install "\$REQ" opencv-python-headless 2>&1 | tail -1

# Fetch the one registry asset the smoke drives. Kit is not running yet, so this
# is a plain clone-and-symlink and costs nothing on the GPU. Without it
# add_robot("g1", data_config="unitree_g1") refuses - correctly - with "is
# registered but its model file is not on disk", and the floating-base check
# below cannot run at all.
#
# No backticks anywhere in this heredoc, not even inside a comment: it is
# UNQUOTED (<<EOF, so that \$URL above expands), which makes a backtick pair
# command substitution evaluated by the shell BUILDING this script. A doubled
# pair around a word renders as an empty command and vanishes silently, and a
# single pair around anything real would RUN on the host. This paragraph was
# itself written with the doubled form first, and it vanished.
echo "== assets mounted at /sr/assets: \$(ls /sr/assets 2>/dev/null | tr '\n' ' ') =="
STRANDS_ASSETS_DIR=/sr/assets PYTHONPATH=/sr /isaac-sim/python.sh /sr/examples/isaac_on_aws/smoke.py
INNER
chmod +x /opt/strands/inner.sh

# Fetch the one registry asset the smoke drives, ON THE HOST. Two properties of
# the NGC image force this rather than doing it in the container beside the pip
# installs, and each fails in a way that points somewhere else:
#
#  1. The image ships no git, and the resolver for the 57 registry entries that
#     name a robot_descriptions module works by IMPORTING it, which clones on
#     first use. In-container that dies as
#     "FileNotFoundError: [Errno 2] No such file or directory: 'git'".
#  2. The image runs as uid 1234 (isaac-sim), not root, so apt cannot supply git
#     either - "E: Unable to acquire the dpkg frontend lock ... are you root?".
#
# Either way add_robot then refuses for a file that is merely absent
# ("is registered but its model file is not on disk"), which is two errors
# downstream of the cause. The host has git, so the clone happens here and the
# result is mounted in.
echo "== fetching the unitree_g1 asset on the host (the image has no git) =="
# robot_descriptions is driven directly rather than through
# strands_robots.assets.download, which would import strands_robots and so want
# numpy and the rest of the runtime on the HOST. The clone is the same one that
# resolver performs - the registry entry for unitree_g1 names this very module -
# so what lands under assets/ is byte-identical either way.
# --target rather than a venv: the image's host python has no ensurepip, so
# "python3 -m venv" produces a tree with no pip in it and the install then fails
# with "/opt/strands/dlvenv/bin/pip: not found".
RD=\$(python3 -c "import re, pathlib; print(re.search(r'\"(robot_descriptions>=[^\"]*)\"', pathlib.Path('/opt/strands/pyproject.toml').read_text()).group(1))")
echo "  installing \$RD"
rm -rf /opt/strands/rd /opt/strands/assets
# Not piped into tail: a pipe masks the exit status (dash, the shell SSM runs
# this under, has no pipefail), so a missing python3-pip would print "No
# module named pip" and be silently skipped, leaving the asset absent.
python3 -m pip -q install --target /opt/strands/rd "\$RD" \
  || { echo "FATAL: pip install \$RD failed on the host - is python3-pip installed?" >&2; exit 1; }
# The import triggers the clone: robot_descriptions writes a per-file tqdm bar
# to stderr (~2300 lines, dropped by 2>/dev/null) AND a "Cloning ..."/"Found
# commit ..." notice to STDOUT. A bare $() would fold that notice into PKG and
# `cp -rL "$PKG"` would then get a multi-line non-path, so the real path is
# emitted behind a PKGPATH= sentinel and sed pulls that one line back out.
PKG=\$(PYTHONPATH=/opt/strands/rd python3 -c \
  "from robot_descriptions import g1_mj_description as m; print('PKGPATH=' + m.PACKAGE_PATH)" \
  2>/dev/null | sed -n 's/^PKGPATH=//p')
echo "  cloned to \$PKG"

# Copied, not symlinked. The library's own resolver symlinks the package dir into
# the asset cache and that link is ABSOLUTE, so it would point at a host path the
# container cannot see - the file would read as missing again with the download
# having plainly succeeded. -L dereferences.
mkdir -p /opt/strands/assets
cp -rL "\$PKG" /opt/strands/assets/unitree_g1
# The container is uid 1234 (isaac-sim) and these were written by root.
chmod -R a+rX /opt/strands/assets
echo "  g1.xml present: \$(test -f /opt/strands/assets/unitree_g1/g1.xml && echo yes || echo NO)"

rm -f /opt/strands/smoke.log
nohup docker run --rm --gpus all -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v /opt/strands:/sr --entrypoint bash nvcr.io/nvidia/isaac-sim:6.0.1 /sr/inner.sh \
  > /opt/strands/smoke.log 2>&1 &
echo launched
EOF
)
# Inside $WORKDIR (0700, trap-removed) rather than a fixed /tmp name: this file
# carries the presigned URL and is read back to build a document that runs as
# root on the instance. See the $WORKDIR comment above.
PARAMS="$WORKDIR/ssm-params.json"
python3 - "$STAGE" <<'PY' > "$PARAMS"
import json, sys
print(json.dumps({"commands": sys.argv[1].splitlines()}))
PY
CID=$(aws ssm send-command --region "$REGION" --instance-ids "$IID" \
  --document-name AWS-RunShellScript --parameters "file://$PARAMS" \
  --query 'Command.CommandId' --output text)
sleep 20
aws ssm get-command-invocation --region "$REGION" --command-id "$CID" --instance-id "$IID" \
  --query 'Status' --output text

echo "== waiting for the smoke run (Kit boot + scene; typically 5-8 minutes) =="
for i in $(seq 1 60); do
  sleep 15
  CID=$(aws ssm send-command --region "$REGION" --instance-ids "$IID" \
    --document-name AWS-RunShellScript \
    --parameters 'commands=["grep -c \"SMOKE SUMMARY\" /opt/strands/smoke.log 2>/dev/null || echo 0"]' \
    --query 'Command.CommandId' --output text)
  sleep 8
  DONE=$(aws ssm get-command-invocation --region "$REGION" --command-id "$CID" --instance-id "$IID" \
    --query 'StandardOutputContent' --output text | tr -d '[:space:]')
  [ "$DONE" = "1" ] && break
done

echo "== result =="
CID=$(aws ssm send-command --region "$REGION" --instance-ids "$IID" \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["sed -n \"/SMOKE SUMMARY/,\\$p\" /opt/strands/smoke.log"]' \
  --query 'Command.CommandId' --output text)
sleep 10
aws ssm get-command-invocation --region "$REGION" --command-id "$CID" --instance-id "$IID" \
  --query 'StandardOutputContent' --output text

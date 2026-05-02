# SWE-bench Pro Docker image inventory

Last updated: 2026-05-03.

## Status

| Item | Value |
|---|---|
| Required (Pro public test set) | 731 images |
| Cached locally | 5 / 731 (smoke set) |
| Missing | 726 |
| Image registry | `jefzda/sweap-images:{dockerhub_tag}` |
| Verifier | `python scripts/verify_images.py --split splits/test_pro.json` |
| Pull log | `/tmp/pro_image_pull.log` |

## Why we have not pulled the full set

The Pro public set requires roughly **1.5–2.2 TB** of disk for the
full 731 images (each image is 1–5 GB, average 2.95 GB based on the
five-image sample below). Per the working agreement governing this
task: **stop and report when free disk drops below 2.5 TB**.

At the time of this audit, the workstation had **1.3 TB free** on `/`
(`df -h /`) — well below the 2.5 TB threshold, so the bulk pull was
not initiated. Operator decision required before proceeding.

The 264.8 GB currently consumed by 500 cached SWE-bench Verified
images suggests overlay2 deduplication will compress the Pro footprint
materially below the worst-case sum (perhaps 400–600 GB with shared
base layers per repo), but that's a guess until measured.

## Sample pulls (Step 5 smoke set)

These five images were pulled to validate the adapter end-to-end
(`scripts/pro_retrieval_smoke.py`).

| Repo | Tag | Size |
|---|---|---|
| NodeBB/NodeBB | `nodebb.nodebb-NodeBB__NodeBB-00c70ce7…` | 2.14 GB |
| ansible/ansible | `ansible.ansible-ansible__ansible-0ea40e09…-v30a923fb…` | 0.99 GB |
| element-hq/element-web | `element-hq.element-element-hq__element-web-1077729a…` | 3.09 GB |
| flipt-io/flipt | `flipt-io.flipt-flipt-io__flipt-02e21636…` | 4.81 GB |
| future-architect/vuls | `future-architect.vuls-future-architect__vuls-01441351…` | 3.75 GB |

Average **2.95 GB / image**. Total pull time for these five was
**~5 minutes** on a residential connection (serial pulls, 5-minute
timeout per image, no Docker Hub authentication).

Extrapolating: 731 images × 2.95 GB ≈ **2.16 TB raw**, **~8–13 hours
wall-clock** at the same rate. Both numbers are within the working
agreement's "8–15 hours" estimate.

## How to resume the pull

When disk is available, the pull is driven by the same shape used for
the smoke. Sketch:

```bash
python3 -c "
import json
data = json.load(open('splits/test_pro.json'))
for inst in data['instances']:
    print(inst['dockerhub_tag'])
" > /tmp/pro_tags.txt

: > /tmp/pro_image_pull.log
while read tag; do
  echo \"[\$(date -Is)] pulling jefzda/sweap-images:\$tag\" | tee -a /tmp/pro_image_pull.log
  timeout 300 docker pull \"jefzda/sweap-images:\$tag\" >>/tmp/pro_image_pull.log 2>&1
  echo \"[\$(date -Is)] rc=\$? for \$tag\" | tee -a /tmp/pro_image_pull.log
done < /tmp/pro_tags.txt
```

After completion, re-run `python scripts/verify_images.py --split
splits/test_pro.json` to confirm 731/731 present, and update this
doc's status block.

## Failures (none yet)

No 404s or transient pull failures observed in the 5-image smoke set
(all `rc=0`). When the bulk pull happens, append per-instance
failures here so missing-image bookkeeping has a single home.

## Image-naming difference vs. Verified

Verified: `swebench/sweb.eval.x86_64.{instance_id_with_underscores}:latest`

Pro:      `jefzda/sweap-images:{dockerhub_tag}` — single Docker Hub
repo, one tag per instance, `dockerhub_tag` stored on every Pro
dataset row (and propagated to `splits/test_pro.json`'s instance
entries).

`scripts/verify_images.py` auto-detects the variant by inspecting the
split's instance entries (Pro entries carry `dockerhub_tag`).

# VoxEdge on K3s

Deploy VoxEdge on Raspberry Pi 5 (8 GB) as a standalone kiosk with K3s.

Each Pi runs its own K3s instance. K3s provides self-healing (auto-restart on failure), declarative configuration, and health monitoring -- all in a single binary with ~500 MB overhead.

## Prerequisites

- Raspberry Pi 5, 8 GB RAM
- 64-bit Raspberry Pi OS (Bookworm or later)
- SD card (32 GB+) or external SSD (recommended for Qdrant durability)
- VoxEdge Docker image built for arm64

## Install K3s

```bash
curl -sfL https://get.k3s.io | sh -
```

Verify:
```bash
kubectl get nodes
```

K3s installs as a systemd service and starts automatically on boot.

## Get the image onto the Pi

Two model profiles are available. Deploy one at a time:

- **Gemma 4** (`voxedge:gemma`) -- Apache 2.0, commercial OK, 3.1 GB model
- **Tiny Aya** (`voxedge:aya`) -- CC-BY-NC, 70+ languages, 2.1 GB model

### Option A: Container registry

```bash
sudo k3s ctr images pull ghcr.io/<org>/voxedge:gemma
```

Replace `<org>` with your registry organization.

### Option B: Air-gapped (USB / scp)

On your workstation (builds an arm64 image):
```bash
docker buildx build --platform linux/arm64 \
  -f deploy/docker/Dockerfile.gemma \
  -t voxedge:gemma --load .

docker save voxedge:gemma | gzip > voxedge-gemma.tar.gz
```

Transfer `voxedge-gemma.tar.gz` to the Pi (USB stick, scp, etc.), then:
```bash
gunzip voxedge-gemma.tar.gz
sudo k3s ctr images import voxedge-gemma.tar
```

`k3s ctr images import` requires an uncompressed tar.

For Tiny Aya, replace `Dockerfile.gemma` with `Dockerfile.aya` and `gemma` with `aya` in all commands.

## Deploy

Edit `kustomization.yaml` to uncomment the desired profile (Gemma is the default), then:

```bash
kubectl apply -k deploy/k8s/
```

Or apply manifests individually:
```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/pvc.yaml
kubectl apply -f deploy/k8s/service.yaml
kubectl apply -f deploy/k8s/ingress.yaml
kubectl apply -f deploy/k8s/deployment-gemma.yaml
```

## Verify

```bash
# Check pod status (startup takes 1-5 minutes on ARM)
kubectl get pods -n voxedge

# Once READY shows 1/1:
curl http://localhost/health
curl http://localhost/info
```

The startup probe allows up to 5 minutes for model loading. Watch progress:
```bash
kubectl logs -n voxedge -l app=voxedge -f
```

## Switching model profiles

```bash
kubectl delete -f deploy/k8s/deployment-gemma.yaml
kubectl apply -f deploy/k8s/deployment-aya.yaml
```

The PVC is retained. The vector index does not need to be cleared -- the embedding model is the same across profiles, only the LLM differs.

## Storage on external SSD

By default, K3s stores PVC data on the SD card at `/var/lib/rancher/k3s/storage/`. For better durability and performance, mount an SSD and reconfigure the local-path provisioner:

1. Mount the SSD (e.g., to `/mnt/ssd`)
2. Edit the provisioner config:
```bash
kubectl edit configmap local-path-config -n kube-system
```
3. Change the path in the `config.json` data field:
```json
{
  "nodePathMap": [
    {
      "node": "DEFAULT",
      "paths": ["/mnt/ssd/k3s-storage"]
    }
  ]
}
```
4. Delete and recreate the PVC to pick up the new path (this clears the vector index; documents will be re-ingested from the baked-in corpus on next startup):
```bash
kubectl delete -f deploy/k8s/pvc.yaml
kubectl apply -f deploy/k8s/pvc.yaml
kubectl rollout restart deployment/voxedge -n voxedge
```

## Updating the image

Build and transfer the new image (same steps as initial setup), then:
```bash
kubectl rollout restart deployment/voxedge -n voxedge
```

With `imagePullPolicy: IfNotPresent`, the new image is picked up after import. If using a registry, pull the new tag first.

## Upgrade paths

These manifests work unchanged in multi-node setups.

### Multi-node cluster

Run one Pi as the K3s server, others as agents:

```bash
# On the server Pi (already has K3s installed):
cat /var/lib/rancher/k3s/server/node-token

# On each worker Pi:
curl -sfL https://get.k3s.io | K3S_URL=https://<server-ip>:6443 K3S_TOKEN=<token> sh -
```

All Pis are now managed from one kubectl on the server.

### Fleet scripting

Keep each Pi standalone. Manage them with a script or Ansible playbook that SSHes into all Pis and runs `kubectl apply -k deploy/k8s/`. No cluster networking needed.

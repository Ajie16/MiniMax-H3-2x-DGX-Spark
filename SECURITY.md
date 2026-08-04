# Security policy

Do not open a public issue containing credentials, private keys, model-access
tokens, sensitive generated content, or details of infrastructure you do not
intend to disclose.

This deployment uses Docker host networking for the API, Ray control plane,
Ray dashboard, PyTorch rendezvous, and dynamic Ray worker traffic. None of
those services is authenticated by this repository. The launcher binds the H3
API and Ray dashboard specifically to the configured `HEAD_IP`; it does not
bind either service to every host interface. Ray control, rendezvous, and
dynamic worker ports still require a mutually trusted, firewalled fabric.
Never expose the deployment or its fabric address directly to the public
internet.

Use a dedicated private interface for `HEAD_IP` and `WORKER_IP`. Apply host or
network firewall rules so the API, dashboard, Ray, and rendezvous ports are
reachable only between the two configured nodes and explicitly trusted
clients. Binding to one address reduces exposure but is not authentication.

The launch scripts execute Docker commands over SSH. Use only SSH aliases,
hosts, images, checkpoints, and `.env` files you trust. Configuration values
embedded in remote commands are restricted to a conservative character set,
but that is not a substitute for host trust.

The containers deliberately have no restart policy and do not use privileged
mode. They receive GPU, IPC, and InfiniBand device access because the verified
workload requires those capabilities. Review every image, dependency, and
model-license change before upgrading the pinned stack.

The accepted upstream image is digest-pinned, and the build script validates
the accepted local base image ID and layer ancestry before building. See
`docs/REPRODUCIBILITY.md`. These checks improve provenance; they do not replace
independent image review or vulnerability scanning.

Report security issues through GitHub private vulnerability reporting when it
is available rather than opening a public issue.

# Security policy

Do not open a public issue containing credentials, private keys, model-access
tokens, sensitive generated content, or details of infrastructure you do not
intend to disclose.

This deployment uses Docker host networking for the API, Ray control plane,
Ray dashboard, PyTorch rendezvous, and dynamic Ray worker traffic. None of
those services is authenticated by this repository. Run it only on mutually
trusted nodes, bind or firewall it for your environment, and never expose the
example deployment directly to the public internet.

The launch scripts execute Docker commands over SSH. Use only SSH aliases,
hosts, images, checkpoints, and `.env` files you trust. Configuration values
embedded in remote commands are restricted to a conservative character set,
but that is not a substitute for host trust.

The containers deliberately have no restart policy and do not use privileged
mode. They receive GPU, IPC, and InfiniBand device access because the verified
workload requires those capabilities. Review every image, dependency, and
model-license change before upgrading the pinned stack.

Report security issues through GitHub private vulnerability reporting when it
is available rather than opening a public issue.

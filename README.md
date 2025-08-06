# Foldy

[![Frontend Tests](https://github.com/JBEI/foldy-internal/actions/workflows/frontend-tests.yml/badge.svg)](https://github.com/JBEI/foldy-internal/actions/workflows/frontend-tests.yml)
[![Python CI](https://github.com/JBEI/foldy-internal/actions/workflows/python-app.yml/badge.svg)](https://github.com/JBEI/foldy-internal/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/JBEI/foldy-internal/branch/main/graph/badge.svg)](https://codecov.io/gh/JBEI/foldy-internal)

<p align="center">
  Foldy is a webtool for doing computational structural biology, centered around protein structure prediction with AlphaFold.
</p>
<p align="center">
  <img src="frontend/public/pksito.gif" width="400" height="400" />
</p>

## Deployment Options

### 🚀 Quick Start: Run Locally

**Want to try Foldy right now?** Run this single command:

```bash
FOLDY_STORAGE_DIRECTORY=$HOME/foldy-data \
  docker-compose -f <(curl -s https://raw.githubusercontent.com/JBEI/foldy/main/deployment/local/docker-compose.yml) up -d
```

Foldy will be available at **http://localhost:3000** in ~2 minutes.

### All Deployment Options

Foldy is a composable set of services which can be deployed many ways. We document four types of deployment: Local (one-command Docker setup), Development (frontend-only for coding), Foldy-in-a-Box (Google Cloud VM), and Helm (scalable Kubernetes).

|Deployment Type|Features|Ease of setup|Best for|Setup|
|---|---|---|---|---|
|**Local**|**Full featured Foldy**|**One command**|**Trying Foldy locally**|**[Instructions](deployment/local/README.md)**|
|Development|Frontend only, no jobs|Very easy|Development work|[Instructions](deployment/development/README.md)|
|Foldy-in-a-Box|Full featured|Easy|Small teams, cloud VM|[Instructions](deployment/foldy-in-a-box/README.md)|
|Helm|Horizontally scalable|Hard|Large institutions|[Instructions](deployment/helm/README.md)|

## The Interface

See [docs/interface.md](docs/interface.md).

## Architecture

See [docs/architecture.md](docs/architecture.md).

## Comparison to Other Tools

There is a rich ecosystem for running structural biology tools, and Foldy is not the right structural biology wrapper for everyone! Please review the Foldy paper for a comparison to other useful structural biology tool wrappers.

## Development Setup

For complete development environment setup instructions, including Python virtual environment, Node.js/npm, pre-commit hooks, and Docker configuration, see:

**[Development Environment Setup Guide](deployment/development/README.md)**

The guide covers:
- Python 3.12 and virtual environment setup
- Node.js installation with nvm
- Pre-commit hooks (pyright, black, isort)
- Docker-based development environment
- Database setup and migrations
- Testing and troubleshooting


## Acknowledgements

Foldy utilizes many separate libraries and packages including:

- [Alphafold](https://github.com/deepmind/alphafold)
- [Autodock Vina](https://vina.scripps.edu/)
- [Pfam](https://www.ebi.ac.uk/interpro/)
- [NGL Viewer](https://nglviewer.org)
- [HMMER Suite](http://eddylab.org/software/hmmer)
- [Flask](https://flask.palletsprojects.com/en/2.2.x/)
- [Plotly](https://github.com/plotly/plotly.js)

We thank all their contributors and maintainers!

Use of the third-party software, libraries or code Foldy may be governed by separate terms and conditions or license provisions. Your use of the third-party software, libraries or code is subject to any such terms and you should check that you can comply with any applicable restrictions or terms and conditions before use.

## License

Foldy is distributed under a modified BSD license (see LICENSE).

## Copyright Notice

Foldy Copyright (c) 2023, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy) and University
of California, Berkeley. All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.

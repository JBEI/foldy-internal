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

Foldy is a composable set of services which can be deployed lots of  ways. We currently document three types of deployment: Development, Foldy-in-a-Box, and Helm. The development deployment is not fully featured - it cannot run jobs - but supports the frontend features and it can easily be run on a laptop for development purposes. Foldy-in-a-box is a quick deployment option - it can be run in under ten minutes - for creating a full featured Foldy instance on a Google Cloud machine. It could also be the starting point for a more bespoke deployment on a large local machine. Finally the Helm deployment is the horizontally scalable, cloud deployment, built on Kubernetes. The name comes from Helm Charts, which are a tool for specifying Kubernetes deployments. The Helm deployment is involved, but it is secure and can be scaled to hundreds of users and tens of thousands of folds.

You can find more information about employing the different deployment options in their respective `deployment` directories.

|Deployment Type|Features|Ease of setup|Setup|
|---|---|---|---|
|Development|No tools, just an interface|Extremely easy|[Instructions](deployment/development/README.md)|
|Foldy-in-a-Box|All tools can run|Easy|[Instructions](deployment/foldy-in-a-box/README.md)|
|Helm|Scalable to hundreds of users|Hard|[Instructions](deployment/helm/README.md)|

## The Interface

See [docs/interface.md](docs/interface.md).

## Architecture

See [docs/architecture.md](docs/architecture.md).

## Comparison to Other Tools

There is a rich ecosystem for running structural biology tools, and Foldy is not the right structural biology wrapper for everyone! Please review the Foldy paper for a comparison to other useful structural biology tool wrappers.

## Development Setup

### Python Environment Setup
1. Install Python 3.12:
   - On macOS with Homebrew: `brew install python@3.12`
   - On Ubuntu/Debian: `sudo apt install python3.12 python3.12-venv`
   - On Windows: Download Python 3.12 from [python.org](https://www.python.org/downloads/)

2. Create and activate a virtual environment with Python 3.12:
```bash
# From the project root
# On Unix/macOS
python3.12 -m venv .venv
source .venv/bin/activate

# On Windows
py -3.12 -m venv .venv
.venv\Scripts\activate
```

3. Verify correct Python version:
```bash
python --version  # Should output Python 3.12.x
```

4. Install the project dependencies:
```bash
cd backend
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### Pre-commit Setup
We use pre-commit hooks to ensure code quality. The hooks run:
- mypy (type checking)
- black (code formatting)
- isort (import sorting)
- basic file checks

1. Install pre-commit:
```bash
pip install pre-commit
```

2. Install the git hooks:
```bash
pre-commit install --install-hooks
pre-commit install -t pre-push  # Also install pre-push hooks
```

3. (Optional) Run against all files:
```bash
pre-commit run --all-files
```

### Updating Pre-commit Hooks
When the `.pre-commit-config.yaml` file changes (either by you or after pulling updates), run:
```bash
pre-commit autoupdate  # Updates hooks to latest versions
pre-commit clean       # Cleans out old hooks
pre-commit install --install-hooks  # Reinstalls hooks
```

### Notes on mypy
- mypy configuration is in `backend/mypy.ini`
- Type checking runs on the `backend/src` directory
- Required type stubs are automatically installed by pre-commit
- The virtual environment's Python version should match the one specified in mypy.ini (Python 3.12)

### Troubleshooting
If you encounter mypy errors:
1. Ensure you're using Python 3.12
2. Try clearing the pre-commit cache:
```bash
pre-commit clean
pre-commit gc
```
3. Verify all dependencies are installed:
```bash
pip install -r backend/requirements.txt
pip install -r backend/requirements-dev.txt
```


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

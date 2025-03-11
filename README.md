# Foldy

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

# Seaborn Heatmap Examples

This repository contains examples of how to create 2D heatmaps using Seaborn, a powerful data visualization library for Python.

## Requirements

Make sure you have the following packages installed:
- numpy
- pandas
- matplotlib
- seaborn

You can install them using pip:
```bash
pip install numpy pandas matplotlib seaborn
```

## Examples

The `heatmap_example.py` file contains two examples:

1. **Random Correlation Matrix**: A simple heatmap of a randomly generated correlation matrix.
2. **Flights Dataset**: A heatmap showing passenger numbers by month and year using Seaborn's built-in flights dataset.

## Running the Examples

Simply run the Python script:
```bash
python heatmap_example.py
```

This will:
1. Display the heatmaps in a window
2. Save the heatmaps as PNG files (`heatmap_example.png` and `flights_heatmap.png`)

## Customizing Heatmaps

The examples demonstrate several customization options:

- **Color Maps**: Different color schemes (`YlGnBu`, `YlOrRd`)
- **Annotations**: Showing values in each cell
- **Formatting**: Different formats for the annotations (`.2f` for decimals, `d` for integers)
- **Line Width**: Adjusting the width of lines between cells
- **Color Bar**: Customizing the color bar appearance

## Additional Customization Options

You can further customize your heatmaps with:

- **Masking**: Hide specific cells (e.g., using `np.triu` to show only upper triangle)
- **Center**: Set a center value for diverging colormaps
- **Robust**: Use robust quantiles for color mapping
- **Square**: Make each cell square-shaped
- **Cluster**: Apply clustering to rows and columns

For more information, refer to the [Seaborn documentation](https://seaborn.pydata.org/generated/seaborn.heatmap.html).


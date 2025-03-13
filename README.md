# README

This research project is for exploring how defensive symbionts affect host-parasite systems using mathematical models, numerical analysis, and simulations. Key interests include: 
- evolution of parasite virulence and symbiont-conferred resistance
- change in population size
- parasite prevalence

### Authors
- Cameron Smith (cameronsmith50@outlook.com)
- Scott Renegado

## Installation
Let the working directory be your project folder (`cd`). The file structure should look like the following

```sh
your_project_name/ <-- cd
|-data/
|-results/
|-src/
.gitignore
README.md
```

**Requirements** are Python 3 and the following packages:
- numpy,
- matplotlib,
- pandas,
- scipy,
- os,
- sys,
- time,
- datetime,
- tqdm,
- pickle.

## Usage

### Figure 2: Parameter sweep

To generate a parameter sweep, we run `resistance_and_c1.py` by typing the
following into our terminal window:

```sh
python src/resistance_and_c1.py
```

The data and figures are automatically saved in the appropriate directories
within a folder of the form `20YY-MM-DD 00h00m00s` where the date and time are
the current timestamp.

### Figure 3: Classification

#### NOTE: This code is relatively time intensive.

In order to run the classification code, we need the `classification.py` file.
We can run this in one of three ways.

For an initial classification (this needs to be done first) we would write the
following into our console:

```sh
python src/classification.py initial directory/ 11
```
This will generate a 11x11 initial classification, saving the data in the
`data/directory/` folder. Both the directory and number may be changed.

To refine the classification, we take a previous classification directory, and
then double the resolution in both c1 and c2. This is done in the following way:

```sh
python src/classification.py refine directory/
```

where `/data/directory/`is a folder previously initialised and/or refined data.

Finally, we are able to plot the classification via the following:

```sh
python src/classification.py plot directory/
```

where again the folder at `data/directory/` contains classification data. This
will generate figures in `results/directory/`. Note that any labels plotted onto
the classification, as in Fig. 3 of the matuscript, may be in the wrong place if
running with new parameters. These lines are 745-748 if you wish to amend them.

### Figure 4: Coevolutionary trajectories

To run and plot these plots, we require two functions. The first is
`coevolution_heat_maps.py` which is used to generate the data, and
`plot_heat_maps.py` is used to plot any generated data.

To run the data, we use:

```sh
python src/coevolution_heat_maps.py
```

which will generate data in the directory `data/traj_20YY-MM-DD_00h00m00s/`
where the date-time is replaced by the timestamp on the computer. To plot any
generated data, we use

```sh
python src/plot_heat_maps.py traj_20YY-MM-DD_00h00m00s/
```

where the directory is one containing data. Note that this may be a little slow
as it has to calculate the heat maps for the background of the plots. It will
save figures in `figures/traj_20YY-MM-DD_00h00m00s/`.

### Further usage
Have a look into `modeling.py` and the `set_default_parameters()` function. This is where you can play with different default parameter values.

The `lookup_table.csv` contains the names of generated datasets and their creation date and time. Modify the date in the lookup table to have the source code read certain versions of datasets. Be sure to adjust parameters in the source code according to the dataset's parameters.txt file.

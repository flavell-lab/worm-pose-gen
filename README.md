# Worm pose

The pose app follows a worm through a behaviour recording. In every frame it
finds the worm's body: its midline from head to tail, its width, and which
end is the head. It points out the moments it is unsure about so you can
check them, and it saves a table of the worm's posture, curvature and speed
for every frame.

You can also help the app get better: **label** frames by drawing the worm,
and those labels train the next models.

The app runs on the lab's flv computers (flv-c2, flv-c3, flv-c4) and on
Engaging. You use it in your web browser.

## Set it up (once)

Log in to an flv computer (or Engaging) and copy these lines into the
terminal, one at a time.

1. Check that `uv` is installed:

   ```bash
   uv --version
   ```

   If that prints an error, install it, then close the terminal and open a
   new one:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Download the app and install it. Put your own name in place of
   `$USER` only if your folder in `/temp_data4` has a different name:

   ```bash
   git clone git@github.com:flavell-lab/worm-pose-gen.git
   cd worm-pose-gen
   scripts/bootstrap_environment.sh --data-dir /temp_data4/$USER/worm-pose-data
   ```

   The last folder is where your work is saved. It needs space: about 2 GB
   per recording you analyse. On Engaging, use a folder in the lab's storage
   under `/orcd/data/flavell/001/` instead of `/temp_data4`.

   This takes a few minutes. It ends with a list of checks and a last line
   that says `OK`. A line starting with `warn` is fine; a line starting with
   `FAIL` needs fixing, so ask for help.

## Start the app

On an flv computer:

```bash
cd worm-pose-gen
./worm-pose-app
```

It prints something like this:

```
pose app at http://127.0.0.1:8768/
From your own computer: ssh -N -L 8768:localhost:8768 you@flv-c3
then open http://localhost:8768/
```

1. On **your own computer**, open a terminal and paste the `ssh` line it
   printed. It looks like nothing happens: that is right. Leave that window
   open.
2. Open the address it printed (`http://localhost:8768/`) in your web
   browser.

To stop the app, go back to the terminal where it runs and press Ctrl-C.

**On Engaging**, ask for a GPU first, then start the app:

```bash
srun -p ou_bcs_normal --gres=gpu:1 -c 8 --mem=32G -t 08:00:00 --pty bash
cd worm-pose-gen
./worm-pose-app
```

## Analyse a recording

The **Workspace** page opens on your recordings: the ones you have analysed
first, with what is left to review, then the ones you have not.

![The recordings list](docs/screenshots/recordings.png)

1. **Add your recordings.** The first time, press **Add a folder…** and
   choose the folder your recordings are in. The app finds every recording
   in it and in the folders inside it. **Add recordings…** adds more later.
2. **Open a recording** from the list.
3. **Analyse.** Press **Analyse** and wait for the progress bars. It takes
   about a minute per thousand frames. You can close the browser and come
   back later.
4. **Review.** On the left is a list of moments the app is unsure about,
   with the reason (for example "head/tail uncertain" or "coiled"). For each
   one, look at the worm and either press **Looks OK** or fix it:
   - **Flip** if the head and tail are swapped.
   - **Refit** to let the app try again (you keep or discard the result).
   - **Edit mask** to paint the worm yourself on a frame.
   - **Relabel** to send the frames to Labeling.

   Every fix can be undone. Press `?` to see the keyboard shortcuts.

   ![A recording open in the Workspace](docs/screenshots/workspace.png)

5. **Export.** Press **Export** at the top. The table is saved in your data
   folder, in `workspaces/`, in the recording's `exports` folder. It has one
   row per frame:
   the midline, curvature, width, head and tail positions, and speed.
   Lengths are in µm.

## Label frames

Labels are how the models learn, and more labels make them better.

1. Go to the **Labeling** page and pick a queue of frames (or make one with
   **New queue**).
2. Paint the worm with the **Worm** brush (and fix mistakes with
   **Background**), or start from the app's suggestion.
3. Trace the midline, starting at the **head**.
4. Press **Save & next** (Enter).

![The Labeling page](docs/screenshots/labeling.png)

Your labels are saved in your own data folder. Editing a lab label saves
your own copy; the lab's version is not changed.

## Train a model

This is optional. On **Training**, **Models** lists the
models with their scores (★ marks the ones in use). **Train from this**
trains a new model from one of them, the same way it was made, on the
labels you choose under **Datasets**.

**Datasets** shows every label for a setup and which recordings each
dataset uses to train, validate and test.

![The Datasets tab of the Training page](docs/screenshots/datasets.png)

## Get the latest version

```bash
cd worm-pose-gen
git pull
scripts/bootstrap_environment.sh
```

Your work and settings are kept.

## If something goes wrong

- **The browser shows nothing.** Check that the `ssh` window on your
  computer is still open, and that the address matches what the app
  printed.
- **A recording says "Cannot be read".** Hover over it to see why. Some old
  recordings are damaged.
- **The app says there is no GPU.** Start it on another flv computer, or on
  Engaging inside the `srun` job above.

For anything else, ask Alex Hiser.

## For developers

Command-line use, file locations, training, publishing to the lab library
and the code are in [`docs/DEVELOPER.md`](docs/DEVELOPER.md).

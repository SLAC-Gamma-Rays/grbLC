from IPython.display import display, clear_output
import numpy as np
import pandas as pd
import re, os


def LC_summary(filepaths):
    lc_data = {}
    fig, ax = None, None
    for filepath in filepaths:
        grb = re.search("(\d{6}[A-Z]?)", filepath)[0]
        df = pd.read_csv(filepath, delimiter=r"\t+|\s+", engine="python", header=0)
        num_rows = len(df.index)
        bands = ",".join(list(df["band"]))  # because lists aren't hashable >:(
        lc_data[grb] = [num_rows, bands]

    return {grb: l for grb, (l, _) in lc_data.items()}


def outlier_check_(filepath):
    try:
        op = OutlierPlot(filepath, plot=True)
        while True:
            try:
                clear_output(wait=True)
                key = op.prompt()
                op.update(key)
            except StopIteration:
                break
    except ImportError as e:
        print(e)
        pass


def check_all_(filepaths):
    num_points = LC_summary(filepaths)
    for filepath in filepaths:
        print(filepath)
        grb = os.path.split(filepath)[-1].rstrip("_converted_flux.txt")
        print("LOOKING AT GRB: " + str(grb))
        try:
            if num_points[grb] > 0:
                outlier_check_(filepath)
        except KeyboardInterrupt:
            break


class OutlierPlot:
    def __init__(self, filepath, plot=True):
        self.main_path = os.path.split(filepath)[0]
        self.grb = re.search("(\d{6}[A-Z]?)", filepath)[0]

        self.df = pd.read_csv(filepath, delimiter=r"\t+|\s+", engine="python", header=0)
        self.df = self.df.sort_values(by=["time_sec"]).reset_index(drop=True)
        self.numpts = len(self.df.index)
        if self.numpts == 0:
            raise ImportError("Empty file given.")

        self.accepted = self._try_import_prev_data("accepted")  # these two functions return {}'s if
        self.rejected = self._try_import_prev_data("rejected")  # nothing is found
        self.queue = []
        self.currpt = 0
        self.prevpt = -1
        # values are tools that will undo the key
        # e.g., to undo a "f", you need to decrement

        self._update_curr_help_vals()
        if plot:
            self.display = self.plot(return_display=True)
        else:
            print("imported", self.numpts, "pts")

    def plot(self, return_display=False):
        import plotly.express as px
        import plotly.graph_objects as go
        import plotly.io as pio

        pio.renderers.default = "plotly_mimetype"

        # plot main sample of points by band
        fig = px.scatter(
            self.df,
            x=np.log10(self.df["time_sec"]),
            y=np.log10(self.df["flux"]),
            error_y=self.df["flux_err"] / (self.df["flux"] * np.log(10)),
            color="band",
            width=700,
            height=400,
        )

        # update overall layout (x & y axis labels, etc.)
        fig.update_layout(
            xaxis_title=r"logT (sec)",
            yaxis_title=r"logF (erg cm-2 s-1)",
            title=self.grb,
            legend_title="Band",
        )

        # plot accepted points if there are any
        scatters = []
        if len(self.accepted) > 0:
            accepted_df = pd.concat(list(self.accepted.values()), axis=0, join="inner")
            accepted_df[["time_sec", "flux", "flux_err"]] = accepted_df[["time_sec", "flux", "flux_err"]].astype(
                "float64"
            )
            band = accepted_df["band"]
            scatters.append(
                go.Scatter(
                    x=np.log10(accepted_df["time_sec"]),
                    y=np.log10(accepted_df["flux"]),
                    error_y=dict(array=accepted_df["flux_err"] / (accepted_df["flux"] * np.log(10))),
                    mode="markers",
                    customdata=band,
                    hovertemplate="band: %{customdata}<br>" + "x: %{x}<br>" + "y: %{y}<br>",
                    name="Accepted",
                )
            )

        # plot rejected points if there are any
        if len(self.rejected) > 0:
            rejected_df = pd.concat(list(self.rejected.values()), axis=0, join="inner")
            rejected_df[["time_sec", "flux", "flux_err"]] = rejected_df[["time_sec", "flux", "flux_err"]].astype(
                "float64"
            )
            band = rejected_df["band"]
            scatters.append(
                go.Scatter(
                    x=np.log10(rejected_df["time_sec"]),
                    y=np.log10(rejected_df["flux"]),
                    error_y=dict(array=rejected_df["flux_err"] / (rejected_df["flux"] * np.log(10))),
                    mode="markers",
                    customdata=band,
                    hovertemplate="band: %{customdata}<br>" + "x: %{x}<br>" + "y: %{y}<br>",
                    name="Rejected",
                )
            )

        if len(scatters) > 0:
            fig.add_traces(scatters)

        # plot current point
        currpt = self.currpt
        x = np.log10(self.df["time_sec"][currpt])
        y = np.log10(self.df["flux"][currpt])
        yerr = self.df["flux_err"][currpt] / (self.df["flux"][currpt] * np.log(10))
        band = self.df["band"][currpt]
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                error_y=dict(array=[yerr]),
                mode="markers",
                hoverinfo=["text+x+y"],
                text=f"band={band}",
                name="Current Point",
                marker_color="rgba(0,0,0, 0.5)",
            )
        )

        if return_display:
            return display(fig, display_id=True)
        else:
            clear_output(wait=True)
            self.display.display(fig)

    def _try_import_prev_data(self, pile: str):
        try:
            filepath = os.path.join(self.main_path, f"{self.grb}_converted_flux_{pile}.txt")
            data = pd.read_csv(filepath, delimiter=r"\t+|\s+", engine="python", header=0)
            if len(data.index) > 0:
                return {idx: data.loc[data.index == idx] for idx in data.index}
            else:
                return {}
        except:
            return {}

    def _save(self):
        acceptedpath = os.path.join(self.main_path, f"{self.grb}_converted_flux_accepted.txt")
        if len(self.accepted) > 0:
            accepted_df = pd.concat(list(self.accepted.values()), axis=0, join="inner")
            accepted_df.to_csv(acceptedpath, sep="\t", index=0)
        else:
            try:
                os.remove(acceptedpath)
            except:
                pass

        rejectedpath = os.path.join(self.main_path, f"{self.grb}_converted_flux_rejected.txt")
        if len(self.rejected) > 0:
            rejected_df = pd.concat(list(self.rejected.values()), axis=0, join="inner")
            rejected_df.to_csv(rejectedpath, sep="\t", index=0)
        else:
            try:
                os.remove(rejectedpath)
            except:
                pass

    # set current index & update current mag, mag_err, and band
    def _set_currpt(self, currpt):
        self.currpt = currpt % self.numpts
        self._update_curr_help_vals()

    # currently unused
    def _update_curr_help_vals(self):
        self.curr_mag = self.df.at[self.currpt, "flux"]
        self.curr_mag_err = self.df.at[self.currpt, "flux_err"]
        self.curr_band = self.df.at[self.currpt, "band"]

    # pop a row from the main sample and move it to accepted or rejected pile
    def _pop(self, index, pileto, pilefrom):
        pileto[index] = self.df.loc[self.df.index == index]
        if index in pilefrom:
            del pilefrom[index]

    # "insert" a row from accepted or rejected back into the main sample
    def _insert(self, index, pile):
        if index in pile:
            del pile[index]

    # increment
    def _inc(self):
        self.prevpt = self.currpt
        self._set_currpt(self.currpt + 1)

    # decrement
    def _dec(self):
        self.prevpt = self.currpt
        self._set_currpt(self.currpt - 1)

    # accept current point
    def _accept(self):
        self._pop(self.currpt, self.accepted, self.rejected)
        self._inc()

    # reject current point
    def _reject(self):
        self._pop(self.currpt, self.rejected, self.accepted)
        self._inc()

    # do the opposite of what the last action ("job") did
    # basically go back in time one step back
    def _undo(self):
        if len(self.queue) == 0:
            return

        last_job, __, old_prevpt = self.queue.pop(-1)
        if last_job == "f":
            self._dec()

        elif last_job == "b":
            self._inc()

        elif last_job == "a":
            self._insert(self.prevpt, self.accepted)
            self._dec()

        elif last_job == "r":
            self._insert(self.prevpt, self.rejected)
            self._dec()

        self.prevpt = old_prevpt

    def update(self, key):

        if key in ["f", "forward"]:
            self.queue.append(["f", self.currpt, self.prevpt])
            self._inc()
        elif key in ["b", "backwards"]:
            self.queue.append(["b", self.currpt, self.prevpt])
            self._dec()
        elif key in ["a", "accept"]:
            self.queue.append(["a", self.currpt, self.prevpt])
            self._accept()
            self._save()
        elif key in ["r", "reject"]:
            self.queue.append(["r", self.currpt, self.prevpt])
            self._reject()
            self._save()
        elif key in ["s", "strip"]:
            self.queue.append(["s", self.currpt, self.prevpt])
            self._insert(self.currpt, self.accepted)
            self._insert(self.currpt, self.rejected)
            self._inc()
            self._save()
        elif key in ["u", "undo"]:
            self._undo()
            self._save()
        elif key in ["d", "done"]:
            clear_output()
            raise StopIteration
        elif key in ["q", "quit"]:
            clear_output()
            raise KeyboardInterrupt

        self.plot()

    def prompt(self):
        return input("a:accept r:reject s:strip u:undo f:forward b:backward d:done with GRB q:quit")

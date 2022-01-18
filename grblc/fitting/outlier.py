import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import clear_output
from IPython.display import display
from plotly.graph_objects import FigureWidget

from ..convert import get_dir
from .constants import grb_regex
from .io import check_header

class OutlierPlot:
    _name_placeholder = "unknown grb"

    def __init__(
        self,
        filename: str = None,
        accepted: dict = None,
        rejected: dict = None,
        name: str = None,
        plot=True,
    ):
        assert isinstance(filename, str) ^ all(
            (x, dict) for x in [accepted, rejected]
        ), "Must provide either a filename or two dicts of accepted and rejected points"

        if name is not None:
            self.grb = name
        elif filename is not None:
            self.grb = grb_regex.search(filename)[0]
        else:
            self.grb = self._name_placeholder

        if filename:
            self.main_path = os.path.dirname(filename)
            self.df = pd.read_csv(
                filename, delimiter=r"\t+|\s+", engine="python", header=check_header(filename)
            )
            # these two functions return {}'s if nothing is found
            self.accepted = self._try_import_prev_data("accepted")
            self.rejected = self._try_import_prev_data("rejected")
        else:
            self.main_path = get_dir()
            self.df = pd.concat([accepted, rejected], axis=0, ignore_index=True)
            self.accepted = accepted
            self.rejected = rejected
            self._save()

        self.df = self.df.sort_values(by=["time_sec"]).reset_index(drop=True)
        self.numpts = len(self.df.index)
        if self.numpts == 0:
            raise ValueError("Empty data given.")

        self.queue = []
        self.currpt = 0
        self.prevpt = -1

        if plot:
            self.display = self.plot(return_display=True)
            self.figure = FigureWidget(self.display)
            display(self.figure)

    def plot(self, hover: list = None, return_display=False):

        if return_display:
            # plot main sample of points by band

            if hover is None:
                pass
            try:
                self.df["source"]
                kwargs = {"hover_data": ["source"]}
            except:
                kwargs = {}

            fig = px.scatter(
                self.df,
                x=np.log10(self.df["time_sec"]),
                y=np.log10(self.df["flux"]),
                error_y=self.df["flux_err"] / (self.df["flux"] * np.log(10)),
                color="band",
                width=700,
                height=400,
                **kwargs,
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
                accepted_df = pd.concat(
                    list(self.accepted.values()), axis=0, join="inner"
                )
                accepted_df[["time_sec", "flux", "flux_err"]] = accepted_df[
                    ["time_sec", "flux", "flux_err"]
                ].astype("float64")
                band = accepted_df["band"]
                # get source type if possible
                try:
                    source = accepted_df["source"]
                except:
                    source = None
                customdata = (
                    np.stack((band, source), axis=-1)
                    if source is not None
                    else np.stack((band), axis=-1)
                )
                addition = "source: %{customdata[1]}<br>" if source is not None else ""

                scatters.append(
                    go.Scatter(
                        x=np.log10(accepted_df["time_sec"]),
                        y=np.log10(accepted_df["flux"]),
                        error_y=dict(
                            array=accepted_df["flux_err"]
                            / (accepted_df["flux"] * np.log(10))
                        ),
                        mode="markers",
                        customdata=customdata,
                        hovertemplate=addition
                        + "band: %{customdata[0]}<br>"
                        + "x: %{x}<br>"
                        + "y: %{y}<br>",
                        name="Accepted",
                    )
                )
            else:
                scatters.append(
                    go.Scatter(
                        x=[],
                        y=[],
                        error_y=dict(array=[]),
                        mode="markers",
                        name="Accepted",
                    )
                )

            # plot rejected points if there are any
            if len(self.rejected) > 0:
                rejected_df = pd.concat(
                    list(self.rejected.values()), axis=0, join="inner"
                )
                rejected_df[["time_sec", "flux", "flux_err"]] = rejected_df[
                    ["time_sec", "flux", "flux_err"]
                ].astype("float64")
                band = rejected_df["band"]
                # get source type if possible
                try:
                    source = rejected_df["source"]
                except:
                    source = None
                customdata = (
                    np.stack((band, source), axis=-1)
                    if source is not None
                    else np.stack((band), axis=-1)
                )
                addition = "source: %{customdata[1]}<br>" if source is not None else ""

                scatters.append(
                    go.Scatter(
                        x=np.log10(rejected_df["time_sec"]),
                        y=np.log10(rejected_df["flux"]),
                        error_y=dict(
                            array=rejected_df["flux_err"]
                            / (rejected_df["flux"] * np.log(10))
                        ),
                        mode="markers",
                        customdata=customdata,
                        hovertemplate=addition
                        + "band: %{customdata[0]}<br>"
                        + "x: %{x}<br>"
                        + "y: %{y}<br>",
                        name="Rejected",
                    )
                )
            else:
                scatters.append(
                    go.Scatter(
                        x=[],
                        y=[],
                        error_y=dict(array=[]),
                        mode="markers",
                        name="Rejected",
                    )
                )

            fig.add_traces(scatters)

            x = np.log10(self.df["time_sec"][self.currpt])
            y = np.log10(self.df["flux"][self.currpt])
            yerr = self.df["flux_err"][self.currpt] / (
                self.df["flux"][self.currpt] * np.log(10)
            )
            band = self.df["band"][self.currpt]
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    error_y=dict(array=[yerr]),
                    mode="markers",
                    hoverinfo=["text+x+y"],
                    text=f"band={band}",
                    name="Current Point",
                    marker_color="rgba(0,0,0,0.5)",
                )
            )
            return fig
        else:
            # update curr pt
            x = np.log10(self.df["time_sec"][self.currpt])
            y = np.log10(self.df["flux"][self.currpt])
            yerr = self.df["flux_err"][self.currpt] / (
                self.df["flux"][self.currpt] * np.log(10)
            )
            band = self.df["band"][self.currpt]

            self.figure.update_traces(
                patch=dict(
                    x=[x], y=[y], error_y=dict(array=[yerr]), text=f"band={band}"
                ),
                selector=dict(name="Current Point"),
                overwrite=True,
            )

            # plot accepted points if there are any
            if len(self.accepted) > 0:
                accepted_df = pd.concat(
                    list(self.accepted.values()), axis=0, join="inner"
                )
                accepted_df[["time_sec", "flux", "flux_err"]] = accepted_df[
                    ["time_sec", "flux", "flux_err"]
                ].astype("float64")
                band = accepted_df["band"]
                # get source type if possible
                try:
                    source = accepted_df["source"]
                except:
                    source = None
                customdata = (
                    np.stack((band, source), axis=-1)
                    if source is not None
                    else np.stack((band), axis=-1)
                )
                addition = "source: %{customdata[1]}<br>" if source is not None else ""

                patch = dict(
                    x=np.log10(accepted_df["time_sec"]),
                    y=np.log10(accepted_df["flux"]),
                    error_y=dict(
                        array=accepted_df["flux_err"]
                        / (accepted_df["flux"] * np.log(10))
                    ),
                    customdata=customdata,
                    hovertemplate=addition
                    + "band: %{customdata[0]}<br>"
                    + "x: %{x}<br>"
                    + "y: %{y}<br>",
                )
                self.figure.update_traces(
                    patch=patch, selector=dict(name="Accepted"), overwrite=True
                )
            else:
                patch = dict(
                    x=[],
                    y=[],
                    error_y=dict(array=[]),
                )
                self.figure.update_traces(
                    patch=patch, selector=dict(name="Accepted"), overwrite=True
                )

            # plot rejected points if there are any
            if len(self.rejected) > 0:
                rejected_df = pd.concat(
                    list(self.rejected.values()), axis=0, join="inner"
                )
                rejected_df[["time_sec", "flux", "flux_err"]] = rejected_df[
                    ["time_sec", "flux", "flux_err"]
                ].astype("float64")
                band = rejected_df["band"]
                # get source type if possible
                try:
                    source = rejected_df["source"]
                except:
                    source = None
                customdata = (
                    np.stack((band, source), axis=-1)
                    if source is not None
                    else np.stack((band), axis=-1)
                )
                addition = "source: %{customdata[1]}<br>" if source is not None else ""

                patch = dict(
                    x=np.log10(rejected_df["time_sec"]),
                    y=np.log10(rejected_df["flux"]),
                    error_y=dict(
                        array=rejected_df["flux_err"]
                        / (rejected_df["flux"] * np.log(10))
                    ),
                    customdata=customdata,
                    hovertemplate="band: %{customdata[0]}<br>"
                    + "x: %{x}<br>"
                    + "y: %{y}<br>",
                )
                self.figure.update_traces(
                    patch=patch, selector=dict(name="Rejected"), overwrite=True
                )
            else:
                patch = dict(
                    x=[],
                    y=[],
                    error_y=dict(array=[]),
                )
                self.figure.update_traces(
                    patch=patch, selector=dict(name="Rejected"), overwrite=True
                )

    def _try_import_prev_data(self, pile: str):
        try:
            filepath = os.path.join(
                self.main_path, f"{self.grb}_converted_flux_{pile}.txt"
            )
            data = pd.read_csv(
                filepath, delimiter=r"\t+|\s+", engine="python", header=0
            )
            if len(data.index) > 0:
                return {idx: data.loc[data.index == idx] for idx in data.index}
            else:
                return {}
        except:
            return {}

    def _save(self):
        acceptedpath = os.path.join(
            self.main_path, f"{self.grb}_converted_flux_accepted.txt"
        )
        if len(self.accepted) > 0:
            accepted_df = pd.concat(list(self.accepted.values()), axis=0, join="inner")
            accepted_df.to_csv(acceptedpath, sep="\t", index=0)
        else:
            try:
                os.remove(acceptedpath)
            except:
                pass

        rejectedpath = os.path.join(
            self.main_path, f"{self.grb}_converted_flux_rejected.txt"
        )
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

    def update(self, key, save=False):
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
            if not save:
                clear_output()
            raise StopIteration
        elif key in ["q", "quit"]:
            if not save:
                clear_output()
            raise KeyboardInterrupt

        self.plot()

    @staticmethod
    def prompt():
        return input(
            "a:accept r:reject s:strip u:undo f:forward b:backward d:done with GRB q:quit"
        )

    @classmethod
    def outlier_check_(cls, filepath, save=False):
        try:
            op = cls(filepath, plot=True)
            while True:
                try:
                    key = op.prompt()
                    op.update(key, save)
                except StopIteration:
                    break
        except ImportError as e:
            print(e)
            pass

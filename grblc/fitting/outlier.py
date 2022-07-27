import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import clear_output
from IPython.display import display
from plotly.graph_objects import FigureWidget

from ..util import get_dir
from .constants import grb_regex
from .io import check_header
from .lightcurve import Lightcurve

__all__ = ["OutlierPlot"]

class OutlierPlot:
    _name_placeholder = "unknown grb"

    def __init__(
        self,
        filename: str = None,
        data=None,
        name: str = None,
        plot=True,
        hover_labels: list = None,
        legend_label: str = None,
    ):
        assert isinstance(filename, str) ^ isinstance( data, (dict, pd.DataFrame) ), \
                                                "Must provide either a filename or data"
        if isinstance(data, dict):
            data = pd.DataFrame(data)

        if name is not None:
            self.grb = name
        elif filename is not None:
            self.grb = grb_regex.search(filename)[0]
        else:
            self.grb = self._name_placeholder

        if filename:
            self.main_path = os.path.dirname(filename)
            self.df = pd.read_csv(
                filename,
                delimiter=r"\t+|\s+",
                engine="python",
                header=check_header(filename),
            )
        else:
            self.main_path = get_dir()
            self.df = data

        # these two functions return {}'s if nothing is found
        self.accepted = self._try_import_prev_data("accepted")
        self.rejected = self._try_import_prev_data("rejected")
        self.df = self.df.sort_values(by=["time_sec"]).reset_index(drop=True)
        self.numpts = len(self.df.index)
        if self.numpts == 0:
            raise ValueError("Empty data given.")

        self.queue = []
        self.currpt = 0
        self.prevpt = -1

        if plot:
            self.display = self.init_plot(hover_labels, legend_label)
            self.figure = FigureWidget(self.display)
            display(self.figure)

    def init_plot(self, hover: list = None, legend_label: str = None):

        if hover is None:
            kwargs = dict(hover_data=list(map(str, self.df.columns[3:])))
        else:
            assert all( h in self.df.columns for h in hover ), \
                            "Hover data supplied not in attributes"

            kwargs = dict(hover_data=hover)

        # color will default to the band column in the dataframe
        if legend_label is not None:
            assert ( legend_label in self.df.columns ), \
                    f"Legend label {legend_label} not an attribute."
            color = legend_label
        elif "band" in self.df.columns:
            color = "band"
        else:
            color = None

        if hover is None:
            self.labels = self.df.columns[3:]
        else:
            self.labels = hover

        customdata = np.dstack(
            np.stack(tuple(list(self.df[n]) for n in self.labels), axis=-1)
        )
        hovertemplate = "x: %{x}<br>" + "y: %{y}<br>"
        hovertemplate += "<br>".join(
            n + ": %{customdata[" + str(idx) + "]}" for idx, n in enumerate(self.labels)
        )

        fig = px.scatter(
            self.df,
            x=np.log10(self.df["time_sec"]),
            y=np.log10(self.df["flux"]),
            error_y=self.df["flux_err"] / (self.df["flux"] * np.log(10)),
            color=color,
            width=700,
            height=400,
            **kwargs,
        )

        # update overall layout (x & y axis labels, etc.)
        fig.update_layout(
            xaxis_title=r"log T (sec)",
            yaxis_title=r"log F (erg cm-2 s-1)",
            title=self.grb,
            legend_title=color.capitalize() if color is not None else None,
        )

        scatter_plots = []

        # plot accepted or rejected points if there are any
        for type, c in {"accepted": "firebrick", "rejected": "royalblue"}.items():
            data_dict = getattr(self, type)
            if len(data_dict) > 0:
                data_df = pd.concat(list(data_dict.values()), axis=0, join="inner")
                data_df[["time_sec", "flux", "flux_err"]] = data_df[
                    ["time_sec", "flux", "flux_err"]
                ].astype("float64")

                customdata = np.dstack(tuple(data_df[n] for n in self.labels))

                scatter_plots.append(
                    go.Scatter(
                        x=np.log10(data_df["time_sec"]),
                        y=np.log10(data_df["flux"]),
                        error_y=dict(
                            array=data_df["flux_err"] / (data_df["flux"] * np.log(10))
                        ),
                        mode="markers",
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                        name=type.capitalize(),
                        marker=dict(color=c),
                    )
                )
            else:
                scatter_plots.append(
                    go.Scatter(
                        x=[],
                        y=[],
                        error_y=dict(array=[]),
                        mode="markers",
                        name=type.capitalize(),
                        marker=dict(color=c),
                    )
                )

        fig.add_traces(scatter_plots)

        # add current point in black over everything else!
        x = np.log10(self.df["time_sec"][self.currpt])
        y = np.log10(self.df["flux"][self.currpt])
        yerr = self.df["flux_err"][self.currpt] / (
            self.df["flux"][self.currpt] * np.log(10)
        )

        customdata = np.dstack(
            np.stack(tuple([self.df[n][self.currpt]] for n in self.labels), axis=-1)
        )

        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                error_y=dict(array=[yerr]),
                mode="markers",
                customdata=customdata,
                hovertemplate=hovertemplate,
                name="Current Point",
                marker=dict(color="rgba(0,0,0,0.5)"),
            )
        )

        return fig

    def update_plot(self):
        assert hasattr(self, "figure"), "Must call init_plot() first"

        # update curr pt
        x = np.log10(self.df["time_sec"][self.currpt])
        y = np.log10(self.df["flux"][self.currpt])
        yerr = self.df["flux_err"][self.currpt] / (
            self.df["flux"][self.currpt] * np.log(10)
        )

        customdata = np.dstack(
            np.stack(tuple([self.df[n][self.currpt]] for n in self.labels), axis=-1)
        )

        hovertemplate = "x: %{x}<br>" + "y: %{y}<br>"
        hovertemplate += "<br>".join(
            n + ": %{customdata[" + str(idx) + "]}" for idx, n in enumerate(self.labels)
        )
        self.figure.update_traces(
            patch=dict(
                x=[x],
                y=[y],
                error_y=dict(array=[yerr]),
                customdata=customdata,
                hovertemplate=hovertemplate,
            ),
            selector=dict(name="Current Point"),
            overwrite=True,
        )

        for type in ["accepted", "rejected"]:
            data_dict = getattr(self, type)
            if len(data_dict) > 0:
                data_df = pd.concat(list(data_dict.values()), axis=0, join="inner")
                data_df[["time_sec", "flux", "flux_err"]] = data_df[
                    ["time_sec", "flux", "flux_err"]
                ].astype("float64")

                customdata = np.dstack(tuple(data_df[n] for n in self.labels))
                patch = dict(
                    x=np.log10(data_df["time_sec"]),
                    y=np.log10(data_df["flux"]),
                    error_y=dict(
                        array=data_df["flux_err"] / (data_df["flux"] * np.log(10))
                    ),
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                )
                self.figure.update_traces(
                    patch=patch, selector=dict(name=type.capitalize()), overwrite=True
                )
            else:
                patch = dict(
                    x=[],
                    y=[],
                    error_y=dict(array=[]),
                )
                self.figure.update_traces(
                    patch=patch, selector=dict(name=type.capitalize()), overwrite=True
                )

    def _try_import_prev_data(self, pile: str):
        try:
            filepath = os.path.join(self.main_path, f"{self.grb}_flux_{pile}.txt")
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

        for type in ["accepted", "rejected"]:

            path = os.path.join(self.main_path, f"{self.grb}_flux_{type}.txt")
            if len(getattr(self, type)) > 0:
                data_df = pd.concat(
                    list(getattr(self, type).values()), axis=0, join="inner"
                )
                data_df.to_csv(path, sep="\t", index=0)
            else:
                try:
                    os.remove(path)
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
    def _insert(self, index):
        for pile in [self.accepted, self.rejected]:
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
            self._insert(self.prevpt)
            self._dec()

        elif last_job == "r":
            self._insert(self.prevpt)
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
            self._insert(self.currpt)
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

        self.update_plot()

    @staticmethod
    def prompt():
        return input(
            "a:accept r:reject s:strip u:undo f:forward b:backward d:done with GRB q:quit"
        )

    @classmethod
    def outlier_check_(cls, data, save=False):
        assert isinstance(
            data, (Lightcurve, str, pd.DataFrame)
        ), "Must provide either a Lightcurve object or a filepath or a dataframe"

        try:
            if isinstance(data, str):
                kwargs = {"filepath": data}
            elif isinstance(data, Lightcurve):
                kwargs = {"data": data.to_df(), "name": data.name}
            else:
                kwargs = {"data": data}

            op = cls(**kwargs)
            while True:
                try:
                    key = op.prompt()
                    op.update(key, save)
                except StopIteration:
                    break
        except ImportError as e:
            print(e)
            pass

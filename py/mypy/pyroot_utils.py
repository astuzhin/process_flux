import numpy as np
import pandas as pd
import ROOT
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline

def th_to_df(h, bin_names=None, col_names=None, is_error=None, is_interval=None, is_time_xbins=False):
    if h.GetSumw2N() == 0:
        h.Sumw2()
    if h.GetDimension() == 1:
        nbins = h.GetNbinsX()
        values = np.array(h.GetArray(), copy=False)[:(nbins+2)].copy()
        values = values[1:-1]
        if is_error is not None:
            errors = np.array(h.GetSumw2().GetArray(), copy=False)[:(nbins+2)].copy()
            if is_error == "error":
                errors = np.sqrt(errors[1:-1])
            elif is_error == "sumw2":
                errors = errors[1:-1]
            else:
                raise ValueError(f"Invalid error type: {is_error}")
        x = np.array([h.GetXaxis().GetBinLowEdge(i) for i in range(1, nbins+2)])
        if is_time_xbins:
            x = pd.to_datetime(x, unit='s', utc=True)
        if is_interval is not None:
            x_index = pd.IntervalIndex.from_breaks(x, closed='right') if is_interval[0] else pd.Index(x[:-1])
            x_index.name = bin_names
        else:
            x_index = pd.Index(x, name=bin_names)
        if is_error is not None:
            if col_names is None:
                col_names = ["val", "err"]
            df = pd.DataFrame({col_names[0]: values, col_names[1]: errors}, index=x_index)
        else:
            df = pd.Series(values, index=x_index)
        return df
    elif h.GetDimension() == 2:
        nbinsx = h.GetNbinsX()
        nbinsy = h.GetNbinsY()
        values = np.array(h.GetArray(), copy=False)[:(nbinsx+2)*(nbinsy+2)].copy().reshape((nbinsy+2, nbinsx+2)).transpose(1, 0)
        values = values[1:-1, 1:-1].ravel()
        if is_error is not None:
            errors = np.array(h.GetSumw2().GetArray(), copy=False)[:(nbinsx+2)*(nbinsy+2)].copy().reshape((nbinsy+2, nbinsx+2)).transpose(1, 0)
            if is_error == "error":
                errors = np.sqrt(errors[1:-1, 1:-1].ravel())
            elif is_error == "sumw2":
                errors = errors[1:-1, 1:-1].ravel()
            else:
                raise ValueError(f"Invalid error type: {is_error}")
        x_edges_full = np.array([h.GetXaxis().GetBinLowEdge(i) for i in range(1, nbinsx+2)])
        y_edges_full = np.array([h.GetYaxis().GetBinLowEdge(i) for i in range(1, nbinsy+2)])
        if is_time_xbins:
            x_edges_full = pd.to_datetime(x_edges_full, unit='s', utc=True)
        if is_interval is not None:
            x_index = pd.IntervalIndex.from_breaks(x_edges_full, closed='right') if is_interval[0] else pd.Index(x_edges_full[:-1])
            y_index = pd.IntervalIndex.from_breaks(y_edges_full, closed='right') if is_interval[1] else pd.Index(y_edges_full[:-1])
            index = pd.MultiIndex.from_product([x_index, y_index], names=bin_names)
        else:
            index = pd.MultiIndex.from_product([x_edges_full[:-1], y_edges_full[:-1]], names=bin_names)
        if is_error is not None:
            if col_names is None:
                col_names = ["val", "err"]
            df = pd.DataFrame({col_names[0]: values, col_names[1]: errors}, index=index)
        else:
            df = pd.Series(values, index=index)
        return df
    elif h.GetDimension() == 3:
        nbinsx = h.GetNbinsX()
        nbinsy = h.GetNbinsY()
        nbinsz = h.GetNbinsZ()
        values = np.array(h.GetArray(), copy=False)[:(nbinsx+2)*(nbinsy+2)*(nbinsz+2)].copy().reshape((nbinsz+2, nbinsy+2, nbinsx+2)).transpose(2, 1, 0)
        values = values[1:-1, 1:-1, 1:-1].ravel()
        if is_error is not None:
            errors = np.array(h.GetSumw2().GetArray(), copy=False)[:(nbinsx+2)*(nbinsy+2)*(nbinsz+2)].copy().reshape((nbinsz+2, nbinsy+2, nbinsx+2)).transpose(2, 1, 0)
            if is_error == "error":
                errors = np.sqrt(errors[1:-1, 1:-1, 1:-1].ravel())
            elif is_error == "sumw2":
                errors = errors[1:-1, 1:-1, 1:-1].ravel()
            else:
                raise ValueError(f"Invalid error type: {is_error}")
                errors = np.sqrt(errors[1:-1, 1:-1, 1:-1].ravel())
        x_edges_full = np.array([h.GetXaxis().GetBinLowEdge(i) for i in range(1, nbinsx+2)])
        y_edges_full = np.array([h.GetYaxis().GetBinLowEdge(i) for i in range(1, nbinsy+2)])
        z_edges_full = np.array([h.GetZaxis().GetBinLowEdge(i) for i in range(1, nbinsz+2)])
        if is_time_xbins:
            x_edges_full = pd.to_datetime(x_edges_full, unit='s', utc=True)
        if is_interval is not None:
            x_index = pd.IntervalIndex.from_breaks(x_edges_full, closed='right') if is_interval[0] else pd.Index(x_edges_full[:-1])
            y_index = pd.IntervalIndex.from_breaks(y_edges_full, closed='right') if is_interval[1] else pd.Index(y_edges_full[:-1])
            z_index = pd.IntervalIndex.from_breaks(z_edges_full, closed='right') if is_interval[2] else pd.Index(z_edges_full[:-1])
            index = pd.MultiIndex.from_product([x_index, y_index, z_index], names=bin_names)
        else:
            index = pd.MultiIndex.from_product([x_edges_full[:-1], y_edges_full[:-1], z_edges_full[:-1]], names=bin_names)
        if is_error is not None:
            if col_names is None:
                col_names = ["val", "err"]
            df = pd.DataFrame({col_names[0]: values, col_names[1]: errors}, index=index)
        else:
            df = pd.Series(values, index=index)
        return df

def tgraph_to_df(tgraph, is_error=False, names=None):
    x = np.array(tgraph.GetX())
    y = np.array(tgraph.GetY())
    if is_error:
        yerr = np.array(tgraph.GetEYhigh())
        if names is None:
            names = ["value", "error"]
        return pd.DataFrame({names[0]: y, names[1]: yerr}, index=x)
    else:
        if names is None:
            names = ["value"]
        return pd.Series(y, index=x, name=names[0])

class UPROOT:
    def th_to_df(h, bin_names=None, col_names=['counts', 'sumw2'], is_interval=None, is_time=None):
        if len(h.axes) == 2:
            edgesx = h.axes[0].edges()
            edgesy = h.axes[1].edges()
            edgesx = pd.IntervalIndex.from_breaks(edgesx, closed='left') if is_interval is not None and is_interval[0] else pd.Index(edgesx[:-1])
            edgesy = pd.IntervalIndex.from_breaks(edgesy, closed='left') if is_interval is not None and is_interval[1] else pd.Index(edgesy[:-1])
            if is_time is not None:
                if is_time[0]:
                    edgesx = pd.to_datetime(edgesx, unit='s', utc=True)
                if is_time[1]:
                    edgesy = pd.to_datetime(edgesy, unit='s', utc=True)
            values = h.values().ravel()
            variances = h.variances().ravel()
            idx = pd.MultiIndex.from_product([edgesx, edgesy], names=bin_names)
        elif len(h.axes) == 3:
            edgesx = h.axes[0].edges()
            edgesy = h.axes[1].edges()
            edgesz = h.axes[2].edges()
            edgesx = pd.IntervalIndex.from_breaks(edgesx, closed='left') if is_interval is not None and is_interval[0] else pd.Index(edgesx[:-1])
            edgesy = pd.IntervalIndex.from_breaks(edgesy, closed='left') if is_interval is not None and is_interval[1] else pd.Index(edgesy[:-1])
            edgesz = pd.IntervalIndex.from_breaks(edgesz, closed='left') if is_interval is not None and is_interval[2] else pd.Index(edgesz[:-1])
            if is_time is not None:
                if is_time[0]:
                    edgesx = pd.to_datetime(edgesx, unit='s', utc=True)
                if is_time[1]:
                    edgesy = pd.to_datetime(edgesy, unit='s', utc=True)
                if is_time[2]:
                    edgesz = pd.to_datetime(edgesz, unit='s', utc=True)
            values = h.values().ravel()
            variances = h.variances().ravel()
            idx = pd.MultiIndex.from_product([edgesx, edgesy, edgesz], names=bin_names)
        return pd.DataFrame({col_names[0]: values, col_names[1]: variances}, index=idx)

import matplotlib.pyplot as plt
import matplotlib
import cmocean
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import cartopy.crs as ccrs
import numpy as np
import utils
import matplotlib as mpl
import os


def initialize_figure(fig_size=20, ratio=1, text_size=1, subplots=(1, 1)):
    fs = np.sqrt(fig_size)
    fig = plt.figure(
        figsize=(np.sqrt(ratio * fig_size), np.sqrt(fig_size / ratio)),
        dpi=300,
        layout="constrained",
    )

    gs = mpl.gridspec.GridSpec(subplots[0], subplots[1], figure=fig)

    ax = [[None] * subplots[1] for _ in range(subplots[0])]

    for i in range(subplots[0]):
        for j in range(subplots[1]):
            ax[i][j] = fig.add_subplot(gs[i, j])
            ax[i][j].grid(which="major", linewidth=fs * 0.015)
            ax[i][j].xaxis.set_tick_params(which="minor", bottom=False)
            ax[i][j].tick_params(
                axis="both",
                which="major",
                labelsize=1.5 * text_size * fs,
                size=fs * 0.5,
                width=fs * 0.15,
            )

    return fig, ax, fs, text_size


def map_plot(
    ds,
    ax,
    fig,
    extend="max",
    label="",
    fs=1,
    lim=[None, None],
    title="",
    shrink=1,
    cmap="jet",
    cbar=True,
):
    im = ds.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        cmap=cmap,
        vmin=lim[0],
        vmax=lim[1],
    )

    if cbar != False:
        cbar = fig.colorbar(im, orientation="vertical", extend=extend, shrink=shrink)
        cbar.set_label(label, fontsize=fs, labelpad=fs)
        cbar.ax.tick_params(labelsize=fs)
    ax.set_title(title, fontsize=fs)


def err_plot(ax, x_axis, y, y_err, fs, color="k", label_leg="", zorder=0):
    ax.errorbar(
        x_axis,
        y,
        yerr=y_err,
        fmt="s-",
        color=color,
        zorder=zorder,
        elinewidth=fs * 0.1,
        capsize=fs * 0.2,
        capthick=fs * 0.1,
        markersize=fs * 0.2,
        linewidth=fs * 0.1,
        label=label_leg,
    )


def distr_plot(
    ax, distr, fs, color="r", h=1, ref=None, median=False, mean=False, units=""
):
    values = distr["distribution"]
    bins = distr["bins"][:-1]
    if ref == None:
        max = np.max(values)
    else:
        max = np.max(ref)
    values_ = max * np.array(values) / (np.max(values))
    ax.barh(bins, values_, height=h, color=color)

    def find_closest(lst, value):
        index = min(range(len(lst)), key=lambda i: abs(lst[i] - value))
        closest_element = lst[index]
        return index, closest_element

    if median != False:
        idx_median, value_median = find_closest(bins, median)
        ax.hlines(
            bins[idx_median],
            0,
            values[idx_median - 1],
            color="gray",
            linewidth=fs * 0.15,
            zorder=2,
        )

    if mean != False:
        idx_mean, value_mean = find_closest(bins, mean)
        bins_ = bins[idx_mean - 1 : idx_mean + 1]
        ax.hlines(
            bins[idx_mean],
            0,
            values[idx_mean - 1],
            color="k",
            linewidth=fs * 0.15,
            zorder=2,
        )

    custom_lines = [
        Line2D([0], [0], color="k", lw=fs * 0.5),
        Line2D([0], [0], color="gray", lw=fs * 0.5),
    ]

    if mean != False and median != False:
        ax.legend(
            custom_lines,
            [
                f"Mean ({value_mean:0.01f} {units})",
                f"Median ({value_median:0.01f} {units})",
            ],
            loc="upper right",
            fontsize=fs * 0.5,
        )


def plot_horizontal_histogram(
    ax, hist, bin_edges, fs, color="gray", plot_median=False, plot_mean=False, units=""
):
    # Create the bar plot
    ax.barh(
        bin_edges[:-1],
        hist,
        height=np.diff(bin_edges),
        left=0,
        align="edge",
        color=color,
    )

    values = [hist[i] * bin_edges[i] for i in range(len(hist))]
    mean = sum(values) / (sum(hist))

    cumulative_freq = np.cumsum(hist)
    index = np.argmax(cumulative_freq >= 0.5 * np.sum(hist))
    median = bin_edges[index] + (
        0.5 * np.sum(hist) - cumulative_freq[index - 1]
    ) / hist[index] * (bin_edges[index + 1] - bin_edges[index])

    def find_closest(lst, value):
        index = min(range(len(lst)), key=lambda i: abs(lst[i] - value))
        closest_element = lst[index]
        return index, closest_element

    if plot_median != False:
        ax.hlines(
            bin_edges[index],
            0,
            hist[index - 1],
            color="gray",
            linewidth=fs * 0.15,
            zorder=2,
        )
    if plot_mean != False:
        idx, val = find_closest(bin_edges, mean)
        ax.hlines(
            bin_edges[idx], 0, hist[idx], color="black", linewidth=fs * 0.15, zorder=2
        )

    custom_lines = [
        Line2D([0], [0], color="k", lw=fs * 0.4),
        Line2D([0], [0], color="gray", lw=fs * 0.4),
    ]

    if plot_mean != False and plot_median != False:
        ax.legend(
            custom_lines,
            [f"Mean ({mean:0.01f} {units})", f"Median ({median:0.01f} {units})"],
            loc="upper right",
            fontsize=fs * 0.8,
        )


def mhw_metrics(
    ds,
    distributions,
    ratio=1.65,
    fig_size=25,
    text_size=1.2,
    map_lims=[[0, 150], [0, 30], [1.5, 3], [50, 250]],
    savepath='',
):
    time_series = {
        variable: utils.weighted_mean(ds[variable])
        for variable in list(ds.data_vars.keys())
    }

    metrics = ["MHW", "mean_duration", "mean_anomaly", "cumulative_anomaly"]
    hist_metrics = [
        "MHW_days_year",
        "MHW_event_duration",
        "MHW_event_mean_anomaly",
        "MHW_anual_cumulative_anomaly",
    ]

    units = ["days", "days", "$^\circ \!$C", "$^\circ \!$C$\cdot$day"]
    labels = [
        "MHW (days)",
        "MHW dur. (days)",
        "Av. an. ($^\circ \!$C)",
        "Cum. an. ($^\circ \!$C$\cdot$day)",
    ]
    label_legs = ["MHW", "MHW duration", "Average anomaly", "Cumulative anomaly"]

    # map_lims = [[0, 150], [0, 30], [1.5, 3], [50, 250]]
    lims = [[0, 300], [0, 70], [0, 4], [0, 400]]
    x_axis = [i for i in range(int(ds.time.min()), int(ds.time.max()) + 1)]

    cmap = mpl.cm.get_cmap("seismic")
    fillcolors = [cmap(i) for i in [0.6, 0.75, 0.9, 0.95]]
    extent = [
        float(ds.lon.min()),
        float(ds.lon.max()),
        float(ds.lat.min()),
        float(ds.lat.max()),
    ]

    subplots = (4, 3)
    fs = np.sqrt(fig_size)
    fig = plt.figure(
        figsize=(np.sqrt(ratio * fig_size), np.sqrt(fig_size / ratio)),
        dpi=300,
        layout="constrained",
    )
    gs = mpl.gridspec.GridSpec(
        subplots[0], subplots[1], width_ratios=(3, 1.2, 2), figure=fig
    )

    ax = [[None] * subplots[1] for _ in range(subplots[0])]

    for i in range(subplots[0]):
        for j in range(subplots[1]):
            if j == 2:
                ax[i][j] = fig.add_subplot(gs[i, j], projection=ccrs.Mercator())
                ax[i][j].set_extent(extent, crs=ccrs.PlateCarree())
                ax[i][j].coastlines(linewidth=fs * 0.1)
                map_plot(
                    ds[metrics[i]].mean("time"),
                    ax=ax[i][2],
                    cmap=cmocean.cm.thermal,
                    fig=fig,
                    fs=fs,
                    label=labels[i],
                    lim=map_lims[i],
                )
            else:
                ax[i][j] = fig.add_subplot(gs[i, j])
                ax[i][j].grid(which="major", linewidth=fs * 0.015)
                ax[i][j].tick_params(
                    axis="both",
                    which="major",
                    labelsize=text_size * fs,
                    size=fs * 0.5,
                    width=fs * 0.15,
                )
                ax[i][j].xaxis.set_tick_params(which="minor", bottom=True)
                ax[i][j].yaxis.set_minor_locator(AutoMinorLocator())
                ax[i][j].yaxis.set_tick_params(
                    which="minor", left=True, size=fs * 0.3, width=fs * 0.1
                )

            if j == 1:
                ax[i][j].set_xticks([])
                ax[i][j].set_yticklabels([])
                ax[i][j].set_ylim(lims[i])
                colors = [fillcolors[0], "lightgray", "lightgray", "lightgray"]
                plot_horizontal_histogram(
                    ax[i][j],
                    distributions[hist_metrics[i]]["hist"],
                    distributions[hist_metrics[i]]["bin_edges"],
                    fs,
                    color=colors[i],
                    plot_median=True,
                    plot_mean=True,
                    units=units[i],
                )

    pos_err = np.abs(time_series[f"MHS_pos"] - time_series[f"MHS"])
    neg_err = np.abs(time_series[f"MHS_neg"] - time_series[f"MHS"])
    err_plot(
        ax[0][0],
        x_axis,
        time_series["MHS"],
        (pos_err, neg_err),
        fs,
        label_leg="MHS",
        color="gray",
    )

    i = 0
    for metric in ["MHW", "mean_duration", "mean_anomaly", "cumulative_anomaly"]:
        pos_err = np.abs(time_series[f"{metric}_pos"] - time_series[metric])
        neg_err = np.abs(time_series[f"{metric}_neg"] - time_series[metric])
        err_plot(
            ax[i][0],
            x_axis,
            time_series[metric],
            (pos_err, neg_err),
            fs,
            label_leg=label_legs[i],
            zorder=1,
        )
        ax[i][0].set_ylim(lims[i])
        ax[i][0].set_xlim([2001.9, 2021.1])
        ax[i][0].set_xticks(list(range(2002, 2021, 3)))
        ax[i][0].set_ylabel(labels[i], fontsize=text_size * fs)
        ax[i][0].legend(loc="upper left", fontsize=fs)
        if i != 3:
            ax[i][0].set_xticklabels([])

        i += 1

    ax[0][0].fill_between(
        x_axis,
        np.zeros(len(x_axis)),
        time_series["MHW"],
        color=fillcolors[0],
        alpha=1,
        linewidth=0,
        zorder=0,
        label="Moderate",
    )
    ax[0][0].fill_between(
        x_axis,
        np.zeros(len(x_axis)),
        time_series["MHW_cat_2"],
        color=fillcolors[1],
        alpha=1,
        linewidth=0,
        zorder=0,
        label="Strong",
    )
    # ax[0][0].fill_between(
    #     x_axis,
    #     np.zeros(len(x_axis)),
    #     time_series["MHW_cat_3"],
    #     color=fillcolors[3],
    #     alpha=1,
    #     linewidth=0,
    #     zorder=1,
    #     label="Severe",
    # )

    ax[0][0].legend(loc="upper left", fontsize=fs)

    ax[0][0].yaxis.set_minor_locator(AutoMinorLocator())

    plot_horizontal_histogram(
        ax[0][1],
        np.array(distributions["MHW_cat_2_days_year"]["hist"]) / 10,
        distributions["MHW_cat_2_days_year"]["bin_edges"],
        fs,
        color=fillcolors[1],
        plot_median=False,
        plot_mean=False,
    )

    # plot_horizontal_histogram(
    #     ax[0][1],
    #     np.array(distributions["MHW_cat_3_days_year"]["hist"]) / 10,
    #     distributions["MHW_cat_2_days_year"]["bin_edges"],
    #     fs,
    #     color=fillcolors[3],
    #     plot_median=False,
    #     plot_mean=False,
    # )

    if savepath!='':
        plt.savefig(savepath, bbox_inches="tight", dpi=300)


def season_plot(
    ds,
    median,
    quantiles,
    fig_size=20,
    ratio=1.2,
    cmap="jet",
    text_size=1.35,
    y_lim=[0, 1],
    y_lim_map=[0, 1],
    y_label="",
    label_pos="upper center",
    savepath = '',
):
    subplots = (3, 2)
    fs = np.sqrt(fig_size)
    fig = plt.figure(
        figsize=(np.sqrt(ratio * fig_size), np.sqrt(fig_size / ratio)),
        dpi=300,
        layout="constrained",
    )
    gs = mpl.gridspec.GridSpec(
        subplots[0], subplots[1], width_ratios=(1, 1), figure=fig
    )

    ax = [[None] * subplots[1] for _ in range(subplots[0])]
    extent = [
        float(ds.lon.min()) - 0.3,
        float(ds.lon.max()) + 0.3,
        float(ds.lat.min()) - 0.4,
        float(ds.lat.max()) + 0.3,
    ]

    seasons = ["DJF", "JJA", "MAM", "SON"]

    titles = ["Winter", "Summer", "Spring", "Autumn"]

    labels = ["Median", "Q25\u2014Q75", "Q05\u2014Q95"]

    k = 0
    for i in range(subplots[0]):
        for j in range(subplots[1]):
            if i in [0, 1]:
                ax[i][j] = fig.add_subplot(gs[j, i], projection=ccrs.Mercator())
                ax[i][j].set_extent(extent, crs=ccrs.PlateCarree())
                ax[i][j].coastlines(linewidth=fs * 0.1)
                map_plot(
                    ds.sel(season=seasons[k]),
                    ax=ax[i][j],
                    fig=fig,
                    fs=fs,
                    lim=y_lim_map,
                    shrink=0.7,
                    cmap=cmap,
                    cbar=False,
                )

                ax[i][j].text(
                    0.83,
                    0.77,
                    titles[k],
                    fontsize=fs * text_size,
                    transform=ax[i][j].transAxes,
                    va="bottom",
                    ha="left",
                )

                mean = np.float32(ds.sel(season=seasons[k]).mean())
                ax[i][j].text(
                    0.025,
                    0.075,
                    "Mean: {:0.02f} {}".format(mean, y_label),
                    fontsize=fs * text_size,
                    transform=ax[i][j].transAxes,
                    va="bottom",
                    ha="left",
                )

                k += 1

    cbar_ax = fig.add_axes([0.95, 0.4, 0.014, 0.57])
    norm = mpl.colors.Normalize(vmin=y_lim_map[0], vmax=y_lim_map[1])
    cbar = mpl.colorbar.ColorbarBase(
        cbar_ax, cmap=cmap, extend="max", norm=norm, orientation="vertical"
    )
    cbar.set_label(y_label, fontsize=fs * text_size * 1.1, rotation=1)
    cbar.ax.tick_params(labelsize=text_size * fs)

    ax[2][0] = fig.add_subplot(gs[2, :])
    ax[2][0].grid(which="major", linewidth=fs * 0.015)
    ax[2][0].tick_params(
        axis="both",
        which="major",
        labelsize=text_size * fs * 1.1,
        size=fs * 0.5,
        width=fs * 0.15,
    )
    ax[2][0].xaxis.set_tick_params(which="minor", bottom=True)
    ax[2][0].yaxis.set_minor_locator(AutoMinorLocator())
    ax[2][0].yaxis.set_tick_params(
        which="minor", left=True, size=fs * 0.3, width=fs * 0.1
    )
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "June",
        "July",
        "Aug",
        "Sept",
        "Oct",
        "Nov",
        "Dec",
    ]
    m_doy = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    tt = list(range(1, 365 + 1))
    (l1,) = ax[2][0].plot(
        tt[:365], median, color="k", linewidth=fs * 0.2, zorder=2, label=labels[0]
    )
    j = 0
    for filled_region in quantiles:
        ax[2][0].fill_between(
            tt[:365],
            filled_region[0],
            filled_region[1],
            alpha=0.5 - j * 0.15,
            color="gray",
            label=labels[j + 1],
        )
        j += 1

    ax[2][0].set_xticks(m_doy)
    ax[2][0].set_xticklabels(months, size=text_size * fs * 1.2)
    ax[2][0].yaxis.tick_right()
    ax[2][0].yaxis.set_label_position("right")
    ax[2][0].set_ylabel(y_label, size=text_size * fs * 1.1, rotation=0)
    ax[2][0].set_xlim(tt[0], tt[-1])
    ax[2][0].set_ylim(y_lim[0], y_lim[1])

    plt.legend(loc=label_pos, fontsize=text_size * fs)

    if savepath!='':
        plt.savefig(savepath, bbox_inches="tight", dpi=300)



def figure_S3(sst,err,extras=[],
              fig_size=20,
              ratio=1.5,
              text_size=0.25,
              text=None,
              title='',
              colors=['r','orange','green'],
              line_styles = ['-','--','--'],
              labels=[''],
              y_lim=[],
              savepath=''):

    """
    
    """
    fig, ax= plt.subplots(figsize=(np.sqrt(ratio*fig_size),np.sqrt(fig_size/ratio)), dpi=300)
    
    ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='minor', bottom=False)
    ax.tick_params(axis='both', which='major', labelsize=text_size*fig_size)

    ax.grid(which='major', linewidth=fig_size*0.0075)
    ax.grid(which='minor', axis='y', linewidth=fig_size*0.003)
    
    months=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July','Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    #List of days of year where month start
    m_doy=[1,32,60,91,121,152,182,213,244,274,305,335]
    
    if labels==['']:
        labs = labels*(2+len(extras))
    else:
        labs = labels
    
    tt=list(range(1, 365+1))
       
    lw = 0.015

    le = len(sst) 
    l1, = ax.plot(tt[:le], sst, 
                  linewidth=fig_size*lw,
                  zorder=10,
                  color='k',
                  label=labs[0]
                  )

    err = err[:le]
    ax.fill_between(tt[:le],sst-err,sst+err,
                    alpha=0.5,
                    color='gray',
                    label=labs[1])
    
    
    i=0
    for extra in extras:
           
        l2, = ax.plot(tt, extra, 
                      linewidth=fig_size*lw,
                      color=colors[i],
                      linestyle=line_styles[i],
                      label=labs[i+2]
                      )
        i+=1
        

        
    ax.set_xticks(m_doy)
    ax.set_xticklabels(months, size=text_size*fig_size)
    ax.set_xlim(tt[0], tt[-1])
    
    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])
    
    ax.locator_params(axis='x', nbins=12)
    ax.set_ylabel('$^\circ$$\!$C',size=text_size*fig_size, rotation=0)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    
    # ax.set_title('ºC',size=1.75*fig_size, x=-0.03, y=1.02)
    #ax.set_ylabel('ºC',size=1.75*fig_size)
    #ax.set_xlabel('Month',size=1.75*fig_size)
    
    if text!=None:
        ax.text(0.85,0.9, text, transform=ax.transAxes, fontsize=text_size*fig_size)
    
    4
    if labels!=['']:
        plt.legend(loc='upper left', fontsize=text_size*fig_size)
    
    if savepath!='':
        
        dir_path = os.path.dirname(os.path.realpath(savepath))+'/'
        if not os.path.exists(dir_path):  
            os.makedirs(dir_path)
        
        plt.savefig(savepath,dpi=300,bbox_inches='tight')
     
    return   
"""Project for the CMEPDA exam.
"""
import os
import sys
import argparse
import logging
import time
from collections import OrderedDict

import ROOT
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import joblib

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical, plot_model



#Necessary option in order for argparse to function, if not present ROOT will
#prevail over argparse when option are passed from command line
# ROOT.PyConfig.IgnoreCommandLineOptions = True

logger = logging.getLogger('higgs_2e2mu')
# The log file is in "write" mode, to return to "append" mode change w with a
hdlr = logging.FileHandler('higgs_2e2mu.log', mode='w')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

_description = "The program will perform standard analysis if no option is given."

#Include cpp library for RDataFrame modules
ROOT.gInterpreter.ProcessLine('#include "library.h"')

# Enable multi-threading
# ROOT.ROOT.EnableImplicitMT(4)
base_url = "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/"
#Imput links to signal, background and data samples
#Signal of Higgs -> 4 leptons
link_sig = "".join([base_url, "SMHiggsToZZTo4L.root"])
#Background of ZZ -> 4 leptons
link_bkg = "".join([base_url, "ZZTo2e2mu.root"])
#CMS data tacken in 2012 (11.6 fb^(-1) integrated luminosity)
link_data = ROOT.std.vector("string")(2)
link_data[0] = "".join([base_url, "Run2012B_DoubleMuParked.root"])
link_data[1] = "".join([base_url, "Run2012C_DoubleMuParked.root"])
#Dict of link
dataset_link = {"signal": link_sig, "background": link_bkg, "data": link_data}

def style(h, mode, Rcolor):
    """Set the basic style of a given histogram depending on which mode
    is required. (Lazy)

    Args:
        h (TH1D): Root histogram to style.
        mode (str): "e" for empty style, "f" for filled style,
                       "m" for marker style.

        Rcolor (Tcolor): Root color to apply.

    Returns:
        TH1D: styled histogram
    """
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(1)
    ROOT.gStyle.SetTextFont(42)
    h.SetStats(0)
    h.SetLineStyle(1)
    h.SetLineWidth(1)
    h.GetYaxis().SetTitle("Counts")

    #empty histogram
    if mode == "e":
        h.SetLineColor(Rcolor)
        h.SetLineWidth(2)

    #full histogram
    elif mode == "f":
        h.SetFillStyle(1001)
        h.SetFillColor(Rcolor)
        h.SetLineColor(ROOT.kBlack)

    #mark histogram
    elif mode == "m":
        h.SetMarkerStyle(20)
        h.SetMarkerSize(1.0)
        h.SetMarkerColor(Rcolor)
        h.SetLineColor(Rcolor)

    return h

def snapshot(df, filename):
    """Make a snapshot of RDataFrame to local file. (Lazy)

    Args:
        df (RDataFrame): RDF to save.
        filename (str): name of the file to save

    Returns:
        RDataFrame: same RDF as input.
    """
    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    snapshotOptions.fLazy = True
    # Former version selected only requested columns but there is a problem
    # in ROOT with some columns name that raise an error.
    # To avoid it, all columns are selected instead.
    # columns = ROOT.vector("string")()
    # for branch in branches_list:
    #     columns.push_back(branch)
    # df_out = df.Snapshot("Events", filename, columns, snapshotOptions)
    df_out = df.Snapshot("Events", filename, "", snapshotOptions)
    return df_out

def df_prefilter(df_key):
    """Prefilter RDF selecting events with 2 or more electrons and
    2 or more muons.

    Args:
        df_key (str): key to access global dataset_link dictionary
                         containing the chosen dataset (e.g. "signal").

    Returns:
        RDataFrame: Filtered RDF of chosen dataset.
    """
    # In this code we will consider only events with 2 or more
    # electrons and muons
    df_net = ROOT.RDataFrame("Events", dataset_link.get(df_key))
    df_2e2m = df_net.Filter("nElectron>=2 && nMuon>=2",
                            "Events with at least two Electrons and Muons")
    return df_2e2m

def access_df(df_key, local_flag, mode):
    """Access given dataset from appropriate location. Download to local
    file if needed. (Lazy)

    Args:
        df_key (str): label of the chosen dataset
                         ("signal", "background" or "data").
        local_flag (bool): boolean flag to save local file or not.
        mode (str): tag to identify preliminar or standard analysis
                    ("p" or "std")

    Returns:
        RDataFrame: requested RDF
    """
    if local_flag:
        filename = f"{df_key}_{mode}.root"
        # Try/except method difficult to use because ROOT exceptions are not easily
        # catched by python
        if os.path.isfile(filename):
            df_local = ROOT.RDataFrame("Events", filename)
            df_out = df_local
        else:
            logger.warning(f"Problem loading local df:{df_key}.\n Starting download")
            df_2e2m = df_prefilter(df_key)
            df_out = snapshot(df_2e2m, filename)
    else:
        df_out = df_prefilter(df_key)
    return df_out

def preliminar_request(flag):
    """First look at the chosen input file. (Lazy)

    Prompt the user to select dataset and branche(s) to preview.

    This function is meant to explore only first hand characteristics of
    events and by default it will show three variables reconstructed
    form the components present in datasets.

    Prepare histograms of chosen data.

    Args:
        flag (bool): boolean flag to save local file or not.

    Returns:
        str: key to access the dataset (e.g. "signal").
        list: selected branches list.
        list: selected THisto1D list.

    Raises: key error if user insert an invalid key.
    """
    key_df = input("Insert the data frame you want to look at first "
                   f"choosing from this list:\n{dataset_link.keys()}\n")

    try:
        #3D reconstruction of some fundamental variables and its Histograms
        df_In = access_df(key_df, flag, "p")
        # These are the default branches that are reconstructed regardless the input given
        df_In3d = df_In.Define("PV_3d", "sqrt(PV_x*PV_x + PV_y*PV_y + PV_z*PV_z)")\
                       .Define("Muon_ip3d", "sqrt(Muon_dxy*Muon_dxy + Muon_dz*Muon_dz)")\
                       .Define("El_ip3d", "sqrt(Electron_dxy*Electron_dxy + Electron_dz*Electron_dz)")

        # In this way nothing is saved locally but the original list of available
        # columns can be shown
        df = access_df(key_df, False, "p")
        list_branches = df.GetColumnNames()

        # Decide which variables to look first
        dictOfBranches = {i:list_branches[i] for i in range(0, len(list_branches))}
        list_In = input("Insert the variable numbers to look at, separated by a space"
                        f"press return when you are done:\n{dictOfBranches}\n")

        # Control input and retrieve the required branches
        list_str = list_In.split(" ")
        b_In = ["PV_3d", "Muon_ip3d", "El_ip3d"]
        for i in list_str:
            if i.isdigit() and int(i) < 32:
                current_branch = dictOfBranches[int(i)]
                b_In.append(current_branch)
            else:
                logger.warning(f"Error! {i} is an invalid key!")

        # Make sure that each branch is taken only once
        unique_b_In = list(set(b_In))
        logger.info(f"These are the chosen branches plus the default 3:{unique_b_In}")

        # Require the chosen df and request the histos
        h_In = []
        for branch in unique_b_In:
            current_histo = df_In3d.Histo1D(branch)
            h_In.append(current_histo)

    except KeyError as e:
        print(f"Cannot read the given key!\n{e}")
        sys.exit(1)
    return key_df, unique_b_In, h_In

def preliminar_plot(dataset, branch_histo, histo_data):
    """Plot preliminar analysis histograms of the selected dataset
    and branches.
    The histograms are saved both in a .pdf and in a .root file
    in order to be further investigated.

    Args:
        dataset (str): selected dataset name.
        branch_histo (list): list of selected branches.
        histo_data (TH1D): histogram of given dataset and branches.
    """
    # General Canvas Settings
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetTextFont(42)
    c_histo = ROOT.TCanvas(f"c_histo_{branch_histo}", "", 800, 700)
    c_histo.cd()

    # Set and Draw histogram for data points
    x_max = histo_data.GetXaxis().GetXmax()
    x_min = histo_data.GetXaxis().GetXmin()
    logger.info(f"One day minumun and maximun of the {branch_histo}"
                f" will be useful...but it is NOT This day!{x_max,x_min}")
    # histo_data.GetXaxis().SetRangeUser(x_min,x_max)

    histo_data.SetLineStyle(1)
    histo_data.SetLineWidth(1)
    histo_data.SetLineColor(ROOT.kBlack)
    histo_data.Draw()

    # Add Legend
    legend = ROOT.TLegend(0.7, 0.5, 0.85, 0.7)
    legend.SetFillColor(0)
    legend.SetBorderSize(1)
    legend.SetLineWidth(0)
    legend.SetTextSize(0.04)
    legend.AddEntry(histo_data, dataset)
    legend.Draw()

    #Save plot
    c_histo.SaveAs(f"{branch_histo}_{dataset}.pdf")
    c_histo.SaveAs(f"{branch_histo}_{dataset}.root")

def preliminar_retrieve(key_df, branches, histos):
    """Retrieve and plot requested histograms. (Instant)

    Args:
        key_df (str): dataset key to access (e.g. "signal").
        branches (list): list of branches to process.
        histos (list): list of THisto1D to draw and save.
    """
    # Trigger event loop and plot
    for branch, hist in zip(branches, histos):
        h = hist.GetValue()
        preliminar_plot(key_df, branch, h)

def standard_define(df):
    """Define and add to the input RDF the variables that
    will be used for filter events in the standard analysis. (Lazy)

    All the variables that are more significant for the analysis, such as
    Z, H and the angular variables are regrouped and defined in a
    specific function to make clearer the main points of the workflow
    and to be more easily accessible from the ML functions.

    Args:
        df (RDataFrame): RDF to add new variables to.

    Returns:
        RDataFrame: RDF with added variables.
    """
    el_sip3d = "Electron_ip3d/sqrt(Electron_dxyErr*Electron_dxyErr+ Electron_dzErr*Electron_dzErr)"
    el_ip3d = "sqrt(Electron_dxy*Electron_dxy + Electron_dz*Electron_dz)"
    mu_ip3d = "sqrt(Muon_dxy*Muon_dxy + Muon_dz*Muon_dz)"
    mu_sip3d = "Muon_ip3d/sqrt(Muon_dxyErr*Muon_dxyErr + Muon_dzErr*Muon_dzErr)"

    df = df.Define("Electron_ip3d", el_ip3d)\
           .Define("Electron_sip3d", el_sip3d)\
           .Define("Muon_ip3d", mu_ip3d)\
           .Define("Muon_sip3d", mu_sip3d)

    return df

def show_cut(df_2e2m):
    """Compare unfiltered and filtered data for each filter,
    considering the main cuts used in the analysis published on CERN Open Data.

    Args:
        df_2e2m (RDataFrame): RDF to be filtered.

    Returns:
        dict: filters dictionary.
              The keys are the filters, the values are lists containing:
              unfiltered histos, filtered histos, filter RDF Reports.
              Each single filter contains the histograms of all the
              variables that define it.
    """
    #1st filter: Good isolation it's not symmetrical for electron and muon
    h_unfil_eliso3 = df_2e2m.Histo1D(("h_eliso3", "", 56, -0.05, 1.2), "Electron_pfRelIso03_all")
    h_unfil_muiso4 = df_2e2m.Histo1D(("h_muiso4", "", 56, -0.05, 1.2), "Muon_pfRelIso04_all")

    df_eliso3 = df_2e2m.Filter("All(abs(Electron_pfRelIso03_all)<0.40)", "ElIso03 cut")
    h_fil_eliso3 = df_eliso3.Histo1D(("h_eliso3", "", 56, -0.05, 1.2), "Electron_pfRelIso03_all")
    df_muiso4 = df_2e2m.Filter("All(abs(Muon_pfRelIso04_all)<0.40)", "MuIso04 cut")
    h_fil_muiso4 = df_muiso4.Histo1D(("h_muiso4", "", 56, -0.05, 1.2), "Muon_pfRelIso04_all")

    filter_dict = {"Isolation":[[h_unfil_eliso3, h_unfil_muiso4],
                                [h_fil_eliso3, h_fil_muiso4],
                                [df_eliso3.Report(), df_muiso4.Report()]],
                   "Eta":[[], [], []], "Dr":[[], [], []], "Pt":[[], [], []],
                   "Electron_track":[[], [], []], "Muon_track":[[], [], []],
                   "Z_mass":[[], [], []]}

    # Dataframe for filter in pt, it's symmetrical but involves mu and e at the same time
    df_pt = df_2e2m.Filter("pt_cut(Muon_pt, Electron_pt)", "Pt cuts")

    #Filters that either are simmetrical or don't involve mu and e at the same time
    for part in ["Electron", "Muon"]:
        #2nd filter:Eta cut
        h_unfil_eta = df_2e2m.Histo1D((f"h_{part}_eta", "", 56, -3, 3), f"{part}_eta")
        df_eta = df_2e2m.Filter(f"All(abs({part}_eta)<2.5)", f"{part}_Eta cut")
        h_fil_eta = df_eta.Histo1D((f"h_{part}", "", 56, -3, 3), f"{part}_eta")
        filter_dict["Eta"][0].append(h_unfil_eta)
        filter_dict["Eta"][1].append(h_fil_eta)
        filter_dict["Eta"][2].append(df_eta.Report())
        #3rd filter:Dr cut
        df_dr = df_2e2m.Define(f"{part}_dr", f"dr_def({part}_eta, {part}_phi)")
        h_unfil_dr = df_dr.Histo1D((f"h_{part}", "", 56, -0.5, 5.5), f"{part}_dr")
        df_fil_dr = df_dr.Filter(f"{part}_dr>=0.02", f"{part}_Dr cut")
        h_fil_dr = df_fil_dr.Histo1D((f"{part}", f"{part}_dr", 56, -0.5, 5.5), f"{part}_dr")
        filter_dict["Dr"][0].append(h_unfil_dr)
        filter_dict["Dr"][1].append(h_fil_dr)
        filter_dict["Dr"][2].append(df_fil_dr.Report())
        #4th filter:Pt cut
        h_unfil_pt = df_2e2m.Histo1D((f"h_{part}pt", "", 56, -0.5, 120), f"{part}_pt")
        h_fil_pt = df_pt.Histo1D((f"h_{part}_pt", "", 56, -0.5, 120), f"{part}_pt")
        filter_dict["Pt"][0].append(h_unfil_pt)
        filter_dict["Pt"][1].append(h_fil_pt)
        filter_dict["Pt"][2].append(df_pt.Report())
        #5th filter: Track
        h_unfil_sip3d = df_2e2m.Histo1D((f"h_{part}sip3d", "", 56, -0.1, 3), f"{part}_sip3d")
        h_unfil_dxy = df_2e2m.Histo1D((f"h_{part}_dxy", "", 56, -0.02, 0.02), f"{part}_dxy")
        h_unfil_dz = df_2e2m.Histo1D((f"h_{part}_dz", "", 56, -0.02, 0.02), f"{part}_dz")

        df_sip3d = df_2e2m.Filter(f"All({part}_sip3d<4)", f"{part}_Sip3d cut")
        h_fil_sip3d = df_sip3d.Histo1D((f"h_{part}_sip3d", "", 56, -0.1, 3), f"{part}_sip3d")
        df_dxy = df_2e2m.Filter(f"All(abs({part}_dxy)<0.5)", f"{part}_Dxy cut")
        h_fil_dxy = df_dxy.Histo1D((f"h_{part}_dxy", "", 56, -0.02, 0.02), f"{part}_dxy")
        df_dz = df_2e2m.Filter(f"All(abs({part}_dz)<1.0)", f"{part}_Dz cut")
        h_fil_dz = df_dz.Histo1D((f"h_{part}_dz", "", 56, -0.02, 0.02), f"{part}_dz")
        filter_dict[f"{part}_track"][0].extend([h_unfil_sip3d, h_unfil_dxy, h_unfil_dz])
        filter_dict[f"{part}_track"][1].extend([h_fil_sip3d, h_fil_dxy, h_fil_dz])
        filter_dict[f"{part}_track"][2].extend([df_sip3d.Report(), df_dxy.Report(), df_dz.Report()])

    # Compute z masses and filter
    df_z_mass = df_2e2m.Define("Z_mass", "z_mass(Electron_pt, Electron_eta,"\
                               " Electron_phi, Electron_mass,"\
                               "Muon_pt, Muon_eta, Muon_phi, Muon_mass)")
    h_unfil_z = df_z_mass.Histo1D(("h_Z_mass", "", 56, -5, 140), "Z_mass")
    # 6th filter: z masses
    df_z = df_z_mass.Filter("Z_mass[0] > 40 && Z_mass[0] < 120"\
                            " && Z_mass[1] > 12 && Z_mass[1] < 120",
                            "First candidate in [40, 120]"\
                            " and Second candidate in [12, 120]")
    h_fil_z = df_z.Histo1D(("h_Z_mass", "", 56, -5, 140), "Z_mass")
    filter_dict["Z_mass"][0].append(h_unfil_z)
    filter_dict["Z_mass"][1].append(h_fil_z)
    filter_dict["Z_mass"][2].append(df_z.Report())
    return filter_dict

def good_events(df_2e2m):
    """Select the events that pass the cuts used in the 2012 CERN article.
    (Lazy)

    Args:
        df_2e2m (RDataFrame): Prefiltered (2 e and 2 mu) RDF.

    Returns:
        RdataFrame: Filtered RDF
    """

    #angular cuts
    df_eta = df_2e2m.Filter("All(abs(Electron_eta)<2.5) && All(abs(Muon_eta)<2.4)",
                            "Eta_cuts")
    #transvers momenta cuts
    df_pt = df_eta.Filter("pt_cut(Muon_pt, Electron_pt)",
                          "Pt cuts")
    df_dr = df_pt.Filter("dr_cut(Muon_eta, Muon_phi, Electron_eta, Electron_phi)",
                         "Dr_cuts")
    #Request good isolation
    df_iso = df_dr.Filter("All(abs(Electron_pfRelIso03_all)<0.40) &&"
                          "All(abs(Muon_pfRelIso04_all)<0.40)",
                          "Require good isolation")
    #Filter of Muon and Electron tracks
    df_el_track = df_iso.Filter("All(Electron_sip3d<4) &&"
                                " All(abs(Electron_dxy)<0.5) && "
                                " All(abs(Electron_dz)<1.0)",
                                "Electron track close to primary vertex")
    df_mu_track = df_el_track.Filter("All(Muon_sip3d<4) && All(abs(Muon_dxy)<0.5) &&"
                                     "All(abs(Muon_dz)<1.0)",
                                     "Muon track close to primary vertex")
    df_2p2n = df_mu_track.Filter("Sum(Electron_charge) == 0 && Sum(Muon_charge) == 0",
                                 "Two opposite charged electron and muon pairs")

    return df_2p2n

def reco_define(df_2e2mu):
    """Defines all the main variables, such as Z, H and its angular
    variables. (Lazy)

    Args:
        df (RDataFrame): RDF to add new variables to.

    Returns:
        RDataFrame: RDF with added variables.
    """
    z_mass_reco = "z_mass(Electron_pt, Electron_eta, Electron_phi, Electron_mass,"\
                         "Muon_pt, Muon_eta, Muon_phi, Muon_mass)"
    h_mass_reco = "h_mass(Electron_pt, Electron_eta, Electron_phi, Electron_mass,"\
                         "Muon_pt, Muon_eta, Muon_phi, Muon_mass)"
    costheta_star_reco = "costheta_star(Electron_pt, Electron_eta, Electron_phi, Electron_mass,"\
                         "Muon_pt, Muon_eta, Muon_phi, Muon_mass)"
    phi_reco = "measure_phi(Electron_pt, Electron_eta, Electron_phi, Electron_mass, Electron_charge,"\
                            "Muon_pt, Muon_eta, Muon_phi, Muon_mass, Muon_charge)[0]"
    phi1_reco = "measure_phi(Electron_pt, Electron_eta, Electron_phi, Electron_mass, Electron_charge,"\
                            "Muon_pt, Muon_eta, Muon_phi, Muon_mass, Muon_charge)[1]"
    costheta1_reco = "costheta(Electron_pt, Electron_eta, Electron_phi, Electron_mass, Electron_charge,"\
                              "Muon_pt, Muon_eta, Muon_phi, Muon_mass, Muon_charge)[0]"
    costheta2_reco = "costheta(Electron_pt, Electron_eta, Electron_phi, Electron_mass, Electron_charge,"\
                              "Muon_pt, Muon_eta, Muon_phi, Muon_mass, Muon_charge)[1]"

    df_z_mass = df_2e2mu.Define("Z_mass", z_mass_reco)\
                        .Define("H_mass", h_mass_reco)\
                        .Define("CosTheta_star", costheta_star_reco)\
                        .Define("Phi", phi_reco)\
                        .Define("Phi1", phi1_reco)\
                        .Define("CosTheta1", costheta1_reco)\
                        .Define("CosTheta2", costheta2_reco)
    return df_z_mass

def reco_higgs(df, weight, weight_ang):
    """Recontruct the Higgs mass and the angular variables. (Lazy)

    Args:
        df (RDataFrame): RDF to process.
        weight (float): Weight values to rescale "sig" and "bkg"
                        for Higgs mass histogram.
        weight_ang (float): Weight values to rescale "sig" and "bkg"
                            for angular variables histograms.

    Returns:
        THist1D: Histogram of the reconstructed Higgs mass.
        RResultPtr< RCutFlowReport >: Report of filters applied.
        list: list of 5 angular variables histograms.
    """
    # Selection of only the potential good events
    df_base = good_events(df)

    # Compute z and H masses from it
    df_z_mass = reco_define(df_base)
    #Filter on z masses
    df_z1 = df_z_mass.Filter("Z_mass[0] > 40 && Z_mass[0] < 120", "First candidate in [40, 120]")
    df_z2 = df_z1.Filter("Z_mass[1] > 12 && Z_mass[1] < 120", "Second candidate in [12, 120]")

    # Define weights
    df_reco = df_z2.Define("weight", f"{weight}").Define("weight_ang", f"{weight_ang}")
    h_reco_higgs = df_reco.Histo1D(("h_sig_2el2mu", "", 36, 70, 180), "H_mass", "weight")

    # Reconstruction of angular variables
    h_costheta_star = df_reco.Histo1D(("h_costheta_star", "CosTheta_star", 56, -1, 1), "CosTheta_star", "weight_ang")
    h_costheta1 = df_reco.Histo1D(("h_costheta1", "CosTheta1", 56, -1, 1), "CosTheta1", "weight_ang")
    h_costheta2 = df_reco.Histo1D(("h_costheta2", "CosTheta2", 56, -1, 1), "CosTheta2", "weight_ang")
    h_phi = df_reco.Histo1D(("h_phi", "Phi", 56, -3.14, 3.14), "Phi", "weight_ang")
    h_phi1 = df_reco.Histo1D(("h_phi1", "Phi1", 56, -3.14, 3.14), "Phi1", "weight_ang")
    list_h_ang = [h_costheta_star, h_costheta1, h_costheta2, h_phi, h_phi1]

    # Filter on a narrow window of the Higgs mass if applied before lowers
    # too much the events available to make decent plots.
    # The report will be useful in the ml analysis.
    df_reco_h = df_reco.Filter("H_mass > 110 && H_mass <140",
                               "Cut on a narrow window: H_mass in [110, 140]")
    report_higgs = df_reco_h.Report()

    return h_reco_higgs, report_higgs, list_h_ang

def standard_request(flag):
    """Prepares all the necessary requests for the standard analysis.
    (Lazy)

    Args:
        flag (bool): boolean flag to save local file or not.

    Returns:
        dict: dictionary of "sig" and "bkg" filters.
        list: list of 3 THisto1D of Higgs mass ("sig", "bkg", "data").
        list: list of 3 RDF Report of Higgs mass ("sig", "bkg", "data").
        RDataFrame: RDF for ML analysis.
        OrderedDict: angular variables ordered dictionary.
                     The keys are the angular variables, the values are
                     lists of 3 THisto1D of the given variable
                     ("sig", "bkg", "data").
    """
    df_s = access_df("signal", flag, "std")
    df_b = access_df("background", flag, "std")
    df_d = access_df("data", flag, "std")

    df_s1 = standard_define(df_s)
    df_b1 = standard_define(df_b)
    df_d1 = standard_define(df_d)

    start1 = time.time()
    #Request filtered and unfiltered data
    dict_filter = {}
    dict_filter["sig"] = show_cut(df_s1)
    dict_filter["bkg"] = show_cut(df_b1)

    # Request ml dataframe
    df_ml = [df_s, df_b]

    # Weights
    luminosity = 11580.0  # Integrated luminosity of the data samples
    xsec_sig = 0.0065  # ZZ->2el2mu: Standard Model cross-section
    nevt_sig = 299973.0  # ZZ->2el2mu: Number of simulated events
    scale_ZZTo4l = 1.386  # ZZ->4l: Scale factor for ZZ to four leptons
    xsec_bkg = 0.18  # ZZ->2el2mu: Standard Model cross-section
    nevt_bkg = 1497445.0  # ZZ->2el2mu: Number of simulated events
    weight_sig = luminosity * xsec_sig / nevt_sig
    weight_bkg = luminosity * xsec_bkg * scale_ZZTo4l / nevt_bkg
    weight_data = weight_ang_data = 1.0

    weight_ang_sig = 91/12065
    weight_ang_bkg = 91/51237

    # Request all the necessary to reconstruct the Higgs mass
    h_sig_higgs, report_sig, h_sig_ang = reco_higgs(df_s1, weight_sig, weight_ang_sig)
    h_bkg_higgs, report_bkg, h_bkg_ang = reco_higgs(df_b1, weight_bkg, weight_ang_bkg)
    h_data_higgs, report_data, h_data_ang = reco_higgs(df_d1, weight_data, weight_ang_data)
    stop1 = time.time()
    dict_angular = OrderedDict({"costheta_star":[], "costheta1":[], "costheta2":[],
                                "phi":[], "phi1":[]})
    for idx, h_list in enumerate(dict_angular.values()):
        h_list.extend([h_sig_ang[idx], h_bkg_ang[idx], h_data_ang[idx]])

    list_higgs = [h_sig_higgs, h_bkg_higgs, h_data_higgs]
    list_rep_higgs = [report_sig, report_bkg, report_data]

    logger.info(f"standard request time: {stop1-start1:.2f}s")
    return dict_filter, list_higgs, list_rep_higgs, df_ml, dict_angular

def filtered_plot(h_sig, h_bkg, rep_s, rep_b, fil):
    """Generate histogram(s) figure for given filter. Save to file.

    Multiple histograms (filter with more than one variable) are generated
    on separated plots and put in the same figure.

    Args:
        h_sig (list): list of unfiltered and filtered histogram(s) of "sig".
        h_bkg (list): list of unfiltered and filtered histogram(s) of "bkg".
        rep_s (list): list of filter report(s) of "sig".
        rep_b (list): list of filter report(s) of "bkg".
        fil (str): filter name, used to set title of plot and save file.
    """
    h_uf_s, h_f_s = h_sig
    h_uf_b, h_f_b = h_bkg

    # Add canvas
    canvas = ROOT.TCanvas("canvas", "", 800, 500)
    canvas.cd()

    # Add Legend
    legend = ROOT.TLegend(0.18, 0.7, 0.33, 0.85)
    legend.SetFillColor(0)
    legend.SetBorderSize(1)
    legend.SetLineWidth(1)
    legend.SetTextSize(0.025)

    delta_y = 1/len(h_f_s)
    # Add latek note
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)

    units_dict = {"Isolation":["Electron PfRellIso3", "Muon PfRellIso4"],
                  "Eta":["Electron Eta [rad]", "Muon Eta [rad]"],
                  "Dr":["Electron #DeltaR [rad]", "Muon #DeltaR [rad]"],
                  "Pt":["Electron Pt [GeV]", "Muon Pt [GeV]"],
                  "Muon_track":["Muon_sip3d", "Muon dxy [cm]", "Muon dz [cm]"],
                  "Electron_track":["Electron_sip3d", "Electron dxy [cm]", "Electron dz [cm]"],
                  "Z_mass":["Z mass [GeV]"]}

    for i, (h1, h2, h3, h4, rep12, rep34) in enumerate(zip(h_uf_s, h_f_s,
                                                           h_uf_b, h_f_b,
                                                           rep_s, rep_b)):
        canvas.cd()
        h_ustyle_s = style(h1, "e", ROOT.kRed)
        h_fstyle_s = style(h2, "f", ROOT.kRed)
        h_ustyle_b = style(h3, "e", ROOT.kAzure)
        h_fstyle_b = style(h4, "f", ROOT.kAzure)

        pad = ROOT.TPad(f"pad_{i}", f"pad_{i}", 0, i*delta_y, 1, (1+i)*delta_y)
        # pad.SetTopMargin(0)
        pad.Draw()
        pad.cd()
        h_ustyle_b.GetXaxis().SetTitle(f"{units_dict[fil][i]}")
        h_ustyle_b.Draw()
        h_fstyle_b.Draw("SAME")
        h_ustyle_s.Draw("SAME")
        h_fstyle_s.Draw("SAME")
        cut_eff_s = []
        cut_eff_b = []
        for cur_rep_s, cur_rep_b in zip(rep12, rep34):
            cut_eff_s.append(cur_rep_s.GetEff())
            cut_eff_b.append(cur_rep_b.GetEff())

        latex.DrawText(0.68, 0.67, f"Sig evt pass: {cut_eff_s[0]:.2f}%")
        latex.DrawText(0.68, 0.73, f"Bkg evt pass: {cut_eff_b[0]:.2f}%")

        sig_pass = h_fstyle_s.GetEntries()/h_ustyle_s.GetEntries()*100
        bkg_pass = h_fstyle_b.GetEntries()/h_ustyle_b.GetEntries()*100
        latex.DrawText(0.68, 0.79, f"Sig cnt pass: {sig_pass:.2f}%")
        latex.DrawText(0.68, 0.85, f"Bkg cnt pass: {bkg_pass:.2f}%")

    h_ustyle_b.SetTitle(f"{fil}")
    legend.AddEntry(h_uf_s[0], "Signal Unfilterd")
    legend.AddEntry(h_f_s[0], "Signal Filtered")
    legend.AddEntry(h_uf_b[0], "Bkg Unfilterd")
    legend.AddEntry(h_f_b[0], "Bkg Filtered")
    legend.Draw()

    #Save plot
    canvas.SaveAs(f"{fil}.pdf")

def higgs_plot(list_histo_higgs, filename):
    """Plot reconstructed Higgs mass for signal, background and data.
    Save figure to file.

    Args:
        list_histo_higgs (list): list of histograms of Higgs mass
                                 of "sig", "bkg" and "data".
        filename (str): name of file to save.
    """


    titles_units_dict = {"costheta_star":["cos#Theta*", ""], "costheta1":["cos#Theta_{1}", ""],
                         "costheta2":["cos#Theta_{2}", ""],
                         "phi":["#Phi", "[rad]"], "phi1":["#Phi_{1}", "[rad]"],
                         "h_mass": ["M_{2e2#mu}", "[GeV]"]}

    # Add canvas
    canvas_s = ROOT.TCanvas("canvas_s", "", 800, 700)
    #  header = ROOT.TLatex()

    h_signal = style(list_histo_higgs[0], "e", ROOT.kRed)
    h_background = style(list_histo_higgs[1], "f", ROOT.kAzure)
    h_data = style(list_histo_higgs[2], "m", ROOT.kBlack)

    if filename == "h_mass":
        h_background.GetYaxis().SetRangeUser(0, 5)
        h_background.SetTitle("H#rightarrowZZ#rightarrow2e2#mu")
    else:
        h_background.GetYaxis().SetRangeUser(0, 10)
        h_background.SetTitle(f"{titles_units_dict[filename][0]}")

    h_background.GetXaxis().SetTitle(f"{' '.join(titles_units_dict[filename])}")
    h_background.GetYaxis().SetTitle("Normalized event counts")

    canvas_s.cd()
    h_background.Draw("HIST")
    h_signal.Draw("HIST SAME")
    h_data.Draw("PE1 SAME")

    # Add Legend
    legend = ROOT.TLegend(0.7, 0.7, 0.85, 0.85)
    legend.SetFillColor(0)
    legend.SetBorderSize(1)
    legend.SetLineWidth(1)
    legend.SetTextSize(0.025)

    legend.AddEntry(list_histo_higgs[0], "Signal")
    legend.AddEntry(list_histo_higgs[1], "Background")
    legend.AddEntry(list_histo_higgs[2], "Data")
    legend.Draw()

    if filename == "h_mass":
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.02)
        latex.DrawLatexNDC(0.7, 0.92, "#sqrt{s} = 8 TeV, L_{int} = 11.6 fb^{-1}")

    #Save plot
    canvas_s.SaveAs(f"{filename}.pdf")

def standard_retrieve(filters, h_higgs, rep_higgs, angular):
    """Retrieve and plot requested histograms of Higgs mass and
    angular variables. (Instant)

    Generate 6 histogram plots: one for Higgs mass and one for each
    of the angular variables. Each histogram contains 3 datasets.

    If the code is in only standard mode, here the event loop will be triggered

    Args:
        filters (dict): dictionary of "sig" and "bkg" filters.
        h_higgs (list): list of 3 THisto1D of Higgs mass ("sig", "bkg", "data").
        rep_higgs (list): list of 3 RDF Report of Higgs mass ("sig", "bkg", "data").
        angular (OrderedDict): angular variables ordered dictionary.
                               The keys are the angular variables, the values are
                               lists of 3 THisto1D of the given variable
                               ("sig", "bkg", "data").
    """
    start2 = time.time()
    filters_data = {"sig":{"Isolation":[[], [], []], "Eta":[[], [], []],
                           "Dr":[[], [], []], "Pt":[[], [], []],
                           "Electron_track":[[], [], []], "Muon_track":[[], [], []],
                           "Z_mass":[[], [], []]},
                    "bkg":{"Isolation":[[], [], []], "Eta":[[], [], []],
                           "Dr":[[], [], []], "Pt":[[], [], []],
                           "Electron_track":[[], [], []], "Muon_track":[[], [], []],
                           "Z_mass":[[], [], []]}}
    for cut in filters["sig"].keys():
        for ch in filters.keys():
            for i, h_list in enumerate(filters[ch][cut][:2]):
                for histo in h_list:
                    filters_data[ch][cut][i].append(histo.GetValue())
            filters_data[ch][cut][2] = filters[ch][cut][2]
        h_sig = filters_data["sig"][cut][:2]
        rep_sig = filters_data["sig"][cut][2]

        h_bkg = filters_data["bkg"][cut][:2]
        rep_bkg = filters_data["bkg"][cut][2]

        filtered_plot(h_sig, h_bkg, rep_sig, rep_bkg, cut)

    list_higgs_data = []
    for h in h_higgs:
        h_higgs_data = h.GetValue()
        list_higgs_data.append(h_higgs_data)
    higgs_plot(list_higgs_data, "h_mass")

    angular_data = OrderedDict({"costheta_star":[], "costheta1":[], "costheta2":[],
                                "phi":[], "phi1":[]})
    for key, h_list in angular.items():
        for hist in h_list:
            angular_data[key].append(hist.GetValue())
        higgs_plot(angular_data[key], key)

    # Print on shell the complete report on the standard analysis
    for ch, rep in zip(["signal", "bakground", "data"], rep_higgs):
        print(f"{ch} report:\n")
        rep.Print()
        print()
    stop2 = time.time()
    logger.info(f"Standard retrieve time: {stop2 - start2:.2f}s")

def ml_request(ml_data):
    """Prepare dataset to be readable by the machine learning  method.
    (Lazy)

    Check if local dataset are already present (from previous execution
    or manual import) and load the RDF to return.

    If not then filter data and define variables useful for ML training.
    Save to local file (through lazy Snapshot) the RDF to return.

    Perform additional filter on Higgs mass only to produce Reports, the
    output RDF are not filtered on it.

    Args:
        ml_data (list): list of RDF to process.

    Returns:
        list: list of prepared RDF
        list: list of Report H mass filters on the prepared RDF
    """
    list_df_train = []
    if os.path.isfile("train_signal.root") and os.path.isfile("train_background.root"):
        list_df_train.extend([ROOT.RDataFrame("Events", "train_signal.root"),
                              ROOT.RDataFrame("Events", "train_background.root")])
    else:
        # These are all the variables that will be put in the training df
        ml_var = ["Muon_pt_1", "Muon_pt_2", "Electron_pt_1", "Electron_pt_2",
                  "Muon_mass_1", "Muon_mass_2", "Electron_mass_1", "Electron_mass_2",
                  "Muon_eta_1", "Muon_eta_2", "Electron_eta_1", "Electron_eta_2",
                  "Muon_phi_1", "Muon_phi_2", "Electron_phi_1", "Electron_phi_2",
                  "CosTheta_star", "CosTheta1", "CosTheta2", "Phi", "Phi1", "H_mass"]
        columns = ROOT.std.vector("string")()
        for column in ml_var:
            columns.push_back(column)

        snapshotOptions = ROOT.RDF.RSnapshotOptions()
        snapshotOptions.fLazy = True

        for df, df_key in [[ml_data[0], "signal"], [ml_data[1], "background"]]:
            logger.info(f"Book the training and testing events for {df_key}")
            # Reconstruct H_mass and angular variables for training.
            df_charge = df.Filter("(Electron_charge[0] + Electron_charge[1]) == 0 "\
                                  "&& (Muon_charge[0] + Muon_charge[1]) == 0",
                                  "Two opposite charged electron and muon pairs")
            df_reco = reco_define(df_charge)
            # Define other training variables
            df_all = df_reco.Define("Muon_pt_1", "Muon_pt[0]")\
                            .Define("Muon_pt_2", "Muon_pt[1]")\
                            .Define("Muon_mass_1", "Muon_mass[0]")\
                            .Define("Muon_mass_2", "Muon_mass[1]")\
                            .Define("Electron_mass_1", "Electron_mass[0]")\
                            .Define("Electron_mass_2", "Electron_mass[1]")\
                            .Define("Electron_pt_1", "Electron_pt[0]")\
                            .Define("Electron_pt_2", "Electron_pt[1]")\
                            .Define("Muon_eta_1", "Muon_eta[0]")\
                            .Define("Muon_eta_2", "Muon_eta[1]")\
                            .Define("Electron_eta_1", "Electron_eta[0]")\
                            .Define("Electron_eta_2", "Electron_eta[1]")\
                            .Define("Muon_phi_1", "Muon_phi[0]")\
                            .Define("Muon_phi_2", "Muon_phi[1]")\
                            .Define("Electron_phi_1", "Electron_phi[0]")\
                            .Define("Electron_phi_2", "Electron_phi[1]")

            # Save ml training datasets to file
            filename = f"train_{df_key}.root"
            df_train = df_all.Snapshot("Events", filename, columns, snapshotOptions)
            list_df_train.append(df_train)

    report_sig = list_df_train[0].Filter("H_mass > 110 && H_mass <140", "H_mass in [110, 140]")\
                                 .Report()
    report_bkg = list_df_train[1].Filter("H_mass > 110 && H_mass <140", "H_mass in [110, 140]")\
                                 .Report()
    list_rep_train = [report_sig, report_bkg]
    return list_df_train, list_rep_train

def ml_preprocessing(sig, bkg, branches):
    """Prepare the numpy arrays to be read by most ML tools.

    Reduce the amount of data to a fixed fraction of orginal dataset.
    (The fraction can be changed editing entries_frac variable)
    Perform data labeling, standardization and shuffling.

    The arrays are saved to local .npy files.

    Args:
        sig (numpy.ndarray): RDF converted to numpy array of "sig".
        bkg (numpy.ndarray): RDF converted to numpy array of "bkg".
        branches (list): list of branches name.

    Returns:
        tuple: standardized and shuffled data, data labels.
    """
    # Convert inputs to format readable by machine learning tools
    # T is the transposed vector
    x_sig = np.vstack([sig[str(var)] for var in branches]).T
    x_bkg = np.vstack([bkg[str(var)] for var in branches]).T
    # Reduce the number of entries to save on disk keeping the original proportion
    entries_frac = 0.5
    nEntries_s = round(entries_frac * x_sig.shape[0])
    nEntries_b = round(entries_frac * x_bkg.shape[0])
    x_sig_red = x_sig[:nEntries_s]
    x_bkg_red = x_bkg[:nEntries_b]
    x = np.vstack([x_sig_red, x_bkg_red])
    # Create labels
    num_sig = x_sig_red.shape[0]
    num_bkg = x_bkg_red.shape[0]
    y = np.hstack([np.ones(num_sig), np.zeros(num_bkg)])
    # Scaling data
    sc = StandardScaler()
    x_sc = sc.fit_transform(x)

    x_sh, y_sh = shuffle(x_sc, y)
    np.save("x_sh_unbalanced.npy", x_sh)
    np.save("y_sh_unbalanced.npy", y_sh)

    return (x_sh, y_sh)

def keras_model(in_dim):
    """Initialize, define and customize Keras model.

    Args:
        in_dim (int): input dimension to model.

    Returns:
        tensorflow.keras.models.Sequential: defined Keras model.
    """
    # Definition of a costumized keras model
    model = Sequential()
    model.add(Dense(24, input_dim=in_dim,
                    activation="relu"))
    model.add(Dense(48, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="sigmoid"))
    model.compile(loss="binary_crossentropy",
                  optimizer=SGD(lr=0.01),
                  metrics=["accuracy"])
    return model

def model_train(name, IN_DIM, X_train, X_val, y_train, y_val):
    """Try to load the already trained model of the chosen name, if not
    present initialize and train the default model, save and return it.

    WARNING: joblib is basically equivalet to Python pickle with an optimized handling
    for large numpy arrays, therefore it works only with matching python and sklearn
    versions. If different, the code is set to train again the models.
    Edit of the models characteristics is possible but only the N_EPOCHS is
    recorded in the saved file of the model

    Args:
        name (str): model name.
        IN_DIM (int): input dimension to model.
        X_train (numpy.ndarray): train dataset.
        X_val (numpy.ndarray): validation dataset.
        y_train (numpy.ndarray): training data labels.
        y_val (numpy.ndarray): validation data labels.

    Returns:
        dict: dicionary containing model (and training history if keras).
    """
    N_EPOCHS = 200
    # Series of specific instruction to train a keras model
    if name == "k_model":
        # Check saved model
        try:
            # Load saved model
            k_model = load_model(f"k_model_ep{N_EPOCHS}_nf{IN_DIM}")
            k_history = np.load(f"k_history_ep{N_EPOCHS}_nf{IN_DIM}.npy",
                                allow_pickle="TRUE").item()
            return {"model":k_model, "history":k_history}
        except (IOError, OSError, KeyError) as e:
            logger.warning(f"Something went wrong loading keras model or history: {e}")
        # Train and save model to folder
        # Keras needs categorical labels (means one column per class "0" and "1")
        logger.info("Keras model or history impossible to load. start training.")
        Y_train_k = to_categorical(y_train)
        Y_val_k = to_categorical(y_val)
        k_model = keras_model(IN_DIM)
        start1 = time.time()
        k_history = k_model.fit(X_train, Y_train_k,
                                validation_data=(X_val, Y_val_k),
                                epochs=N_EPOCHS, batch_size=64)
        k_model.save(f"k_model_ep{N_EPOCHS}_nf{IN_DIM}")
        np.save(f"k_history_ep{N_EPOCHS}_nf{IN_DIM}.npy", k_history.history)
        k_history = k_history.history
        stop1 = time.time()
        logger.info(f"Elapsed time training keras model:{stop1 - start1:.2f}s")
        return {"model":k_model, "history":k_history}
    # The model is one of the sklearn package
    elif os.path.isfile(f"{name}_nf{IN_DIM}.pkl"):
        try:
            # Load the pickle file
            clf_load = joblib.load(f'{name}_nf{IN_DIM}.pkl')
            return {"model": clf_load}
        except (IOError, OSError, KeyError) as e:
            logger.warning(f"Something went wrong loading {name}_nf{IN_DIM}.pkl: {e}")
    else:
        logger.info(f"Local trained {name} impossible to load. Start training.")
    # If there are problems loading the local trained model or the model is yet
    # to be trained, it will be, and it will be saved locally.
    start3 = time.time()
    if name == "rf_model":
        # Supervised transformation based on random forests
        rf = RandomForestClassifier(max_depth=7, n_estimators=IN_DIM)
        model = rf.fit(X_train, y_train)
    elif name == "mlp_model":
        # Multi-layer Perceptron model
        mlp = MLPClassifier(hidden_layer_sizes=(IN_DIM+5,), activation='tanh',
                            batch_size=64, max_iter=200)
        model = mlp.fit(X_train, y_train)
    elif name == "ada_model":
        # Ada boost decision tree classifier
        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                                 n_estimators=200)
        model = bdt.fit(X_train, y_train)
    elif name == "svc_model":
        # Linear Support Vector Classification
        svc = SVC(C=10, kernel='rbf', probability=True, max_iter=200)
        model = svc.fit(X_train, y_train)
    stop3 = time.time()
    logger.info(f"Elapsed time training {name}: {stop3 - start3:.2f}s")
    # Save model on disk
    joblib.dump(model, f"{name}_nf{IN_DIM}.pkl")
    return {"model":model}

def model_eval(name, model, X_test, y_test):
    """Evaluate the performaces of the classifier.

    ROC parameters are calculated: false positive rate (fpr)
                                   true positive rate (tpr)
                                   thresholds
                                   area under curve (AUC)

    Args:
        name (str): model name.
        model ([various types]): trained model, can be of different
                                 type depending on chosen model.
        X_test (numpy.ndarray): test dataset.
        y_test (numpy.ndarray): test data labels.

    Returns:
        dict: dictionary containing the model ROC parameters
    """
    if name == "k_model":
        # Return the probability to belong to a class (2D array)
        # Takes only the column of the "1" class (containing the probability)
        y_pred = model.predict(X_test)[:, 1]
    elif name in ["rf_model", "mlp_model", "ada_model", "svc_model"]:
        y_pred = model.predict_proba(X_test)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_model = auc(fpr, tpr)
    return {"fpr":fpr, "tpr":tpr, "auc":auc_model, "ths":thresholds}

def ml_plot(models, IN_DIM, rep_train, rep_higgs):
    """Plot all the models and two points from two different filters
    of the std analysis are added.

    Args:
        models (dict): dictionary containing all models evalutation.
        IN_DIM (int): input dimension to model.
        rep_train (Report): Report of filter on Higgs mass only.
        rep_higgs (Report): Report of all filters and Higgs mass.
    """

    # Here there is only the prefilter to ensure that the first two elements,
    # which are the ones that are used for all the reconstructed variables,
    # are of opposite charge, and the narrow filter on the Higgs mass
    for cut_sig, cut_bkg in zip(rep_train[0], rep_train[1]):
        #This works only because there is only registered 1 filter in the report
        h_sig_eff = cut_sig.GetEff()*0.01
        h_bkg_rej = 1 - cut_bkg.GetEff()*0.01

    cut_eff_sig = []
    cut_eff_bkg = []
    for cut_sig, cut_bkg in zip(rep_higgs[0], rep_higgs[1],):
        cut_eff_sig.append(cut_sig.GetEff()*0.01)
        cut_eff_bkg.append(cut_bkg.GetEff()*0.01)
    all_sig_eff = np.prod(cut_eff_sig)
    all_bkg_rej = 1 - np.prod(cut_eff_bkg)

    # Plot the ROC curves
    plt.figure()
    plt.plot([0, 1], [1, 0], "k--")
    plt.plot(h_sig_eff, h_bkg_rej, "kx",
             label=f"std filter only on H mass:({h_sig_eff:.3f}, {h_bkg_rej:.3f})")
    plt.plot(all_sig_eff, all_bkg_rej, "r*",
             label=f"std all filters + H mass:({all_sig_eff:.3f}, {all_bkg_rej:.3f})")
    for name in models.keys():
        model = models[name]
        plt.plot(model["tpr"], 1 - model["fpr"], label=f'{name} (area = {model["auc"]:.3f})')
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background rejection")
    plt.title("ROC curve")
    plt.legend(loc="best")
    plt.savefig(f"Roc_curve_nf{IN_DIM}")

    plt.figure()
    plt.xlim(0.5, 1)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [1, 0], 'k--')
    plt.plot(h_sig_eff, h_bkg_rej, "kx",
             label=f"std filter only on H mass:({h_sig_eff:.3f}, {h_bkg_rej:.3f})")
    plt.plot(all_sig_eff, all_bkg_rej, "r*",
             label=f"std all filters + H mass:({all_sig_eff:.3f}, {all_bkg_rej:.3f})")
    for name in models.keys():
        model = models[name]
        plt.plot(model["tpr"], 1 - model["fpr"], label=f'{name} (area = {model["auc"]:.3f})')
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background rejection")
    plt.title('ROC curve (zoomed in at top right)')
    plt.legend(loc='best')
    plt.savefig(f"Roc_curve_zoomed_nf{IN_DIM}")

    # Plot loss curve for keras model
    plt.figure()
    plt.plot(models["k_model"]["history"]["loss"])
    plt.plot(models["k_model"]["history"]["val_loss"])
    plt.title("Keras Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.savefig(f"loss_curve_nf{IN_DIM}")

def ml_retrieve(df_train, rep_train, rep_higgs):
    """Retrieve the dataset to train a ML or a DNN model,
    perform the training and testing and plot them in a ROC curve for
    comparison.
    The code is set to search the trained models locally and load them,
    to perform the training remove the relative file/dir from the
    working drectory.
    The number of given events, features and ephocs (for the keras model)
    can be set. (Instant unless local trained model is found)
    Warning: In ml_request: ml_var are the features available, they can be
    selected but must always be the same size of IN_DIM.
    Hence if any change is applied over the dataset, make sure to either
    remove or rename the local file when necessary

    Args:
        df_train (list): list of training RDF.
        rep_train (Report): Report of filter on Higgs mass only.
        rep_higgs (Report): Report of all filters and Higgs mass.
    """
    N_EVENTS = 100000 # Max available 134618 (update if edit entries_frac in ml_prepr)
    # N_COLUMNS = 22 # Max available 22
    IN_DIM = 1
    try:
        # From now on all variables >1D have capital letter
        X_sh = np.load("x_sh_unbalanced.npy")
        y_sh = np.load("y_sh_unbalanced.npy")
    except (IOError, OSError, KeyError) as e:
        logger.warning(f"Something went wrong loading .npy file: {e}")
        logger.info("Read data from ROOT files, convert to .npy and prepare for ML."
                    "This trigger the event loop if not done before")
        # Read data from ROOT files, trigger event loop if not done before
        list_branches = df_train[0].GetColumnNames()
        data_sig = df_train[0].AsNumpy()
        data_bkg = df_train[1].AsNumpy()
        # The arrays are now normalized and shuffled, and their file is saved
        X_sh, y_sh = ml_preprocessing(data_sig, data_bkg, list_branches)
    # To have an idea of the distribution in variance of the data, the PCA is given.
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sh)
    # Quick plot
    plt.figure()
    target_names = ["signal", "background"]
    colors = ["navy", "darkorange"]
    lw = 2
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X_pca[y_sh == i, 0], X_pca[y_sh == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of dataset')
    plt.savefig("PCA")

    #If the entries_frac in ml_preprocessing has been changed enable this to see
    # the new availabe dimension of the dataset
    print(X_sh.shape, y_sh.shape)

    # Selection of the events and variables, the sliced columns Must be of
    # IN_DIM size
    X_sh = X_sh[:N_EVENTS, 21:]
    y_sh = y_sh[:N_EVENTS]

    # Prepare train and test set, all models will need this, and validation set for keras
    X_batch, X_val, y_batch, y_val = train_test_split(X_sh, y_sh, test_size=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X_batch, y_batch, test_size=0.3)

    #Dictionary of the used models
    models_dict = {"k_model":{}, "rf_model":{}, "mlp_model":{}, "ada_model":{}, "svc_model":{}}
    # Here the ML models and DNNs are trained, fitted and evaluated, then are ready to be plotted
    for name in models_dict.keys():
        models_dict[name] = model_train(name, IN_DIM, X_train, X_val, y_train, y_val)
        model = models_dict[name]["model"]
        eval_dict = model_eval(name, model, X_test, y_test)
        models_dict[name].update(eval_dict)

    # Plot all the models
    ml_plot(models_dict, IN_DIM, rep_train, rep_higgs)

if __name__ == "__main__":

    # Monitor of the code run time
    start = time.time()

    # Set the standard mode of analysis
    perform_std = True

    # Possible options for different analysis
    parser = argparse.ArgumentParser(description=_description)
    parser.add_argument("-l", "--local",
                        help="perform analysis on local dataframe,"
                        " download of the request data if not present",
                        action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p", "--preliminar",
                       help="perform only preliminar analysis",
                       action="store_true")
    group.add_argument("-b", "--both",
                       help="perform also preliminar analysis",
                       action="store_true")
    args = parser.parse_args()

    # Check the chosen argparse options
    if args.preliminar or args.both:
        # In both cases we need the preliminary requests
        key_df_prel, branches_prel, histos_prel = preliminar_request(args.local)

        if args.preliminar:
            # Standard analysis is excluded
            perform_std = False
            logger.info("You have disabled standard analysis")
            # There are no other requests, event loop can be triggered
            preliminar_retrieve(key_df_prel, branches_prel, histos_prel)
        else:
            logger.info("It will pass to the requests for standard analysis")
            pass
    else:
        pass

    if perform_std:
        # Standard analysis
        filter_dicts, h_higgs, rep_higgs, df_ml, ang_dict = standard_request(args.local)
        ml_req_df, ml_req_rep = ml_request(df_ml)
        if args.both:
            # Preliminary requests done. Let's go to the retrieving part
            preliminar_retrieve(key_df_prel, branches_prel, histos_prel)
            # standard_retrieve(filter_dicts, h_higgs, rep_higgs, ang_dict)
            ml_retrieve(ml_req_df, ml_req_rep, rep_higgs)
        else:
            # standard_retrieve(filter_dicts, h_higgs, rep_higgs, ang_dict)
            ml_retrieve(ml_req_df, ml_req_rep, rep_higgs)
            logger.info("You have chosen to perform only standard analysis")
    else:
        pass
    stop = time.time()
    logger.info(f"elapsed code time: {stop - start:.2f}s\n")

import os
import argparse
import sys
import ROOT
import logging
import time
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

#Necessary option in order for argparse to function, if not present ROOT will 
#prevail over argparse when option are passed from command line
# ROOT.PyConfig.IgnoreCommandLineOptions = True

logging.basicConfig(filename = "test.log", level = logging.DEBUG, 
                    format = "%(asctime)s %(message)s")
_description = "The program will perform standard analysis if no option is given."

#Include cpp library for RDataFrame modules
ROOT.gInterpreter.ProcessLine('#include "library.h"')

# Enable multi-threading
# ROOT.ROOT.EnableImplicitMT(4)

#Imput links to signal, background and data samples
#Signal of Higgs -> 4 leptons
link_sig = "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/SMHiggsToZZTo4L.root"
#Background of ZZ -> 4 leptons
link_bkg = "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/ZZTo2e2mu.root"
#CMS data tacken in 2012 (11.6 fb^(-1) integrated luminosity)
link_data = ROOT.std.vector("string")(2)
link_data[0] = "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root"
link_data[1] = "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012C_DoubleMuParked.root"
#Dict of link 
dataset_link = {"signal": link_sig, "background": link_bkg, "data": link_data}

def style(h, mode, Rcolor):
    """Set the basic style of a given histogram depending on which mode is required
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
    """Make a snapshot of df and selected branches to local filename.
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
    """In this code we will consider only events with 2 or more
    electrons and muons
    """
    df_net = ROOT.RDataFrame("Events", dataset_link.get(df_key))
    df_2e2m = df_net.Filter("nElectron>=2 && nMuon>=2",
                            "Events with at least two Electrons and Muons")
    return df_2e2m

def access_df(df_key, local_flag, mode):
    """Access given dataset from appropriate location. Download to local
    file if needed.
    """
    if local_flag:
        filename = f"{df_key}_{mode}.root"
        if os.path.isfile(filename):
            df_local = ROOT.RDataFrame("Events", filename)
            # local_branches = df_local.GetColumnNames()
            df_out = df_local
        else:
            df_2e2m = df_prefilter(df_key)
            df_out = snapshot(df_2e2m, filename)
    else:
        df_out = df_prefilter(df_key)
    return df_out

def preliminar_request(flag):
    """First look at the chosen input file:
    this function is meant to explore only first hand characteristics of events
    and by default it will show three variables that are stored per component
    """
    key_df = input("Insert the data frame you want to look at first "
                         f"choosing from this list:\n{dataset_link.keys()}\n")

    try:
        #In this way nothing is saved locally but the list of chosen columns can be shown
        df = access_df(key_df, False, "p")
        list_branches = df.GetColumnNames()

        # decide which variables to look first
        dictOfBranches = {i:list_branches[i] for i in range (0, len(list_branches))}
        list_In = input("Insert the variable numbers to look at, separated by a space"
                        f"press return when you are done:\n{dictOfBranches}\n")
        
        #control input and retrieve the required branches
        list_str = list_In.split(" ")
        b_In = []
        for i in list_str:
            if i.isdigit() and int(i) < 32:
                current_branch = dictOfBranches[int(i)]
                b_In.append(current_branch)
            else:
                logging.warning(f"Error! {i} is an invalid key!")

        logging.info(f"These are the branches you chose: {b_In}")

        #Require the chosen df and request the histos
        b_In.extend(["PV_x", "PV_y", "PV_z", "Muon_dxy", "Muon_dz", 
                    "Electron_dxy", "Electron_dz"])
        unique_b_In = list(set(b_In))  
        df_In = access_df(key_df, flag, "p")
        h_In = []

        for branch in unique_b_In:
            current_histo = df_In.Histo1D(branch)
            h_In.append(current_histo)
        #3D reconstruction of some fundamental variables
        pv_3d = df_In.Define("PV_3d","sqrt(PV_x*PV_x + PV_y*PV_y + PV_z*PV_z)")
        mu_3d = df_In.Define("Muon_3d","sqrt(Muon_dxy*Muon_dxy + Muon_dz*Muon_dz)")
        el_3d = df_In.Define("El_3d","sqrt(Electron_dxy*Electron_dxy + Electron_dz*Electron_dz)")

        #Retrieve the relative histograms
        h_pv_3d = pv_3d.Histo1D("PV_3d")
        h_mu_3d = mu_3d.Histo1D("Muon_3d")
        h_el_3d = el_3d.Histo1D("El_3d")

        #Update of branches and histogram lists
        unique_b_In.extend(["PV_3d", "Muon_3d", "El_3d"])
        h_In.extend([h_pv_3d, h_mu_3d, h_el_3d])
    except KeyError as e:
        print(f"Cannot read the given key!\n{e}")
        sys.exit(1)
    return key_df, unique_b_In, h_In

def preliminar_retrieve(df_prel, b_prel, h_prel):
    """Retrieve and plot histos previously requested
    """
    #Trigger event loop and plot
    for branch, hist in zip(b_prel, h_prel):
        h = hist.GetValue()
        preliminar_plot(df_prel, branch, h)

def standard_define(df):
    """It defines and adds to the current df
    the variables that will be used for filter events in the standard analysis.
    All the variables that are more significant for the analysis, such as Z, H
    and his angular variables will be regrouped in a specific to make clearer the main
    points of the workflow and to be more easily accessible from the ml routines.
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

def reco_define(df_2e2mu):
    """It defines all the main variables, such as Z, H and its angular variables  
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
                        .Define("CosTheta_star",costheta_star_reco)\
                        .Define("Phi", phi_reco)\
                        .Define("Phi1", phi1_reco)\
                        .Define("CosTheta1", costheta1_reco)\
                        .Define("CosTheta2", costheta2_reco)
    return df_z_mass

def standard_request(flag):
    """All the necessary requests for the standard analysis will be prepared:
    """ 
    df_s = access_df("signal", flag, "std")
    df_b = access_df("background", flag, "std")
    df_d = access_df("data", flag, "std")

    df_s1 = standard_define(df_s)
    df_b1 = standard_define(df_b)
    df_d1 = standard_define(df_d)

    start = time.time()
    #Request filtered and unfiltered data
    dict_filter = {}
    dict_filter["sig"] = show_cut(df_s1)
    dict_filter["bkg"] = show_cut(df_b1)

    # Request ml dataframe
    df_ml = [df_s, df_b]

    #Weights
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

    #Request all the necessary to reconstruct the Higgs mass
    h_sig_higgs, report_sig, h_sig_ang = reco_higgs(df_s1, weight_sig, weight_ang_sig)
    h_bkg_higgs, report_bkg, h_bkg_ang = reco_higgs(df_b1, weight_bkg, weight_ang_bkg)
    h_data_higgs,report_data, h_data_ang = reco_higgs(df_d1, weight_data, weight_ang_data)
    stop = time.time()
    dict_angular = OrderedDict({"costheta_star":[], "costheta1":[], "costheta2":[],
                                "phi":[], "phi1":[]})
    for idx, h_list in enumerate(dict_angular.values()):
        h_list.extend([h_sig_ang[idx], h_bkg_ang[idx], h_data_ang[idx]])

    list_higgs = [h_sig_higgs, h_bkg_higgs, h_data_higgs]
    list_rep_higgs = [report_sig, report_bkg, report_data]
    print("request time:", stop-start)

    return dict_filter, list_higgs, list_rep_higgs, df_ml, dict_angular

def show_cut(df_2e2m):
    """Comparison between unfiltered and filtered data for each filter, 
    considering the main cuts used in the analysis published on CERN Open Data.
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
    df_z_mass = df_2e2m.Define("Z_mass", "z_mass(Electron_pt, Electron_eta, Electron_phi, Electron_mass,"\
                               "Muon_pt, Muon_eta, Muon_phi, Muon_mass)")
    h_unfil_z = df_z_mass.Histo1D(("h_Z_mass", "", 56, -5, 140), "Z_mass")
    # 6th filter: z masses
    df_z = df_z_mass.Filter("Z_mass[0] > 40 && Z_mass[0] < 120 && Z_mass[1] > 12 && Z_mass[1] < 120",
                            "First candidate in [40, 120] and Second candidate in [12, 120]")
    h_fil_z = df_z.Histo1D(("h_Z_mass", "", 56, -5, 140), "Z_mass")
    filter_dict["Z_mass"][0].append(h_unfil_z)
    filter_dict["Z_mass"][1].append(h_fil_z)
    filter_dict["Z_mass"][2].append(df_z.Report())
    return filter_dict

def good_events(df_2e2m):
    """Selection of 2electrons and 2 muons that pass the cuts used in the 2012 CERN article
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

def reco_higgs(df, weight, weight_ang):
    """Recontruction of the Higgs mass
    """
    #Selection of only the potential good events 
    df_base = good_events(df)
    
    #Compute z and H masses from it
    df_z_mass = reco_define(df_base)
    #Filter on z masses
    df_z1 = df_z_mass.Filter("Z_mass[0] > 40 && Z_mass[0] < 120", "First candidate in [40, 120]") 
    df_z2 = df_z1.Filter("Z_mass[1] > 12 && Z_mass[1] < 120", "Second candidate in [12, 120]")

    #Define weights
    df_reco = df_z2.Define("weight", f"{weight}").Define("weight_ang", f"{weight_ang}")
    h_reco_h = df_reco.Histo1D(("h_sig_2el2mu", "", 36, 70, 180), "H_mass", "weight")
    #Filter on Higgs mass
    df_reco_h = df_reco
    # df_reco_h = df_reco.Filter("H_mass > 110 && H_mass <140", "H_mass in [110, 140]")

    #Reconstructed angular variables
    h_costheta_star = df_reco_h.Histo1D(("h_costheta_star", "CosTheta_star", 56, -1, 1), "CosTheta_star", "weight_ang")
    h_costheta1 = df_reco_h.Histo1D(("h_costheta1", "CosTheta1", 56, -1, 1), "CosTheta1", "weight_ang")
    h_costheta2 = df_reco_h.Histo1D(("h_costheta2", "CosTheta2", 56, -1, 1), "CosTheta2", "weight_ang")
    h_phi = df_reco_h.Histo1D(("h_phi", "Phi", 56, -3.14, 3.14), "Phi", "weight_ang")
    h_phi1 = df_reco_h.Histo1D(("h_phi1", "Phi1", 56, -3.14, 3.14), "Phi1", "weight_ang")
    list_h_ang = [h_costheta_star, h_costheta1, h_costheta2, h_phi, h_phi1]
    report_higgs = df_reco_h.Report() 
    
    return h_reco_h, report_higgs, list_h_ang

def standard_retrieve (filters, h_higgs, rep_higgs, angular):
    """If the code is in only standard mode, here the event loop will be triggered
    """
    filters_data = {"sig":{"Isolation":[[], [], []], "Eta":[[], [], []], "Dr":[[], [], []], "Pt":[[], [], []],
                     "Electron_track":[[], [], []], "Muon_track":[[], [], []], "Z_mass":[[], [], []]},
                    "bkg":{"Isolation":[[], [], []], "Eta":[[], [], []], "Dr":[[], [], []], "Pt":[[], [], []],
                     "Electron_track":[[], [], []], "Muon_track":[[], [], []], "Z_mass":[[], [], []]}}
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

    [(rep.Print(), print("")) for rep in rep_higgs]

def preliminar_plot(df, branch_histo, histo_data):
    """For now it"s just a simple plot of unprocessed data
    """
    #General Canvas Settings
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetTextFont(42)
    c_histo = ROOT.TCanvas(f"c_histo_{branch_histo}","",800,700)
    c_histo.cd()
    # ROOT.SetOwnership(c_histo, False)

    #Set and Draw histogram for data points
    x_max = histo_data.GetXaxis().GetXmax()
    x_min = histo_data.GetXaxis().GetXmin()
    logging.info(f"One day minumun and maximun of the {branch_histo}"
                 f" will be useful...but it is NOT This day!{x_max,x_min}")
    # histo_data.GetXaxis().SetRangeUser(x_min,x_max)
    
    histo_data.SetLineStyle(1)
    histo_data.SetLineWidth(1)
    histo_data.SetLineColor(ROOT.kBlack)
    histo_data.Draw()

    # Add Legend
    legend = ROOT.TLegend(0.7, 0.6, 0.85, 0.9)
    legend.SetFillColor(0)
    # legend.SetBorderSize(1)
    legend.SetLineWidth(0)
    legend.SetTextSize(0.04)
    legend.AddEntry(histo_data,df)
    legend.Draw()

    #Save plot
    filename = f"{branch_histo}_{df}.pdf"
    c_histo.SaveAs(filename)

def filtered_plot(h_sig, h_bkg, rep_s, rep_b, fil):
    """Plot of the filtered data versus the unprocessed ones"""
    
    h_uf_s, h_f_s = h_sig
    h_uf_b, h_f_b = h_bkg
    
    # Add canvas
    canvas = ROOT.TCanvas("canvas","",800,500)
    canvas.cd()

    # Add Legend
    legend = ROOT.TLegend(0.18 ,0.7 ,0.33 ,0.85)
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

    for i, (h1, h2, h3, h4, rep12, rep34) in enumerate(zip(h_uf_s, h_f_s, h_uf_b, h_f_b, rep_s, rep_b)):
        canvas.cd()
        h_ustyle_s = style(h1, "e", ROOT.kRed)
        h_fstyle_s = style(h2,"f", ROOT.kRed)
        h_ustyle_b = style(h3,"e", ROOT.kAzure)
        h_fstyle_b = style(h4,"f", ROOT.kAzure)

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

        latex.DrawText (0.68 ,0.67 ,f"Sig evt pass: {cut_eff_s[0]:.2f}%")
        latex.DrawText (0.68 ,0.73 ,f"Bkg evt pass: {cut_eff_b[0]:.2f}%")

        sig_pass = h_fstyle_s.GetEntries()/h_ustyle_s.GetEntries()*100
        bkg_pass = h_fstyle_b.GetEntries()/h_ustyle_b.GetEntries()*100
        latex.DrawText (0.68 ,0.79 ,f"Sig cnt pass: {sig_pass:.2f}%")
        latex.DrawText (0.68 ,0.85 ,f"Bkg cnt pass: {bkg_pass:.2f}%")

    h_ustyle_b.SetTitle(f"{fil}")
    legend.AddEntry(h_uf_s[0],"Signal Unfilterd")
    legend.AddEntry(h_f_s[0],"Signal Filtered")
    legend.AddEntry(h_uf_b[0],"Bkg Unfilterd")
    legend.AddEntry(h_f_b[0],"Bkg Filtered")
    legend.Draw()

    #Save plot
    canvas.SaveAs(f"{fil}.pdf")

def higgs_plot(list_histo_higgs, filename):
    """Plot reconstructed Higgs mass for signal, background and data
    """
    titles_units_dict = {"costheta_star":["cos#Theta*", ""], "costheta1":["cos#Theta_{1}", ""],
                         "costheta2":["cos#Theta_{2}", ""],
                         "phi":["#Phi", "[rad]"], "phi1":["#Phi_{1}", "[rad]"],
                         "h_mass": ["M_{2e2#mu}", "[GeV]"]}
    
    # Add canvas
    canvas_s = ROOT.TCanvas("canvas_s","",800,700)
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
    legend = ROOT.TLegend(0.7 ,0.7 ,0.85 ,0.85)
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

def ml_request(ml_data):
    """Prepare dataset to be readable by the machine learning  method
    """
    list_df_train = []
    if os.path.isfile("train_signal.root") and os.path.isfile("train_background.root"):
        list_df_train.extend([ROOT.RDataFrame("Events", "train_signal.root"),
                              ROOT.RDataFrame("Events", "train_background.root")])
    else:
        # These are all the variables that will be put in the training df
        ml_var = ["Muon_pt_1", "Muon_pt_2", "Electron_pt_1", "Electron_pt_2",
                  "Muon_mass_1", "Muon_mass_2","Electron_mass_1", "Electron_mass_2", 
                  "Muon_eta_1", "Muon_eta_2", "Electron_eta_1", "Electron_eta_2",
                  "Muon_phi_1", "Muon_phi_2", "Electron_phi_1", "Electron_phi_2",
                  "H_mass", "CosTheta_star", "CosTheta1", "CosTheta2", "Phi", "Phi1"]
        columns = ROOT.std.vector("string")()
        for column in ml_var:
            columns.push_back(column)

        snapshotOptions = ROOT.RDF.RSnapshotOptions()
        snapshotOptions.fLazy = True

        for df, df_key in [[ml_data[0], "signal"], [ml_data[1], "background"]]:
            logging.info(f"Book the training and testing events for {df_key}")
            #Reconstruct H_mass and angular variables for training.
            df_charge = df.Filter("(Electron_charge[0] + Electron_charge[1]) == 0 && (Muon_charge[0] + Muon_charge[1]) == 0",
                                  "Two opposite charged electron and muon pairs")
            df_reco = reco_define(df_charge)
            # Define other training variables
            df_all = df_reco.Define("Muon_pt_1", "Muon_pt[0]").Define("Muon_pt_2", "Muon_pt[1]")\
                            .Define("Muon_mass_1", "Muon_mass[0]").Define("Muon_mass_2", "Muon_mass[1]")\
                            .Define("Electron_mass_1", "Electron_mass[0]").Define("Electron_mass_2", "Electron_mass[1]")\
                            .Define("Electron_pt_1", "Electron_pt[0]").Define("Electron_pt_2", "Electron_pt[1]")\
                            .Define("Muon_eta_1","Muon_eta[0]").Define("Muon_eta_2","Muon_eta[1]")\
                            .Define("Electron_eta_1","Electron_eta[0]").Define("Electron_eta_2", "Electron_eta[1]")\
                            .Define("Muon_phi_1", "Muon_phi[0]").Define("Muon_phi_2","Muon_phi[1]")\
                            .Define("Electron_phi_1","Electron_phi[0]").Define("Electron_phi_2", "Electron_phi[1]")

            # Save ml training datasets to file
            filename = f"train_{df_key}.root"
            df_train = df_all.Snapshot("Events", filename, columns, snapshotOptions)
            list_df_train.append(df_train)            

    report_sig = list_df_train[0].Filter("H_mass > 110 && H_mass <140", "H_mass in [110, 140]").Report()
    report_bkg = list_df_train[1].Filter("H_mass > 110 && H_mass <140", "H_mass in [110, 140]").Report()
    list_df_train.extend([report_sig, report_bkg])
    return list_df_train

def ml_preprocessing(sig, bkg, branches):
    """It prepares the arrays to be read by most ML tools.
    the arrays are then saved in a .npy file
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
    # Definition of a costumed keras model
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
    """Load the already trained model, if not present initialize and train 
    the default model and return self
    """
    N_EPOCHS = 100
    # Series of specific instruction to train a keras model 
    if name == "k_model":
        # Check saved model
        if os.path.isdir(f"k_model_ep{N_EPOCHS}_nf{IN_DIM}") and\
           os.path.isfile(f"k_history_ep{N_EPOCHS}_nf{IN_DIM}.npy"):
           # Load saved model
           k_model = load_model(f"k_model_ep{N_EPOCHS}_nf{IN_DIM}")
           k_history = np.load(f"k_history_ep{N_EPOCHS}_nf{IN_DIM}.npy",
                               allow_pickle="TRUE").item()
        else:
            # Train and save model to folder
            # Keras needs categorical labels (means one column per class "0" and "1")
            Y_train_k = to_categorical(y_train)
            Y_val_k = to_categorical(y_val)
            start = time.time()
            k_model = keras_model(IN_DIM)
            k_history = k_model.fit(X_train, Y_train_k,
                                        validation_data = (X_val,Y_val_k), 
                                        epochs=N_EPOCHS, batch_size=64)
            k_model.save(f"k_model_ep{N_EPOCHS}_nf{IN_DIM}")
            np.save(f"k_history_ep{N_EPOCHS}_nf{IN_DIM}.npy", k_history.history)
            k_history = k_history.history
            stop = time.time()
            logging.info(f"Elapsed time training the net:{stop - start}")
        return {"model":k_model, "history":k_history}
    
    elif name == "rf_model": #this may be a list
        # Supervised transformation based on random forests
        rf = RandomForestClassifier(max_depth=7, n_estimators=IN_DIM)
        rf.fit(X_train, y_train)
        return {"model":rf}

def model_eval(name, model, X_test, y_test):
    """Evaluate the performaces of the classifier
    """
    if name == "k_model":
        # Return the probability to belong to a class (2D array) 
        # Takes only the column of the "1" class (containing the probability)
        y_pred = model.predict(X_test)[:, 1]
    elif name == "rf_model":
        y_pred = model.predict_proba(X_test)[:, 1]
    
    # fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
    # auc_keras = auc(fpr_keras, tpr_keras)
    # return {"fpr":fpr_keras, "tpr":tpr_keras, "auc":auc_keras}
    
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_model = auc(fpr, tpr)
    return {"fpr":fpr, "tpr":tpr, "auc":auc_model, "ths":thresholds}

def ml_plot(models):
    """All the models are plotted
    """
    # Plot the ROC curves
    plt.figure(1)
    plt.plot([0, 1], [1, 0], "k--")
    plt.plot(0.7368, 1 - 0.0906, "kx", label="std filter only higgs mass")
    plt.plot(0.25, 1 - 0.0103, "r*", label="std all filters + higgs mass")
    for name in models.keys():
        model = models[name]
        plt.plot(model["tpr"], 1 - model["fpr"], label=f'{name} (area = {model["auc"]:.3f})')
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background rejection")
    plt.title("ROC curve")
    plt.legend(loc="best")
    plt.savefig("Roc_curve")
    # plt.show()

    # Plot loss curve for keras model
    plt.figure(2)
    plt.plot(models["k_model"]["history"]["loss"]) 
    plt.plot(models["k_model"]["history"]["val_loss"]) 
    plt.title("Keras Model loss") 
    plt.ylabel("Loss") 
    plt.xlabel("Epoch") 
    plt.legend(["Train", "Test"], loc="upper left") 
    plt.savefig("loss_curve")
    # plt.show()

def ml_retrieve(list_df):
    """Retrieving the dataset to train a machine learning model
    """
    N_EVENTS = 80000
    N_COLUMNS = 22
    IN_DIM = 22
    if os.path.isfile("x_sh_unbalanced.npy") and os.path.isfile("y_sh_unbalanced.npy"):
        # From now on all variables >1D have capital letter
        X_sh = np.load("x_sh_unbalanced.npy")
        y_sh = np.load("y_sh_unbalanced.npy")
    else:
        # Read data from ROOT files, trigger event loop if not done before
        list_branches = list_df[0].GetColumnNames()
        data_sig = list_df[0].AsNumpy()
        data_bkg = list_df[1].AsNumpy()
        # The arrays are now normalized and shuffled, and their file is saved
        X_sh, y_sh = ml_preprocessing(data_sig, data_bkg, list_branches)
    
    # Selection of the events and variables, the sliced columns Must be of IN_DIM size
    X_sh = X_sh[:N_EVENTS , :N_COLUMNS]
    y_sh = y_sh[:N_EVENTS]
    # Prepare train and test set, all models will need this, and validation set for keras
    # X_train, X_test, y_train, y_test = train_test_split(X_sh, y_sh, test_size = 0.1)
    X_batch, X_val, y_batch, y_val = train_test_split(X_sh, y_sh, test_size = 0.1)
    X_train, X_test, y_train, y_test = train_test_split(X_batch, y_batch, test_size = 0.3)
       
    #Dictionary of the used models
    models_dict = {"k_model":{}, "rf_model":{}}
    # Here the ML models and DNNs are trained, fitted and evaluated, then are ready to be plotted
    for name in models_dict.keys():
        # models_dict[name] = model_train(name, IN_DIM, X_train, X_test, y_train, y_test)
        models_dict[name] = model_train(name, IN_DIM, X_train, X_val, y_train, y_val)
        model = models_dict[name]["model"]
        eval_dict = model_eval(name, model, X_test, y_test)
        models_dict[name].update(eval_dict)

    ml_plot(models_dict)
    list_df[2].Print()
    list_df[3].Print()
    return (list_df[2], list_df[3])

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
        df_prel, branches_prel, histo_prel = preliminar_request(args.local)
        
        if args.preliminar:
            # Standard analysis is excluded
            perform_std = False
            logging.info("You have disabled standard analysis")
            # There are no other requests, event loop can be triggered
            preliminar_retrieve(df_prel, branches_prel, histo_prel)
        else:
            logging.info("It will pass to the requests for standard analysis")
            pass
    else:
        pass
    
    if perform_std: 
        # Standard analysis
        filter_dicts, h_higgs, rep_higgs, df_ml, ang_dict = standard_request(args.local)
        ml_req_df = ml_request(df_ml) 
        if args.both:
            # Preliminary requests done. Let's go to the retrieving part
            preliminar_retrieve(df_prel, branches_prel, histo_prel)
            standard_retrieve(filter_dicts, h_higgs, rep_higgs, ang_dict)
            ml_retrieve(ml_req_df)
        else:
            inizio = time.time()
            standard_retrieve(filter_dicts, h_higgs, rep_higgs, ang_dict)
            fine = time.time()
            print("Retrieve time:", fine - inizio)
            ml_retrieve(ml_req_df)            
            logging.info("You have chosen to perform only standard analysis")
    else:
        pass
    stop = time.time()
    # logging.info(f"elapsed code time: {stop - start}\n")
    print(f"elapsed code time: {stop - start}\n")

import os
import argparse
import sys
import ROOT
import logging
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from collections import OrderedDict
# import keras
# from keras.models import Sequentials
# from keras.layers import Dense

#Necessary option in order for argparse to function, if not present ROOT will 
#prevail over argparse when option are passed from command line
ROOT.PyConfig.IgnoreCommandLineOptions = True

logging.basicConfig(filename = "test.log", level = logging.DEBUG, 
                    format = '%(asctime)s %(message)s')
_description = 'The program will perform standard analysis if no option is given.'

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

    #empty histogram
    if mode == 'e':
        h.SetLineColor(Rcolor)

    #full histogram
    elif mode == 'f':
        h.SetFillStyle(1001)
        h.SetFillColor(Rcolor)

    #mark histogram
    elif mode == 'm':
        h.SetMarkerStyle(20)
        h.SetMarkerSize(1.0)
        h.SetMarkerColor(Rcolor)
        h.SetLineColor(Rcolor)

    return h

def snapshot(df, filename, branches_list):
    """Make a snapshot of df and selected branches to local filename.
    """
    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    snapshotOptions.fLazy = True
    # columns = ROOT.vector('string')()
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
    # e = df_2e2m.Count()
    # print(f"number of entries that has passed the filter:{e.GetValue()}")
    return df_2e2m

def access_df(df_key, branches_list, local_flag, mode):
    """Access given dataset from appropriate location. Download to local
    file if needed.
    """
    if local_flag:
        filename = f"{df_key}_{mode}.root"
        if os.path.isfile(filename):
            df_local = ROOT.RDataFrame("Events", filename)
            local_branches = df_local.GetColumnNames()
            if set(branches_list) <= set(local_branches):
                #local df are already prefiltered since they have been downloaded
                #with the df_prefilter function
                df_out = df_local
            else:
                df_2e2m = df_prefilter(df_key)
                df_out = snapshot(df_2e2m, filename, branches_list)
        else:
            df_2e2m = df_prefilter(df_key)
            df_out = snapshot(df_2e2m, filename, branches_list)
    else:
        df_out = df_prefilter(df_key)
    return df_out

def preliminar_request(flag):
    """First look at the chosen input file:
    this function is meant to explore only first hand characteristics of events
    and by default it will show three variables that are stored per component
    """
    key_df = input('Insert the data frame you want to look at first '
                         f'choosing from this list:\n{dataset_link.keys()}\n')

    try:
        # df_In = dict_df.get(key_df)
        #In this way nothing is saved locally but the list of chosen columns can be shown
        df = access_df(key_df, [""], False, 'p')
        list_branches = df.GetColumnNames()

        # decide which variables to look first
        dictOfBranches = {i:list_branches[i] for i in range (0, len(list_branches))}
        list_In = input('Insert the variable numbers to look at, separated by a space'
                        f'press return when you are done:\n{dictOfBranches}\n')
        
        #control input and retrieve the required branches
        list_str = list_In.split(" ")
        b_In = []
        for i in list_str:
            if i.isdigit() and int(i) < 32:
                current_branch = dictOfBranches[int(i)]
                b_In.append(current_branch)
            else:
                logging.warning(f'Error! {i} is an invalid key!')

        logging.info(f'These are the branches you chose: {b_In}')

        #Require the chosen df and request the histos
        b_In.extend(["PV_x", "PV_y", "PV_z", "Muon_dxy", "Muon_dz", 
                    "Electron_dxy", "Electron_dz"])
        unique_b_In = list(set(b_In))  
        df_In = access_df(key_df, unique_b_In, flag, 'p')
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
        unique_b_In.extend(['PV_3d', 'Muon_3d', 'El_3d'])
        h_In.extend([h_pv_3d, h_mu_3d, h_el_3d])
    except KeyError as e:
        print(f'Cannot read the given key!\n{e}')
        sys.exit(1)
    return key_df, unique_b_In, h_In

def preliminar_retrieve(df_prel, b_prel, h_prel):
    """Retrieve and plot histos previously requested
    """
    #Trigger event loop and plot
    for branch, hist in zip(b_prel, h_prel):
        h = hist.GetValue()
        preliminar_plot(df_prel, branch, h)

def standard_request(flag):
    """All the necessary requests for the standard analysis will be prepared:
    """
    branches_std = ["Electron_eta", "Muon_eta", "Electron_phi", "Muon_phi", "Electron_pt",
                    "Muon_pt", "Electron_pfRelIso03_all", "Muon_pfRelIso04_all",
                    "Electron_dxy", "Electron_dz", "Electron_dxyErr", "Electron_dzErr",
                    "Muon_dxy", "Muon_dz", "Muon_dxyErr", "Muon_dzErr",
                    "Electron_charge", "Muon_charge", "Electron_mass", "Muon_mass"]
    
    df_s = access_df("signal", branches_std, flag, 'std')
    df_b = access_df("background", branches_std, flag, 'std')
    df_d = access_df("data", branches_std, flag, 'std')

    #Request filtered and unfiltered data
    dict_filter = {'sig':{}, 'bkg':{}}
    # list_, list_h_fil, list_rep_filters = show_cut(df_b)
    dict_filter['sig'] = show_cut(df_s)
    dict_filter['bkg'] = show_cut(df_b)
    df_train, ml_var = ml_request(df_s, df_b)

    #Weights
    luminosity = 11580.0  # Integrated luminosity of the data samples
    xsec_sig = 0.0065  # ZZ->2el2mu: Standard Model cross-section
    nevt_sig = 299973.0  # ZZ->2el2mu: Number of simulated events
    scale_ZZTo4l = 1.386  # ZZ->4l: Scale factor for ZZ to four leptons    
    xsec_bkg = 0.18  # ZZ->2el2mu: Standard Model cross-section
    nevt_bkg = 1497445.0  # ZZ->2el2mu: Number of simulated events 
    weight_sig = luminosity * xsec_sig / nevt_sig
    weight_bkg = luminosity * xsec_bkg * scale_ZZTo4l / nevt_bkg
    weight_data = 1.0   

    #Request all the necessary to reconstruct the Higgs mass
    h_signal, report_sig = reco_higgs(df_s, weight_sig)
    h_bkg, report_bkg = reco_higgs(df_b, weight_bkg)
    h_data,report_data = reco_higgs(df_d, weight_data)
    
    list_higgs = [h_signal, h_bkg, h_data]
    list_rep_higgs = [report_sig, report_bkg, report_data]

    return dict_filter, list_higgs, list_rep_higgs, df_train, ml_var

def show_cut(df_2e2m):
    """Comparison between unfiltered and filtered data considering the main cuts
    used in the analysis published on CERN Open Data
    """
    #Preparation of a list of filtered and unfiltered data to plot aferwards
    #And of a list of reports for each filter
    list_h_unfil = []
    list_h_fil = []
    list_report = []

    #1st filter:Eta cut
    h_unfil_eleta = df_2e2m.Histo1D(("h_Eleta", "Electron_eta", 56, -2.6, 2.6), "Electron_eta")
    h_unfil_mueta = df_2e2m.Histo1D(("h_Mueta", "Muon_eta", 56, -2.6, 2.6), "Muon_eta") 
    
    df_eleta = df_2e2m.Filter("All(abs(Electron_eta)<2.5)", "Eleta cut")
    df_mueta = df_2e2m.Filter("All(abs(Muon_eta)<2.4)", "Mueta cut")
    h_fil_eleta = df_eleta.Histo1D(("h_Eleta","", 56, -2.6, 2.6), "Electron_eta")
    h_fil_mueta = df_mueta.Histo1D(("h_Mueta","", 56, -2.6, 2.6), "Muon_eta")

    #2nd filter:Dr cut
    df_dr = df_2e2m.Define("Electron_dr",
                             "dr_def(Electron_eta, Electron_phi)").Define("Muon_dr",
                             "dr_def(Muon_eta, Muon_phi)")
    h_unfil_eldr = df_dr.Histo1D(("h_eldr","Electron_dr", 56, -0.5, 6), "Electron_dr")
    h_unfil_mudr = df_dr.Histo1D(("h_mudr","Muon_dr", 56, -0.5, 6), "Muon_dr")

    df_eldr = df_dr.Filter("Electron_dr>=0.02","Eldr cut")
    df_mudr = df_dr.Filter("Muon_dr>=0.02","Mudr cut")
    h_fil_eldr = df_eldr.Histo1D(("h_eldr","Electron_dr", 56, -0.5, 6), "Electron_dr")
    h_fil_mudr = df_mudr.Histo1D(("h_mudr","Muon_dr", 56, -0.5, 6), "Muon_dr")

    #3rd filter:Pt cut
    h_unfil_elpt = df_2e2m.Histo1D(("h_Elpt", "Electron_pt", 56, -0.5, 120), "Electron_pt")
    h_unfil_mupt = df_2e2m.Histo1D(("h_Mupt", "Muon_pt", 56, -0.5, 120), "Muon_pt")

    df_pt = df_2e2m.Filter("pt_cut(Muon_pt, Electron_pt)", "Pt cuts")
    h_fil_elpt = df_pt.Histo1D(("h_Elpt", "", 56, -0.5, 120), "Electron_pt")
    h_fil_mupt = df_pt.Histo1D(("h_Mupt", "", 56, -0.5, 120), "Muon_pt")

    #4th filter: Good isolation
    h_unfil_eliso3 = df_2e2m.Histo1D(("h_eliso3","Electron_Iso3", 400, -1020, 50), "Electron_pfRelIso03_all")
    h_unfil_muiso4 = df_2e2m.Histo1D(("h_muiso4","Muon_Iso4", 400, -1020, 50), "Muon_pfRelIso04_all")

    df_eliso = df_2e2m.Filter("All(abs(Electron_pfRelIso03_all)<0.40)", "ElIso03 cut")
    df_muiso = df_2e2m.Filter("All(abs(Muon_pfRelIso04_all)<0.40)", "MuIso04 cut")   
    h_fil_eliso3 = df_eliso.Histo1D(("h_eliso3","", 400, -1020, 50), "Electron_pfRelIso03_all")
    h_fil_muiso4 = df_muiso.Histo1D(("h_muiso4","", 400, -1020, 50), "Muon_pfRelIso04_all")

    #5th filter: Electron track
    el_sip3d = "sqrt(Electron_dxy*Electron_dxy + Electron_dz*Electron_dz)/sqrt(Electron_dxyErr*Electron_dxyErr+ Electron_dzErr*Electron_dzErr)"
    df_eltrack = df_2e2m.Define("Electron_sip3d", el_sip3d)
    h_unfil_elsip3d = df_eltrack.Histo1D(("h_elsip3d", "Electron_sip3d", 56, -0.5, 5),
                                          "Electron_sip3d")
    h_unfil_eldxy = df_eltrack.Histo1D(("h_eldxy", "Electron_dxy", 56, -0.03, 0.03),
                                        "Electron_dxy")
    h_unfil_eldz = df_eltrack.Histo1D(("h_eldz", "Electron_dz", 56, -0.03, 0.03),
                                       "Electron_dz")

    df_elsip3d = df_eltrack.Filter("All(Electron_sip3d<4)", "Elsip3d cut")
    df_eldxy = df_eltrack.Filter("All(abs(Electron_dxy)<0.5)", "Eldxy cut")
    df_eldz = df_eltrack.Filter(" All(abs(Electron_dz)<1.0)", "Eldz cut")
    h_fil_elsip3d = df_elsip3d.Histo1D(("h_elsip3d", "", 56, -0.5, 5), "Electron_sip3d")
    h_fil_eldxy = df_eldxy.Histo1D(("h_eldxy", "", 56, -0.03, 0.03), "Electron_dxy")
    h_fil_eldz = df_eldz.Histo1D(("h_eldz", "", 56, -0.03, 0.03), "Electron_dz")

    #6th filter: Muon track
    mu_sip3d = "sqrt(Muon_dxy*Muon_dxy + Muon_dz*Muon_dz)/sqrt(Muon_dxyErr*Muon_dxyErr+ Muon_dzErr*Muon_dzErr)"
    df_mutrack = df_2e2m.Define("Muon_sip3d", mu_sip3d)
    h_unfil_musip3d = df_mutrack.Histo1D(("h_musip3d", "Muon_sip3d", 56, -0.5, 5),
                                          "Muon_sip3d")
    h_unfil_mudxy = df_mutrack.Histo1D(("h_mudxy", "Muon_dxy", 56, -0.03, 0.03),
                                        "Muon_dxy")
    h_unfil_mudz = df_mutrack.Histo1D(("h_mudz", "Muon_dz", 56, -0.03, 0.03), "Muon_dz")

    df_musip3d = df_mutrack.Filter("All(Muon_sip3d<4)", "Musip3d cut")
    df_mudxy = df_mutrack.Filter("All(abs(Muon_dxy)<0.5)", "Mudxy cut")
    df_mudz = df_mutrack.Filter("All(abs(Muon_dz)<1.0)","Mudz cut")
    h_fil_musip3d = df_musip3d.Histo1D(("h_musip3d", "", 56, -0.5, 5), "Muon_sip3d")
    h_fil_mudxy = df_mudxy.Histo1D(("h_mudxy", "", 56, -0.03, 0.03), "Muon_dxy")
    h_fil_mudz = df_mudz.Histo1D(("h_mudz", "", 56, -0.03, 0.03), "Muon_dz")

    #Update the lists previously created and create a Report list to print afterwards
    list_h_unfil.extend([h_unfil_eleta, h_unfil_mueta, 
                        h_unfil_eldr, h_unfil_mudr,
                        h_unfil_elpt, h_unfil_mupt,
                        h_unfil_eliso3, h_unfil_muiso4,
                        h_unfil_elsip3d, h_unfil_eldxy, h_unfil_eldz,
                        h_unfil_musip3d, h_unfil_mudxy, h_unfil_mudz])
    list_h_fil.extend([h_fil_eleta, h_fil_mueta, 
                      h_fil_eldr, h_fil_mudr,
                      h_fil_elpt, h_fil_mupt,
                      h_fil_eliso3, h_fil_muiso4,
                      h_fil_elsip3d, h_fil_eldxy, h_fil_eldz,
                      h_fil_musip3d, h_fil_mudxy, h_fil_mudz])
    list_report.extend([df_eleta.Report(), df_mueta.Report(),
                        df_eldr.Report() , df_mudr.Report(), df_pt.Report(), 
                        df_eliso.Report(), df_muiso.Report(),
                        df_elsip3d.Report(), df_eldxy.Report(), df_eldz.Report(),
                        df_musip3d.Report(), df_mudxy.Report(), df_mudz.Report()])

    output = {'h_unfil':list_h_unfil, 'h_fil':list_h_fil, 'rep':list_report}   
    return output 

def good_events(df_2e2m):
    """Selection of 2electrons and 2 muons
    that pass the cuts used in the 2012 CERN article
    """
    #angular cuts
    df_eta = df_2e2m.Filter("All(abs(Electron_eta)<2.5) && All(abs(Muon_eta)<2.4)",
                           "Eta_cuts")
    #transvers momenta cuts
    df_pt = df_eta.Filter("pt_cut(Muon_pt, Electron_pt)", "Pt cuts")
    df_dr = df_pt.Filter("dr_cut(Muon_eta, Muon_phi, Electron_eta, Electron_phi)",
                           "Dr_cuts")
    #Request good isolation
    df_iso = df_dr.Filter("All(abs(Electron_pfRelIso03_all)<0.40) &&"
                              "All(abs(Muon_pfRelIso04_all)<0.40)",
                              "Require good isolation")
                           
    #Reconstruction and filter of Muon and Electron tracks
    el_ip3d = "sqrt(Electron_dxy*Electron_dxy + Electron_dz*Electron_dz)"
    df_el_ip3d = df_iso.Define("Electron_ip3d", el_ip3d)
    el_sip3d = "Electron_ip3d/sqrt(Electron_dxyErr*Electron_dxyErr+ Electron_dzErr*Electron_dzErr)"
    df_el_sip3d = df_el_ip3d.Define("Electron_sip3d", el_sip3d)
    df_el_track = df_el_sip3d.Filter("All(Electron_sip3d<4) &&"
                                     " All(abs(Electron_dxy)<0.5) && "
                                     " All(abs(Electron_dz)<1.0)",
                                     "Electron track close to primary vertex")
    
    mu_ip3d = "sqrt(Muon_dxy*Muon_dxy + Muon_dz*Muon_dz)"
    df_mu_ip3d = df_el_track.Define("Muon_ip3d", mu_ip3d)
    mu_sip3d = "Muon_ip3d/sqrt(Muon_dxyErr*Muon_dxyErr + Muon_dzErr*Muon_dzErr)"
    df_mu_sip3d = df_mu_ip3d.Define("Muon_sip3d", mu_sip3d)
    df_mu_track = df_mu_sip3d.Filter("All(Muon_sip3d<4) && All(abs(Muon_dxy)<0.5) &&"
                                     "All(abs(Muon_dz)<1.0)",
                                     "Muon track close to primary vertex")
    df_2p2n = df_mu_track.Filter("Sum(Electron_charge) == 0 && Sum(Muon_charge) == 0",
                                 "Two opposite charged electron and muon pairs")
    return df_2p2n

def reco_higgs(df, weight):
    """Recontruction of the Higgs mass
    """
    #Selection of only the potential good events 
    df_base = good_events(df)
    
    #Compute z masses from it
    df_z_mass = df_base.Define("Z_mass", "z_mass(Electron_pt, Electron_eta, Electron_phi, Electron_mass, Muon_pt, Muon_eta, Muon_phi, Muon_mass)")
    #Filter on z masses
    df_z1 = df_z_mass.Filter("Z_mass[0] > 40 && Z_mass[0] < 120", "First candidate in [40, 120]") 
    df_z2 = df_z1.Filter("Z_mass[1] > 12 && Z_mass[1] < 120", "Second candidate in [12, 120]")

    #Reconstruct H mass
    df_reco_h = df_z2.Define("H_mass", "h_mass(Electron_pt, Electron_eta, Electron_phi, Electron_mass, Muon_pt, Muon_eta, Muon_phi, Muon_mass)")

    h_reco_h = df_reco_h.Define("weight", f"{weight}")\
                        .Histo1D(("h_sig_2el2mu", "", 36, 70, 180), "H_mass", "weight")
    report_higgs = df_reco_h.Report()    
    return h_reco_h, report_higgs

def standard_retrieve (filters, h_higgs, rep_higgs):
    """If the code is in only standard mode, here the event loop will be triggered
    """
    dict_cut = OrderedDict(Eta=(0, 2), Dr=(2, 4), Pt=(4, 6), Isolation=(6, 8), 
                           Electron_track=(8, 11), Muon_track=(11, None))    

    # Take the same structure that will be filled with actual data
    filters_data = {'sig':{'h_unfil':[], 'h_fil':[], 'rep':[]},
                    'bkg':{'h_unfil':[], 'h_fil':[], 'rep':[]}}
    # First step: iterate on (sig,(huf, hf, rep)) and (bkg,(huf, hf, rep))
    for ch, ch_dict in filters.items():
        # Second step: iterate on (hf, list), (huf, list), (rep, list)
        for list_name, list_el in ch_dict.items():
            #Third step: iterate on list elements and retrieve data
            for elem in list_el:
                filters_data[ch][list_name].append(elem.GetValue())
    
    for cut in dict_cut.keys():
        start, end = dict_cut[cut]
        h_uf_s = filters_data['sig']['h_unfil'][start:end]
        h_f_s = filters_data['sig']['h_fil'][start:end]
        h_uf_b = filters_data['bkg']['h_unfil'][start:end]
        h_f_b = filters_data['bkg']['h_fil'][start:end]
        filtered_plot(h_uf_s, h_uf_b, h_f_s, h_f_b, cut)
    
    #Print, for now o screen, of the stats for each applied filter
    # [rep.Print() for rep in rep_fil]

    # #Series of instructions to retriev and plot higgs mass
    list_higgs_data = []
    
    for h in h_higgs:
        h_higgs_data = h.GetValue()
        list_higgs_data.append(h_higgs_data)
    
    higgs_plot(list_higgs_data)
    [(rep.Print(), print("")) for rep in rep_higgs]

def preliminar_plot(df, branch_histo, histo_data):
    """For now it's just a simple plot of unprocessed data
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
    logging.info(f'One day minumun and maximun of the {branch_histo}'
                 f' will be useful...but it is NOT This day!{x_max,x_min}')
    # histo_data.GetXaxis().SetRangeUser(x_min,x_max)
    
    histo_data.SetLineStyle(1)
    histo_data.SetLineWidth(1)
    histo_data.SetLineColor(ROOT.kBlack)
    histo_data.Draw()

    # Add Legend
    legend = ROOT.TLegend(0.7, 0.6, 0.85, 0.75)
    legend.SetFillColor(0)
    legend.SetBorderSize(1)
    legend.SetTextSize(0.04)
    legend.AddEntry(histo_data,df)
    legend.Draw()

    #Save plot
    filename = f'{branch_histo}_{df}.pdf'
    c_histo.SaveAs(filename)

def filtered_plot(histo_unfil_s, histo_unfil_b, histo_fil_s, histo_fil_b, fil):
    """Plot of the filtered data versus the unprocessed ones"""
    # Add canvas
    canvas = ROOT.TCanvas("canvas","",800,700)
    canvas.cd()

    # Add Legend
    legend = ROOT.TLegend(0.7, 0.6, 0.85, 0.75)
    legend.SetFillColor(0)
    legend.SetBorderSize(1)
    legend.SetTextSize(0.04)

    delta_y = 1/len(histo_fil_s)

    for i in range(len(histo_fil_s)):
        canvas.cd()
        h_ustyle_s = style(histo_unfil_s[i],'e', ROOT.kRed)
        h_fstyle_s = style(histo_fil_s[i],'f', ROOT.kRed)
        h_ustyle_b = style(histo_unfil_b[i],'e', ROOT.kAzure)
        h_fstyle_b = style(histo_fil_b[i],'f', ROOT.kAzure)
        pad = ROOT.TPad(f"pad_{i}", f"pad_{i}", 0, i*delta_y, 1, (1+i)*delta_y)
        pad.Draw()
        pad.cd()
        h_ustyle_b.Draw()
        h_fstyle_b.Draw("SAME")
        h_ustyle_s.Draw("SAME")
        h_fstyle_s.Draw("SAME")
    
    # print(list_histo_fil[0].GetXaxis.GetTitle())
    
    legend.AddEntry(histo_unfil_s[0],"Signal Unfilterd Data")
    legend.AddEntry(histo_fil_s[0],"Signal Filtered Data")
    legend.AddEntry(histo_unfil_b[0],"Background Unfilterd Data")
    legend.AddEntry(histo_fil_b[0],"Background Filtered Data")
    legend.Draw()

    # pad1.SetBottomMargin(0)    
    # pad2.SetTopMargin(0)
    # pad2.SetBottomMargin(0)

    #Save plot
    canvas.SaveAs(f'{fil}.pdf')

def higgs_plot(list_histo_higgs):
    """Plot reconstructed Higgs mass for signal, background and data
    """
    # Add canvas
    canvas_s = ROOT.TCanvas("canvas_s","",800,700)

    h_signal = style(list_histo_higgs[0], 'e', ROOT.kRed)
    h_background = style(list_histo_higgs[1], 'f', ROOT.kAzure)
    h_data = style(list_histo_higgs[2], 'm', ROOT.kBlack)

    canvas_s.cd()
    h_background.Draw("HIST")
    h_signal.Draw("HIST SAME")
    h_data.Draw("PE1 SAME")
    
    # print(list_histo_fil[0].GetXaxis.GetTitle())
    # Add Legend
    legend = ROOT.TLegend(0.7, 0.6, 0.85, 0.75)
    legend.SetFillColor(0)
    legend.SetBorderSize(1)
    legend.SetTextSize(0.04)    
    legend.AddEntry(list_histo_higgs[0], "Signal")
    legend.AddEntry(list_histo_higgs[1], "Background")
    legend.AddEntry(list_histo_higgs[2], "Data")
    legend.Draw()

    #Save plot
    canvas_s.SaveAs('higgs_mass.pdf')

def ml_request(df_sig, df_bkg):
    """Prepare dataset to be readable by the machine learning  method
    """
    ml_var = ["Muon_pt_1", "Muon_pt_2", "Electron_pt_1", "Electron_pt_2",
              "Muon_mass_1", "Muon_mass_2","Electron_mass_1", "Electron_mass_2", 
              "Muon_eta_1", "Muon_eta_2", "Electron_eta_1", "Electron_eta_2",
              "Muon_phi_1", "Muon_phi_2", "Electron_phi_1", "Electron_phi_2","Higgs_mass"]
    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    snapshotOptions.fLazy = True
    list_df_train = []
    for df, df_key in [[df_sig, "signal"], [df_bkg, "background"]]:
        logging.info(f"Book the training and testing events for {df_key}")
        # Define the training variables
        df = df.Define("Muon_pt_1", "Muon_pt[0]").Define("Muon_pt_2", "Muon_pt[1]")\
               .Define("Muon_mass_1", "Muon_mass[0]").Define("Muon_mass_2", "Muon_mass[1]")\
               .Define("Electron_mass_1", "Electron_mass[0]").Define("Electron_mass_2", "Electron_mass[1]")\
               .Define("Electron_pt_1", "Electron_pt[0]").Define("Electron_pt_2", "Electron_pt[1]")\
               .Define("Muon_eta_1","Muon_eta[0]").Define("Muon_eta_2","Muon_eta[1]")\
               .Define("Electron_eta_1","Electron_eta[0]").Define("Electron_eta_2", "Electron_eta[1]")\
               .Define("Muon_phi_1", "Muon_phi[0]").Define("Muon_phi_2","Muon_phi[1]")\
               .Define("Electron_phi_1","Electron_phi[0]").Define("Electron_phi_2", "Electron_phi[1]")\
               .Define("Higgs_mass","h_mass(Electron_pt, Electron_eta, Electron_phi, Electron_mass, Muon_pt, Muon_eta, Muon_phi, Muon_mass)")

        # Split dataset by event number for training and testing
        columns = ROOT.std.vector("string")()
        for column in ml_var:
            columns.push_back(column)
        filename = f"train_{df_key}.root"
        df_train = df.Snapshot("Events", filename, columns, snapshotOptions)
        list_df_train.append(df_train)
        
    return list_df_train, ml_var

def ml_training_retrieve(list_df, variables):
    """Retrieving the dataset to train a machine learning model
    """
    # Read data from ROOT files, trigger event loop if not done before
    print("Columns in data_sig: ", list_df[0].GetColumnNames())
    data_sig = list_df[0].AsNumpy()
    data_bkg = list_df[1].AsNumpy()
    # Convert inputs to format readable by machine learning tools
    # T is the transposed vector
    x_sig = np.vstack([data_sig[var] for var in variables]).T
    x_bkg = np.vstack([data_bkg[var] for var in variables]).T
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

    #Scaling data (from here it's different from the ROOT tutorial)
    sc = StandardScaler()
    x_sc = sc.fit_transform(x)
    x_sh, y_sh = shuffle(x_sc, y)
    np.save("x_sh_unbalanced.npy", x_sh)
    np.save("y_sh_unbalanced.npy", y_sh)
    return x_sh, y_sh

# def ml_training(list_df, variables):
    
    # # Load data
    # x, y = ml_training_retrieve(list_df, variables)

    # # Neural network
    # model = Sequential()
    # model.add(Dense(16, input_dim))

if __name__ == '__main__':
    
    #monitor of the code run time
    start = time.time()

    #Set the standard mode of analysis
    perform_std = True

    #Possible options for different analysis
    parser = argparse.ArgumentParser(description=_description)
    parser.add_argument('-l', '--local',
                        help='perform analysis on local dataframe,'
                        ' download of the request data if not present',
                        action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-p', '--preliminar',
                        help='perform only preliminar analysis',
                        action="store_true")
    group.add_argument('-b', '--both',
                        help='perform also preliminar analysis',
                        action="store_true")
    args = parser.parse_args()

    #Check the chosen argparse options        
    if args.preliminar or args.both:   
        #In both cases we need the preliminary requests
        df_prel, branches_prel, histo_prel = preliminar_request(args.local)
        
        if args.preliminar:
            #Standard analysis is excluded
            perform_std = False
            logging.info('You have disabled standard analysis')
            #There are no other requests, event loop can be triggered
            preliminar_retrieve(df_prel, branches_prel, histo_prel)
        else:
            logging.info("It will pass to the requests for standard analysis")
            pass
    else:
        pass
    
    if perform_std: 
        #Standard analysis
        dict_fil, h_higgs, rep_higgs, df_train, ml_var = standard_request(args.local)
        if args.both:
            #The preliminary requests have already been done.
            #Let's go to the retrieving part
            preliminar_retrieve(df_prel, branches_prel, histo_prel)
            ml_training_retrieve(df_train, ml_var)
            standard_retrieve(dict_fil, h_higgs, rep_higgs)
        else:
            ml_training_retrieve(df_train, ml_var)
            standard_retrieve(dict_fil, h_higgs, rep_higgs)
            logging.info("You have chosen to perform only standard analysis")
    else:
        pass
    stop = time.time()
    logging.info(f'elapsed time using signal: {stop - start}\n')

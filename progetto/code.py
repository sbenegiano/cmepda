import argparse
import os
import sys
import ROOT
import logging
import time
import numpy

#Necessary option in order for argparse to function, if not present ROOT will 
#prevail over argparse when option are passed from command line
ROOT.PyConfig.IgnoreCommandLineOptions = True

logging.basicConfig(filename = "test.log", level = logging.DEBUG, 
                    format = '%(asctime)s %(message)s')
_description = 'The program will perform standard analysis if no option is given.'

#Include cpp library for RDataFrame modules
ROOT.gInterpreter.ProcessLine('#include "library.h"')

# Enable multi-threading
# ROOT.ROOT.EnableImplicitMT()

def style(h, mode):
    """Set the basic style of a given histogram depending on which mode is required
    """
    h.SetStats(0)
    h.SetLineStyle(1)
    h.SetLineWidth(1)

    if mode == 'u':
        h.SetLineColor(ROOT.kRed)

    elif mode == 'f':
        h.SetFillStyle(1001)
        h.SetFillColor(ROOT.kAzure)

    return h


def preliminar_request(df_s, df_b, df_d):
    """First look at the chosen input file:
    this function is meant to explore only first hand characteristics of events
    and by default it will show three variables that are stored per component
    """
    dict_df = {'signal':df_s, 'bakground':df_b, 'data':df_d}   
    key_df = input('Insert the data frame you want to look at first '
                         f'choosing from this list:\n{dict_df.keys()}\n')
    try:
        df_In = dict_df.get(key_df)
        list_branches = df_In.GetColumnNames()

        # decide which variables to look first
        dictOfBranches = {i:list_branches[i] for i in range (0, len(list_branches))}
        list_In = input('Insert the variable numbers to look at, separated by a space'
                        f'press return when you are done:\n{dictOfBranches}\n')
        
        #control input and retrieve the required branches
        list_str = list_In.split(" ")

        b_In = []
        h_In = []
        for i in list_str and i < 32:
            if i.isdigit():
                current_branch = dictOfBranches[int(i)]
                b_In.append(current_branch)
                current_histo = df_In.Histo1D(current_branch)
                h_In.append(current_histo)
            else:
                logging.warning(f'Error! {i} is an invalid key!')

        #3D reconstruction of some fundamental variables
        pv_3d = df_In.Define("PV_3d","sqrt(PV_x*PV_x + PV_y*PV_y + PV_z*PV_z)")
        mu_3d = df_In.Define("Muon_3d","sqrt(Muon_dxy*Muon_dxy + Muon_dz*Muon_dz)")
        el_3d = df_In.Define("El_3d","sqrt(Electron_dxy*Electron_dxy + Electron_dz*Electron_dz)")

        #Retrieve the relative histograms
        h_pv_3d = pv_3d.Histo1D("PV_3d")
        h_mu_3d = mu_3d.Histo1D("Muon_3d")
        h_el_3d = el_3d.Histo1D("El_3d")

        #Update of branches and histogram lists
        b_In.extend(['PV_3d', 'Muon_3d', 'El_3d'])
        h_In.extend([h_pv_3d, h_mu_3d, h_el_3d])

        logging.info(f'These are the branches you chose: {b_In}')

    except KeyError as e:
        print(f'Cannot read the given key!\n{e}')
        sys.exit(1)
    return key_df, b_In, h_In

def preliminar_retrieve(h_prel, b_prel, df_prel):
    """Retrieve and plot histos previously requested
    """
    #Trigger event loop and plot
    for hist, branch in zip(histo_prel, branches_prel):
        h = hist.GetValue()
        preliminar_plot(h, branch, df_prel)

def standard_request(df_s, df_b, df_d):
    """All the necessary requests for the standard analysis will be prepared:
    """
    #Two filters will be applied but not shown to reduce the number of processed data
    df_2e2m = df_s.Filter("nElectron>=2 && nMuon>=2",
                         "Only events with two or more Electrons and Muons")

    #Request filtered and unfiltered data
    list_h_unfil, list_h_fil, list_rep = show_cut(df_2e2m)
    dr_report = good_events(df_2e2m)
    return list_h_unfil, list_h_fil, list_rep, dr_report

def show_cut(df):
    """Comparison between unfiltered and filtered data considering the main cuts
    used in the analysis published on CERN Open Data
    """
    #Preparation of a list of filtered and unfiltered data to plot aferwards
    #And of a list of reports for each filter
    list_h_unfil = []
    list_h_fil = []
    list_report = []

    #1st filter:Eta cut
    h_unfil_eleta = df.Histo1D(("h_Eleta", "Electron_eta", 56, -2.6, 2.6), "Electron_eta")
    h_unfil_mueta = df.Histo1D(("h_Mueta", "Muon_eta", 56, -2.6, 2.6), "Muon_eta") 
    
    df_eleta = df.Filter("All(abs(Electron_eta)<2.5)", "Eleta cut")
    df_mueta = df.Filter("All(abs(Muon_eta)<2.4)", "Mueta cut")
    h_fil_eleta = df_eleta.Histo1D(("h_Eleta","", 56, -2.6, 2.6), "Electron_eta")
    h_fil_mueta = df_mueta.Histo1D(("h_Mueta","", 56, -2.6, 2.6), "Muon_eta")

    #2nd filter:Dr cut
    df_dr = df.Define("Electron_dr",
                             "dr_def(Electron_eta, Electron_phi)").Define("Muon_dr",
                             "dr_def(Muon_eta, Muon_phi)")
    h_unfil_eldr = df_dr.Histo1D(("h_eldr","Electron_dr", 56, -0.5, 6), "Electron_dr")
    h_unfil_mudr = df_dr.Histo1D(("h_mudr","Muon_dr", 56, -0.5, 6), "Muon_dr")

    df_eldr = df_dr.Filter("Electron_dr>=0.02","Eldr cut")
    df_mudr = df_dr.Filter("Muon_dr>=0.02","Mudr cut")
    h_fil_eldr = df_eldr.Histo1D(("h_eldr","Electron_dr", 56, -0.5, 6), "Electron_dr")
    h_fil_mudr = df_mudr.Histo1D(("h_mudr","Muon_dr", 56, -0.5, 6), "Muon_dr")

    #3rd filter:Pt cut
    h_unfil_elpt = df.Histo1D(("h_Elpt", "Electron_pt", 56, -0.5, 120), "Electron_pt")
    h_unfil_mupt = df.Histo1D(("h_Mupt", "Muon_pt", 56, -0.5, 120), "Muon_pt")

    df_pt = df.Filter("pt_cut(Muon_pt, Electron_pt)", "Pt cuts")
    h_fil_elpt = df_pt.Histo1D(("h_Elpt", "", 56, -0.5, 120), "Electron_pt")
    h_fil_mupt = df_pt.Histo1D(("h_Mupt", "", 56, -0.5, 120), "Muon_pt")

    #4th filter: Good isolation
    h_unfil_eliso3 = df.Histo1D(("h_eliso3","Electron_Iso3", 400, -1020, 50), "Electron_pfRelIso03_all")
    h_unfil_muiso4 = df.Histo1D(("h_muiso4","Muon_Iso4", 400, -1020, 50), "Muon_pfRelIso04_all")

    df_eliso = df.Filter("All(abs(Electron_pfRelIso03_all)<0.40)", "ElIso03 cut")
    df_muiso = df.Filter("All(abs(Muon_pfRelIso04_all)<0.40)", "MuIso04 cut")   
    h_fil_eliso3 = df_eliso.Histo1D(("h_eliso3","", 400, -1020, 50), "Electron_pfRelIso03_all")
    h_fil_muiso4 = df_muiso.Histo1D(("h_muiso4","", 400, -1020, 50), "Muon_pfRelIso04_all")

    #5th filter: Electron track
    el_sip3d = "sqrt(Electron_dxy*Electron_dxy + Electron_dz*Electron_dz)/sqrt(Electron_dxyErr*Electron_dxyErr+ Electron_dzErr*Electron_dzErr)"
    df_eltrack = df.Define("Electron_sip3d", el_sip3d)
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
    df_mutrack = df.Define("Muon_sip3d", mu_sip3d)
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
    return list_h_unfil, list_h_fil, list_report 

def good_events(df_2el2mu):
    """Selection of 2electrons and 2 muons
    that pass the cuts used in the 2012 CERN article
    """
    # selection of 2 of more electrons and muons, request good isolation

    df_iso = df_2el2mu.Filter("All(abs(Electron_pfRelIso03_all)<0.40) &&"
                              "All(abs(Muon_pfRelIso04_all)<0.40)",
                              "Require good isolation")

    #angular cuts
    df_eta = df_iso.Filter("All(abs(Electron_eta)<2.5) && All(abs(Muon_eta)<2.4)",
                           "Eta_cuts")
    df_dr = df_eta.Filter("dr_cut(Muon_eta, Muon_phi, Electron_eta, Electron_phi)",
                           "Dr_cuts")

    #transvers momenta cuts
    df_pt = df_dr.Filter("pt_cut(Muon_pt, Electron_pt)", "Pt cuts")
                           
    #Reconstruction and filter of Muon and Electron tracks
    el_ip3d = "sqrt(Electron_dxy*Electron_dxy + Electron_dz*Electron_dz)"
    mu_ip3d = "sqrt(Muon_dxy*Muon_dxy + Muon_dz*Muon_dz)"
    df_el_ip3d = df_pt.Define("Electron_ip3d", el_ip3d).Define("Muon_ip3d", mu_ip3d)
    el_sip3d = "Electron_ip3d/sqrt(Electron_dxyErr*Electron_dxyErr+ Electron_dzErr*Electron_dzErr)"
    mu_sip3d = "Muon_ip3d/sqrt(Muon_dxyErr*Muon_dxyErr + Muon_dzErr*Muon_dzErr)"
    df_el_sip3d = df_el_ip3d.Define("Electron_sip3d", el_sip3d).Define("Muon_sip3d", mu_sip3d)
    df_el_track = df_el_sip3d.Filter("All(Electron_sip3d<4) &&"
                                     " All(abs(Electron_dxy)<0.5) && "
                                     " All(abs(Electron_dz)<1.0)",
                                     "Electron track close to primary vertex")
    df_mu_track = df_el_track.Filter("All(Muon_sip3d<4) && All(abs(Muon_dxy)<0.5) &&"
                                     "All(abs(Muon_dz)<1.0)",
                                     "Muon track close to primary vertex")
    df_2p2n = df_mu_track.Filter("Sum(Electron_charge) == 0 && Sum(Muon_charge) == 0",
                                 "Two opposite charged electron and muon pairs")
    return df_2p2n.Report()

def standard_retrieve(h_unfil_std, h_fil_std, rep_std, dr_rep_std):
    """If the code is in only standard mode, here the event loop will be triggered
    """
    list_huf_data = []
    list_hf_data = []
    list_cut = ["Eta", "Dr", "Pt", "Isolation", "Electron track", "Muon track"]
    for h_uf, h_f in zip(h_unfil_std, h_fil_std):
        h_unfil_data = h_uf.GetValue()
        h_fil_data = h_f.GetValue()
        list_huf_data.append(h_unfil_data)
        list_hf_data.append(h_fil_data)

    #Plot filtered and unfiltered data for the Eta cut
    filtered_plot(list_huf_data[:2], list_hf_data[:2], list_cut[0])
    #Plot filtered and unfiltered data for the Dr cut
    filtered_plot(list_huf_data[2:4], list_hf_data[2:4], list_cut[1])
    #Plot filtered and unfiltered data for the Pt cut
    filtered_plot(list_huf_data[4:6], list_hf_data[4:6], list_cut[2])
    #Plot filtered and unfiltered data for the Isolation cut
    filtered_plot(list_huf_data[6:8], list_hf_data[6:8], list_cut[3])
    #Plot filtered and unfiltered data for the Electron track
    filtered_plot(list_huf_data[8:11], list_hf_data[8:11], list_cut[4])
    #Plot filtered and unfiltered data for the Muon track
    filtered_plot(list_huf_data[11:], list_hf_data[11:], list_cut[5])

    logging.info(f"{len(rep_std)+3} is the number of df requested")

    [rep.Print() for rep in rep_std]
    # dr_rep_std.Print()


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

def filtered_plot(list_histo_unfil, list_histo_fil, fil):
    """Plot of the filtered data versus the unprocessed ones"""
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(1)
    ROOT.gStyle.SetTextFont(42)
    canvas = ROOT.TCanvas("canvas","",800,700)
    canvas.cd()

    # Add Legend
    legend = ROOT.TLegend(0.7, 0.6, 0.85, 0.75)
    legend.SetFillColor(0)
    legend.SetBorderSize(1)
    legend.SetTextSize(0.04)

    delta_y = 1/len(list_histo_fil)

    for i in range(len(list_histo_fil)):
        
        canvas.cd()
        h_ustyle = style(list_histo_unfil[i],'u')
        h_fstyle = style(list_histo_fil[i],'f')
        pad = ROOT.TPad(f"pad_{i}", f"pad_{i}", 0, i*delta_y, 1, (1+i)*delta_y)
        pad.Draw()
        pad.cd()
        h_ustyle.Draw()
        h_fstyle.Draw("SAME")
    
    # print(list_histo_fil[0].GetXaxis.GetTitle())
    
    legend.AddEntry(list_histo_unfil[0],"Unfilterd Data")
    legend.AddEntry(list_histo_fil[0],"Filtered Data")
    legend.Draw()

    # pad1.SetBottomMargin(0)    
    # pad2.SetTopMargin(0)
    # pad2.SetBottomMargin(0)

    #Save plot
    canvas.SaveAs(f'{fil}.pdf')
    
if __name__ == '__main__':

    #monitor of the code run time
    start = time.time()

    #Set the standard mode of analysis
    perform_std = True

    #Possible options for different analysis
    parser = argparse.ArgumentParser(description=_description)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-p', '--preliminar',
                        help='perform only preliminar analysis',
                        action="store_true")
    group.add_argument('-b', '--both',
                        help='perform also preliminar analysis',
                        action="store_true")
    args = parser.parse_args()

    #Create the imput data frame which comprehends signal, background and
    #data samples

    #Signal of Higgs -> 4 leptons
    df_sig = ROOT.RDataFrame("Events",
                             "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/SMHiggsToZZTo4L.root")
    
    #Background of ZZ -> 4 leptons
    df_bkg = ROOT.RDataFrame("Events",
                             "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/ZZTo4e.root")

    #CMS data tacken in 2012 (11.6 fb^(-1) integrated luminosity)
    data_files = ROOT.std.vector("string")(2)
    data_files[0] = "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root"
    data_files[1] = "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012C_DoubleMuParked.root"

    df_data = ROOT.RDataFrame("Events", data_files)

    #Check the chosen argparse options    
    if args.preliminar or args.both:   
        #In both cases we need the preliminary requests
        df_prel, branches_prel, histo_prel = preliminar_request(df_sig, df_bkg, df_data)
        
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
        h_unfiltered, h_filtered, rep, dr_rep = standard_request(df_sig, df_bkg, df_data)

        if args.both:
            #The preliminary requests have already been done.
            #Let's go to the retrieving part

            preliminar_retrieve(df_prel, branches_prel, histo_prel)
            standard_retrieve(h_unfiltered, h_filtered, rep, dr_rep)
        else:
            standard_retrieve(h_unfiltered, h_filtered, rep, dr_rep)
            logging.info("You have chosen to perform only standard analysis")
    else:
        pass
    stop = time.time()
    logging.info(f'elapsed time using signal: {stop - start}\n')


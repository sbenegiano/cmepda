import argparse
import os
import ROOT
import logging
import time

logging.basicConfig(filename = "test.log", level = logging.DEBUG, 
format = '%(asctime)s %(message)s')

# Enable multi-threading
# ROOT.ROOT.EnableImplicitMT()

def preliminary_analysis(path_to_file):
    """first look at the input file
    this function is meant to explore only first hand characteristics of events
    """
    list_branches = dfIn.GetColumnNames()

    # decide which variables to look first
    dictOfBranches = { i:list_branches[i] for i in range (0,len(list_branches))}
    list_In = input('Insert the variable numbers to look at, separated by a space'
            f'press return when you are done \n {dictOfBranches}:')
        
    #control input and retrieve the required branches
    list_str = list_In.split(" ")
    branches_in = []
    for i in list_str:
        if i.isdigit():
            branches_in.append(dictOfBranches[int(i)])
        else: logging.warning(f'Error! {i} is an invalid key!')
    logging.info(f'These are the branches you chose: {branches_in}')
    return branches_in
    

def plot(histo_data, h_branch, filename):
    """For now it's just a simple plot of unprocessed data
    """
    #General Canvas Settings
    ROOT.gStyle.SetOptStat(11)
    ROOT.gStyle.SetTextFont(42)
    ROOT.gStyle.SetStatX(0.9)
    ROOT.gStyle.SetStatY(0.9)
    c_histo = ROOT.TCanvas("c_histo","",600,400)
    
    #Set and Draw histogram for data points
    x_label = h_branch
    x_max = histo_data.GetXaxis().GetXmax()
    x_min = histo_data.GetXaxis().GetXmin()
    logging.info(f'One day minumun and maximun of the {h_branch}'
     f' will be useful...but it is NOT This day!{x_max,x_min}\n')
    # histo_data.GetXaxis().SetRangeUser(x_min,x_max)

    histo_data.GetXaxis().SetTitle(x_label)
    # histo_data.GetYaxis().SetTitle("N_events")
    histo_data.SetLineStyle(0)
    histo_data.SetLineWidth(1)
    histo_data.SetLineColor(1)
    histo_data.SetMarkerStyle(20)
    histo_data.SetMarkerColor(ROOT.kBlack)
    histo_data.SetMarkerSize(0.5)
    histo_data.Draw()

    # Add Legend
    legend = ROOT.TLegend(0.75, 0.70, 0.88, 0.82)
    legend.SetFillColor(0)
    legend.SetBorderSize(1)
    legend.SetTextSize(0.04)
    legend.AddEntry("Data", "","D")
    legend.Draw()

    #Save plot
    c_histo.SaveAs(filename)

if __name__ == '__main__':
    
    #Insert input file from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='path to the input file')
    args = parser.parse_args()
    try:
        fInput = ROOT.TFile.Open(args.infile)
        mytree = fInput.Get("Events")
        dfIn = ROOT.RDataFrame(mytree)

        #Perform a preliminar analysis
        branch_names = preliminary_analysis(dfIn)
        for branch in branch_names:
            hist1 = dfIn.Histo1D(branch)
            plot(hist1.GetValue(), branch, branch+'.pdf')
        
        #3D reconstruction of some fundamental variables
        pv_3d = dfIn.Define("PV_3d","sqrt(PV_x*PV_x + PV_y*PV_y + PV_z+PV_z)")
        muon_3d = dfIn.Define("Muon_3d","sqrt(Muon_dxy*Muon_dxy + Muon_dz*Muon_dz)")
        el_3d = dfIn.Define("El_3d","sqrt(Electron_dxy*Electron_dxy + Electron_dz*Electron_dz)")

        h_pv_3d = pv_3d.Histo1D("PV_3d")
        h_muon_3d = muon_3d.Histo1D("Muon_3d")
        h_el_3d = el_3d.Histo1D("El_3d")

        plot(h_pv_3d.GetValue(), "PV_3d", "PV_3d.pdf")
        plot(h_muon_3d.GetValue(), "Muon_3d", "Muon_3d.pdf")
        plot(h_el_3d.GetValue(), "El_3d", "El_3d.pdf")


    except OSError as e:
        print(f'Cannot read the file!\n{e}') 
